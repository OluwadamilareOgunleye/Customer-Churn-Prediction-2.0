"""Data drift detection and distribution stability monitoring module."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from churn_prediction.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between baseline and production distributions.
    
    Parameters
    ----------
    expected : np.ndarray
        Baseline/training values.
    actual : np.ndarray
        Production/monitoring values.
    num_bins : int, default=10
        Number of quantiles.
    epsilon : float, default=1e-4
        Small offset to prevent division by zero or log(0).
        
    Returns
    -------
    float
        PSI score.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    # Count occurrences in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to fractions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Apply epsilon smoothing
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    # PSI Formula: sum((Actual - Expected) * ln(Actual / Expected))
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_val, 4))


def evaluate_numerical_drift(
    ref_col: pd.Series,
    cur_col: pd.Series,
) -> Dict[str, Any]:
    """Evaluate drift on a single continuous numerical feature using PSI and KS-test.
    
    Returns
    -------
    dict
        Drift metrics including PSI, KS statistic, p-value, and drift status.
    """
    ref_vals = ref_col.dropna().values
    cur_vals = cur_col.dropna().values

    psi = calculate_psi(ref_vals, cur_vals)
    ks_res = stats.ks_2samp(ref_vals, cur_vals)

    # Determine status based on standard PSI benchmarks
    if psi < 0.10:
        status = "Stable"
    elif psi < 0.25:
        status = "Moderate Drift"
    else:
        status = "Significant Drift"

    return {
        "psi": psi,
        "ks_statistic": float(np.round(ks_res.statistic, 4)),
        "p_value": float(np.round(ks_res.pvalue, 4)),
        "status": status,
        "ref_mean": float(np.round(np.mean(ref_vals), 2)),
        "cur_mean": float(np.round(np.mean(cur_vals), 2)),
    }


def evaluate_categorical_drift(
    ref_col: pd.Series,
    cur_col: pd.Series,
) -> Dict[str, Any]:
    """Evaluate drift on a categorical feature by comparing category proportions."""
    ref_props = ref_col.value_counts(normalize=True)
    cur_props = cur_col.value_counts(normalize=True)

    all_cats = list(set(ref_props.index).union(set(cur_props.index)))
    ref_vec = np.array([ref_props.get(c, 1e-4) for c in all_cats])
    cur_vec = np.array([cur_props.get(c, 1e-4) for c in all_cats])

    # Categorical PSI
    psi = np.sum((cur_vec - ref_vec) * np.log(cur_vec / ref_vec))
    psi = float(np.round(psi, 4))

    status = "Stable" if psi < 0.10 else ("Moderate Drift" if psi < 0.25 else "Significant Drift")

    return {
        "psi": psi,
        "status": status,
        "category_shifts": {
            c: {"baseline_pct": float(round(ref_props.get(c, 0) * 100, 2)), "current_pct": float(round(cur_props.get(c, 0) * 100, 2))}
            for c in all_cats
        },
    }


def run_drift_audit(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Execute complete distribution drift audit comparing reference training data against production batch.
    
    Parameters
    ----------
    reference_df : pd.DataFrame
        Baseline training dataset.
    current_df : pd.DataFrame
        Production batch dataset.
        
    Returns
    -------
    dict
        Comprehensive drift audit report.
    """
    results: Dict[str, Any] = {
        "reference_records": len(reference_df),
        "current_records": len(current_df),
        "numerical_drift": {},
        "categorical_drift": {},
        "alerts": [],
    }

    # Audit numericals
    for col in NUMERICAL_FEATURES:
        if col in reference_df.columns and col in current_df.columns:
            m = evaluate_numerical_drift(reference_df[col], current_df[col])
            results["numerical_drift"][col] = m
            if m["status"] != "Stable":
                results["alerts"].append(f"Numerical feature '{col}' exhibits {m['status']} (PSI: {m['psi']:.4f})")

    # Audit categoricals
    for col in CATEGORICAL_FEATURES:
        if col in reference_df.columns and col in current_df.columns:
            m = evaluate_categorical_drift(reference_df[col], current_df[col])
            results["categorical_drift"][col] = m
            if m["status"] != "Stable":
                results["alerts"].append(f"Categorical feature '{col}' exhibits {m['status']} (PSI: {m['psi']:.4f})")

    results["overall_status"] = "Drift Detected" if results["alerts"] else "All Features Stable"
    return results
