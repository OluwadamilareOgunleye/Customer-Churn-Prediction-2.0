"""
Model Robustness, Subgroup Performance, and Sensitivity Analysis.

This module provides tools to stress-test churn prediction pipelines:
- Subgroup performance auditing (disaggregated precision, recall, sample size)
- Missing value tolerance and edge-case resilience testing
- Numerical extreme stress-testing
- Unseen categorical handling
- Single-feature sensitivity perturbations
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score


def evaluate_subgroup_performance(
    df: pd.DataFrame,
    subgroup_col: str,
    y_true_col: str,
    y_prob_col: str,
    threshold: float = 0.30
) -> pd.DataFrame:
    """
    Compute performance metrics across distinct slices of a categorical attribute.
    """
    results = []
    for group_name, group_data in df.groupby(subgroup_col):
        n_samples = len(group_data)
        y_true = group_data[y_true_col].values
        y_prob = group_data[y_prob_col].values
        n_churn = int(y_true.sum())
        base_rate = float(y_true.mean())

        # If group has only 1 class, ROC-AUC is undefined
        if len(np.unique(y_true)) > 1:
            try:
                auc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                auc = np.nan
        else:
            auc = np.nan

        y_pred = (y_prob >= threshold).astype(int)
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        results.append({
            "subgroup_column": subgroup_col,
            "group": str(group_name),
            "sample_count": n_samples,
            "churn_count": n_churn,
            "churn_rate": round(base_rate, 4),
            "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4)
        })

    return pd.DataFrame(results).sort_values(by="sample_count", ascending=False)


def run_missing_value_stress_test(
    pipeline: Any,
    sample_df: pd.DataFrame,
    features_to_null: List[str]
) -> pd.DataFrame:
    """
    Test how predictions change when individual or combinations of features are set to null/NaN.
    """
    baseline_probs = pipeline.predict_proba(sample_df)[:, 1]
    results = []

    for feat in features_to_null:
        if feat not in sample_df.columns:
            continue
        perturbed = sample_df.copy()
        perturbed[feat] = np.nan
        try:
            perturbed_probs = pipeline.predict_proba(perturbed)[:, 1]
            diff = np.abs(perturbed_probs - baseline_probs)
            results.append({
                "feature_perturbed": feat,
                "perturbation_type": "Set to NaN",
                "status": "Handled Successfully",
                "mean_abs_prob_delta": round(float(np.mean(diff)), 4),
                "max_abs_prob_delta": round(float(np.max(diff)), 4)
            })
        except Exception as e:
            results.append({
                "feature_perturbed": feat,
                "perturbation_type": "Set to NaN",
                "status": f"Failed: {str(e)}",
                "mean_abs_prob_delta": None,
                "max_abs_prob_delta": None
            })

    return pd.DataFrame(results)


def run_numerical_extreme_stress_test(
    pipeline: Any,
    sample_record: Dict[str, Any],
    extreme_cases: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Stress-test single records under extreme numerical edge-cases.
    """
    results = []
    for case in extreme_cases:
        test_dict = sample_record.copy()
        test_dict.update(case["updates"])
        df_test = pd.DataFrame([test_dict])
        try:
            prob = float(pipeline.predict_proba(df_test)[0, 1])
            results.append({
                "scenario": case["name"],
                "input_values": str(case["updates"]),
                "status": "Success",
                "predicted_churn_prob": round(prob, 4),
                "churn_flag_30": int(prob >= 0.30)
            })
        except Exception as e:
            results.append({
                "scenario": case["name"],
                "input_values": str(case["updates"]),
                "status": f"Error: {str(e)}",
                "predicted_churn_prob": None,
                "churn_flag_30": None
            })

    return pd.DataFrame(results)


def run_feature_sensitivity_analysis(
    pipeline: Any,
    base_customer: Dict[str, Any],
    feature_name: str,
    test_values: List[Any]
) -> pd.DataFrame:
    """
    Perform 1D sensitivity curve on a single customer profile across varying values of a feature.
    """
    results = []
    for val in test_values:
        cust = base_customer.copy()
        cust[feature_name] = val
        df_cust = pd.DataFrame([cust])
        prob = float(pipeline.predict_proba(df_cust)[0, 1])
        results.append({
            "feature": feature_name,
            "value": val,
            "predicted_prob": round(prob, 4),
            "churn_pred_30": int(prob >= 0.30)
        })
    return pd.DataFrame(results)
