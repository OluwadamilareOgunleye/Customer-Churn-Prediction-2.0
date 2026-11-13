"""
Model Probability Calibration and Reliability Analysis.

This module provides tools to assess probability calibration (calibration curve, Brier score),
fit post-hoc calibrators (Platt Scaling / Logistic, Isotonic Regression) using cross-validation,
and analyze decision threshold tradeoffs without leaking holdout test sets.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute calibration curve points and Brier score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins.
    strategy : str
        'uniform' or 'quantile'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        prob_true, prob_pred, brier_score
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    brier = float(brier_score_loss(y_true, y_prob))
    return prob_true, prob_pred, brier


def evaluate_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = None
) -> pd.DataFrame:
    """
    Evaluate binary classification metrics across a sweep of probability thresholds.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]

    results = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        results.append({
            "threshold": t,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "fpr": round(float(fpr), 4),
            "fnr": round(float(fnr), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        })

    return pd.DataFrame(results)


def plot_calibration_comparison(
    uncalibrated_data: Tuple[np.ndarray, np.ndarray, float, str],
    calibrated_data: Tuple[np.ndarray, np.ndarray, float, str] = None,
    save_path: str = None
) -> plt.Figure:
    """
    Plot calibration curves comparing uncalibrated vs calibrated model against perfect calibration line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y=x)", alpha=0.7)

    # Plot uncalibrated
    prob_true_uncal, prob_pred_uncal, brier_uncal, label_uncal = uncalibrated_data
    ax.plot(
        prob_pred_uncal, prob_true_uncal, "s-", color="#1f77b4",
        label=f"{label_uncal} (Brier: {brier_uncal:.4f})"
    )

    if calibrated_data is not None:
        prob_true_cal, prob_pred_cal, brier_cal, label_cal = calibrated_data
        ax.plot(
            prob_pred_cal, prob_true_cal, "o-", color="#2ca02c",
            label=f"{label_cal} (Brier: {brier_cal:.4f})"
        )

    ax.set_ylabel("Fraction of Positives (Empirical Churn Rate)", fontsize=11)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_title("Probability Calibration Reliability Curve", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
