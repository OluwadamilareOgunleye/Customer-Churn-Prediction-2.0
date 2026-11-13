"""Model evaluation, metrics computation, and visualization utilities."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn_prediction.config import FIGURES_DIR


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Union[float, int, np.ndarray]]:
    """Compute comprehensive classification metrics for a binary classifier.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for positive class (1).
    threshold : float, default=0.5
        Decision threshold for converting probabilities to binary classes.
        
    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    return metrics


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluate precision, recall, and F1 across a sequence of classification thresholds.
    
    Parameters
    ----------
    y_true : array-like
        True ground truth labels.
    y_prob : array-like
        Predicted positive-class probabilities.
    thresholds : array-like, optional
        Threshold grid. Defaults to np.linspace(0.05, 0.95, 91).
        
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by threshold with precision, recall, f1, tp, fp, fn, tn.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    records = []
    for t in thresholds:
        m = evaluate_predictions(y_true, y_prob, threshold=t)
        records.append(m)

    return pd.DataFrame(records)


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot and save an annotated confusion matrix heatmap.
    
    Parameters
    ----------
    cm : np.ndarray
        2x2 confusion matrix array.
    model_name : str
        Model title.
    save_path : Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual Label", fontsize=11, labelpad=8)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot ROC curve and Precision-Recall curve side-by-side.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities.
    model_name : str
        Model name.
    save_path : Path, optional
        Save path.
        
    Returns
    -------
    plt.Figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    no_skill = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Baseline")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[0].set_title(f"ROC Curve — {model_name}", fontsize=12)
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # PR Curve
    axes[1].plot(rec, prec, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc:.3f})")
    axes[1].axhline(y=no_skill, color="grey", lw=1.5, linestyle="--", label=f"Baseline ({no_skill:.3f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title(f"Precision-Recall Curve — {model_name}", fontsize=12)
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_threshold_tradeoffs(
    sweep_df: pd.DataFrame,
    optimal_threshold: float,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot precision, recall, and F1 across classification thresholds.
    
    Parameters
    ----------
    sweep_df : pd.DataFrame
        DataFrame generated by sweep_thresholds.
    optimal_threshold : float
        Selected optimal threshold to highlight.
    save_path : Path, optional
        Save path.
        
    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color="#2ca02c", lw=2)
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", color="#d62728", lw=2)
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1-Score", color="#1f77b4", lw=2.2)

    ax.axvline(
        x=optimal_threshold,
        color="black",
        linestyle="--",
        lw=1.5,
        label=f"Selected Threshold ({optimal_threshold:.2f})",
    )

    ax.set_title("Precision, Recall, and F1 vs. Classification Threshold", fontsize=13, pad=12)
    ax.set_xlabel("Classification Decision Threshold", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
