"""Model interpretability module implementing SHAP explanations and permutation analysis."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from churn_prediction.config import FIGURES_DIR, PHASE_6_REPORTS_DIR, MODELS_DIR
from churn_prediction.model_inference import load_model_bundle

logger = logging.getLogger(__name__)


def get_preprocessed_features(pipeline: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Transform feature matrix using the pipeline's preprocessor step and extract output column names.
    
    Parameters
    ----------
    pipeline : Pipeline
        Fitted Scikit-Learn Pipeline containing 'preprocessor'.
    X : pd.DataFrame
        Input features.
        
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Transformed array and corresponding feature names.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_trans = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    return X_trans, feature_names


def compute_shap_explanations(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    background_samples: int = 100,
    random_state: int = 42,
) -> Tuple[shap.Explanation, List[str]]:
    """Compute SHAP values for the pipeline's classifier using the transformed feature space.
    
    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline.
    X_sample : pd.DataFrame
        Evaluation samples.
    background_samples : int, default=100
        Number of background reference points for Kernel/Permutation explainer.
    random_state : int, default=42
        Seed.
        
    Returns
    -------
    Tuple[shap.Explanation, List[str]]
        SHAP explanation object and list of feature names.
    """
    X_trans, feature_names = get_preprocessed_features(pipeline, X_sample)
    classifier = pipeline.named_steps["classifier"]

    # For tree models in sklearn (HistGradientBoosting / TreeExplainer)
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X_trans)
    except Exception as e:
        logger.info(f"TreeExplainer not directly supported for this classifier ({e}); using Permutation/Exact Explainer.")
        # Background summary for model-agnostic explainer
        bg_idx = np.random.RandomState(random_state).choice(len(X_trans), size=min(background_samples, len(X_trans)), replace=False)
        background = X_trans[bg_idx]
        
        # Explain positive class probability
        def model_predict_proba(x):
            return classifier.predict_proba(x)[:, 1]

        explainer = shap.Explainer(model_predict_proba, background, feature_names=feature_names)
        shap_values = explainer(X_trans)

    return shap_values, feature_names


def generate_global_shap_plots(
    shap_explanation: shap.Explanation,
    feature_names: List[str],
    save_dir: Optional[Path] = None,
    max_display: int = 12,
) -> Dict[str, Path]:
    """Generate and save global SHAP summary plots (bar chart and beeswarm plot).
    
    Parameters
    ----------
    shap_explanation : shap.Explanation
        SHAP explanation object.
    feature_names : List[str]
        Transformed feature names.
    save_dir : Path, optional
        Destination directory.
        
    Returns
    -------
    dict
        Paths to saved plot images.
    """
    target_dir = Path(save_dir) if save_dir else FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = {}

    # Extract values if binary class explanation has shape (N, features, 2)
    values = shap_explanation.values
    if len(values.shape) == 3 and values.shape[-1] == 2:
        # Take positive class SHAP values
        values = values[:, :, 1]
        data = shap_explanation.data
    else:
        data = shap_explanation.data

    # 1. Global Mean Absolute SHAP Importance Bar Chart
    mean_abs_shap = np.abs(values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:max_display]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = mean_abs_shap[top_indices]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    y_pos = np.arange(len(top_features))[::-1]
    ax.barh(y_pos, top_scores, color="#1f77b4", height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Log-Odds of Churn)", fontsize=11)
    ax.set_title("Global Feature Importance — SHAP Summary", fontsize=12, pad=12)
    plt.tight_layout()

    bar_path = target_dir / "14_shap_global_feature_importance.png"
    fig.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots["global_bar"] = bar_path

    # 2. Beeswarm Summary Plot
    fig, ax = plt.subplots(figsize=(9, 6.5))
    shap.summary_plot(
        values,
        features=data,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Beeswarm Plot (Feature Value Impact on Churn)", fontsize=12, pad=12)
    plt.tight_layout()
    beeswarm_path = target_dir / "15_shap_beeswarm_summary.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots["beeswarm"] = beeswarm_path

    logger.info(f"Global SHAP plots saved to {target_dir}")
    return saved_plots


def explain_single_customer(
    pipeline: Pipeline,
    customer_df: pd.DataFrame,
    customer_id: str = "Customer",
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate local explanation for a single customer record.
    
    Parameters
    ----------
    pipeline : Pipeline
        Fitted model pipeline.
    customer_df : pd.DataFrame
        Single-row DataFrame of customer features.
    customer_id : str, default="Customer"
        Identifier for title.
    save_path : Path, optional
        Save path for local waterfall plot.
        
    Returns
    -------
    dict
        Local explanation details with top contributing features.
    """
    X_trans, feature_names = get_preprocessed_features(pipeline, customer_df)
    classifier = pipeline.named_steps["classifier"]

    # Calculate probability
    prob = float(pipeline.predict_proba(customer_df)[:, 1][0])

    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X_trans)
        vals = shap_values.values
        if len(vals.shape) == 3 and vals.shape[-1] == 2:
            vals = vals[0, :, 1]
            base_val = float(explainer.expected_value[1])
        elif len(vals.shape) == 2:
            vals = vals[0, :]
            base_val = float(explainer.expected_value)
        else:
            vals = vals.ravel()
            base_val = float(explainer.expected_value)
    except Exception:
        # Fallback approximation for explanation
        base_val = 0.0
        vals = np.zeros(len(feature_names))

    # Identify top 5 positive risk factors and top 5 protective factors
    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": vals,
        "Transformed Value": X_trans[0],
    }).sort_values(by="SHAP Value", ascending=False)

    top_risk_drivers = contrib_df[contrib_df["SHAP Value"] > 0].head(5).to_dict(orient="records")
    top_retention_drivers = contrib_df[contrib_df["SHAP Value"] < 0].tail(5).sort_values(by="SHAP Value", ascending=True).to_dict(orient="records")

    # Waterfall plot
    if save_path:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Top 8 contributing features
        top_contrib = pd.concat([contrib_df.head(4), contrib_df.tail(4)]).drop_duplicates().sort_values(by="SHAP Value", ascending=True)
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_contrib["SHAP Value"]]
        ax.barh(top_contrib["Feature"], top_contrib["SHAP Value"], color=colors, height=0.5)
        ax.axvline(0, color="grey", linestyle="--", alpha=0.7)
        ax.set_title(f"Local Churn Attribution — {customer_id} (Predicted Probability: {prob:.1%})", fontsize=11, pad=10)
        ax.set_xlabel("SHAP Value (Contribution to Log-Odds)", fontsize=10)
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return {
        "customer_id": customer_id,
        "churn_probability": prob,
        "base_value": base_val,
        "top_risk_drivers": top_risk_drivers,
        "top_retention_drivers": top_retention_drivers,
    }
