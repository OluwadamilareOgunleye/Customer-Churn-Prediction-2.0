"""
Unit tests for Advanced Phases 15–19 capabilities:
- MLflow & Model Registry
- Calibration & Robustness
- Artifact Storage Abstraction
- Customer Segmentation & Business ROI
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from churn_prediction.model_registry import ModelRegistry
from churn_prediction.artifact_storage import LocalArtifactStore, get_artifact_store
from churn_prediction.calibration import compute_calibration_curve, evaluate_threshold_sweep
from churn_prediction.robustness import evaluate_subgroup_performance
from churn_prediction.customer_segmentation import segment_customers_kmeans, assign_churn_risk_tiers
from churn_prediction.business_impact import calculate_retention_roi


def test_model_registry_lifecycle(tmp_path):
    registry_file = tmp_path / "test_registry.json"
    registry = ModelRegistry(registry_file)

    # Register candidate
    meta = registry.register_model(
        model_name="test_model",
        model_type="LogisticRegression",
        decision_threshold=0.30,
        metrics={"roc_auc": 0.82},
        artifact_path=tmp_path / "model.joblib",
        stage="Staging"
    )
    assert meta.version == 1
    assert meta.status == "Staging"

    # Check retrieve
    versions = registry.list_versions("test_model")
    assert len(versions) == 1
    assert versions[0].status == "Staging"

    # Transition to Production
    prod_meta = registry.transition_stage("test_model", version=1, new_stage="Production")
    assert prod_meta.status == "Production"

    prod = registry.get_production_model("test_model")
    assert prod is not None
    assert prod.version == 1
    assert prod.status == "Production"


def test_artifact_storage_local(tmp_path):
    store = LocalArtifactStore()
    test_obj = {"alpha": 123, "beta": "churn"}
    test_file = str(tmp_path / "test_bundle.joblib")

    saved_uri = store.save_artifact(test_obj, test_file)
    assert Path(saved_uri).exists()

    loaded = store.load_artifact(test_file)
    assert loaded["alpha"] == 123
    assert loaded["beta"] == "churn"


def test_calibration_and_threshold_sweep():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    prob_true, prob_pred, brier = compute_calibration_curve(y_true, y_prob, n_bins=3)
    assert 0.0 <= brier <= 1.0

    df_sweep = evaluate_threshold_sweep(y_true, y_prob, thresholds=[0.3, 0.5])
    assert len(df_sweep) == 2
    assert "recall" in df_sweep.columns
    assert "precision" in df_sweep.columns


def test_subgroup_performance_metrics():
    df = pd.DataFrame({
        "Contract": ["Month-to-month", "Month-to-month", "Two year", "Two year"],
        "y_true": [1, 0, 0, 0],
        "y_prob": [0.8, 0.2, 0.1, 0.05]
    })
    res = evaluate_subgroup_performance(df, "Contract", "y_true", "y_prob", threshold=0.30)
    assert len(res) == 2
    assert "churn_rate" in res.columns


def test_customer_segmentation_and_risk_tiers():
    df = pd.DataFrame({
        "Tenure Months": [2, 10, 48, 60, 5],
        "Monthly Charges": [25.0, 85.0, 95.0, 20.0, 70.0],
        "Total Charges": [50.0, 850.0, 4500.0, 1200.0, 350.0]
    })
    kmeans, df_seg, sil = segment_customers_kmeans(df, n_clusters=2)
    assert "customer_segment" in df_seg.columns
    assert "persona_name" in df_seg.columns

    probs = np.array([0.05, 0.20, 0.45, 0.75])
    tiers = assign_churn_risk_tiers(probs)
    assert list(tiers) == ["Low Risk", "Moderate Risk", "Elevated Risk", "High Risk"]


def test_business_roi_financial_calculation():
    roi = calculate_retention_roi(
        tp=100, fp=50, fn=20, tn=300,
        customer_lifetime_value=1000.0,
        campaign_cost_per_customer=50.0,
        intervention_success_rate=0.20
    )
    assert roi["targeted_customers"] == 150
    assert roi["campaign_cost_usd"] == 7500.0
    assert roi["saved_churners"] == 20
    assert roi["gross_revenue_saved_usd"] == 20000.0
    assert roi["net_profit_usd"] == 12500.0
    assert roi["roi_percent"] > 0
