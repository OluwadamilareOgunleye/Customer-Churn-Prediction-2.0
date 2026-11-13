"""
Unit and integration tests for System Reliability, Resilience, and Health Probes.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from api.main import app
from churn_prediction.health import HealthChecker
from churn_prediction.model_registry import ModelRegistry


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_liveness_probe(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_readiness_probe_success(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is True
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["decision_threshold"] == 0.30
    assert data["available_memory_mb"] > 0
    assert len(data["issues"]) == 0


def test_readiness_probe_missing_model(tmp_path):
    missing_path = tmp_path / "non_existent_bundle.joblib"
    diag = HealthChecker.check_readiness(missing_path)
    assert diag["ready"] is False
    assert diag["status"] == "not_ready"
    assert any("not found" in issue for issue in diag["issues"])


def test_readiness_probe_corrupted_model(tmp_path):
    corrupt_path = tmp_path / "corrupt_bundle.joblib"
    with open(corrupt_path, "w") as f:
        f.write("corrupted garbage text")

    diag = HealthChecker.check_readiness(corrupt_path)
    assert diag["ready"] is False
    assert any("Corrupted model bundle" in issue for issue in diag["issues"])


def test_model_registry_rollback_procedure(tmp_path):
    """
    Test rapid rollback to previous production version when candidate model fails.
    """
    registry_file = tmp_path / "rollback_registry.json"
    registry = ModelRegistry(registry_file)

    # Version 1: Initial Stable Production Model
    registry.register_model(
        model_name="churn_booster",
        model_type="HistGradientBoosting",
        decision_threshold=0.30,
        metrics={"roc_auc": 0.857},
        artifact_path=tmp_path / "v1.joblib",
        stage="Production"
    )
    v1_prod = registry.get_production_model("churn_booster")
    assert v1_prod.version == 1
    assert v1_prod.status == "Production"

    # Version 2: Flawed Candidate promoted
    registry.register_model(
        model_name="churn_booster",
        model_type="HistGradientBoosting",
        decision_threshold=0.30,
        metrics={"roc_auc": 0.810},
        artifact_path=tmp_path / "v2.joblib",
        stage="Production"
    )
    v2_prod = registry.get_production_model("churn_booster")
    assert v2_prod.version == 2

    # Verify v1 was automatically archived
    all_v = registry.list_versions("churn_booster")
    assert all_v[0].status == "Archived"

    # ROLLBACK: Immediately reinstate Version 1 as Production
    registry.transition_stage("churn_booster", version=1, new_stage="Production")

    rolled_back_prod = registry.get_production_model("churn_booster")
    assert rolled_back_prod.version == 1
    assert rolled_back_prod.status == "Production"

    # Verify v2 is now archived
    all_v_after = registry.list_versions("churn_booster")
    assert all_v_after[1].status == "Archived"
