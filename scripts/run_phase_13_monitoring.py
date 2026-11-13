"""Script executing Phase 13: Drift detection audit and monitoring verification."""

import json
import logging
from pathlib import Path
import pandas as pd

from churn_prediction.config import (
    PHASE_13_REPORTS_DIR,
    PROCESSED_DATA_DIR,
)
from churn_prediction.drift_detection import run_drift_audit
from churn_prediction.monitoring import PredictionLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    PHASE_13_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load baseline training data and test data
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    logger.info("Executing distribution drift audit between train.csv (reference) and test.csv (production sample)...")
    drift_report = run_drift_audit(train_df, test_df)

    with open(PHASE_13_REPORTS_DIR / "drift_audit_report.json", "w") as f:
        json.dump(drift_report, f, indent=2)

    logger.info(f"Drift audit complete. Overall Status: {drift_report['overall_status']}")
    print(f"Overall Drift Status: {drift_report['overall_status']}")
    print(f"Numerical features checked: {list(drift_report['numerical_drift'].keys())}")
    for col, m in drift_report["numerical_drift"].items():
        print(f"  {col}: PSI={m['psi']:.4f}, KS p-value={m['p_value']:.4f} -> {m['status']}")

    # 2. Test PredictionLogger
    p_logger = PredictionLogger()
    p_logger.log_prediction(
        customer_id="AUDIT-TEST-1",
        probability=0.745,
        prediction=1,
        risk_category="High Risk",
        threshold=0.30,
        latency_ms=12.4,
    )
    p_logger.log_prediction(
        customer_id="AUDIT-TEST-2",
        probability=0.125,
        prediction=0,
        risk_category="Low Risk",
        threshold=0.30,
        latency_ms=8.1,
    )

    metrics = p_logger.get_summary_metrics()
    print("\n--- Prediction Monitoring Telemetry ---")
    print(json.dumps(metrics, indent=2))

    logger.info("Phase 13 monitoring script executed successfully.")


if __name__ == "__main__":
    main()
