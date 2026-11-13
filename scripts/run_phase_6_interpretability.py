"""Execution script for Phase 6: Model Interpretability."""

import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd

from churn_prediction.config import (
    FIGURES_DIR,
    PHASE_6_REPORTS_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
)
from churn_prediction.interpretability import (
    compute_shap_explanations,
    explain_single_customer,
    generate_global_shap_plots,
)
from churn_prediction.model_inference import load_model_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    PHASE_6_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    X_test = test_df.drop(columns=["Churn Value"])

    logger.info("Computing global SHAP values on test sample (300 records)...")
    sample_df = X_test.head(300)
    shap_vals, feature_names = compute_shap_explanations(pipeline, sample_df)

    logger.info("Generating global SHAP summary plots...")
    saved_plots = generate_global_shap_plots(shap_vals, feature_names, save_dir=FIGURES_DIR)

    # Explain high risk customer and low risk customer
    logger.info("Explaining representative customer records...")
    high_risk_idx = test_df[test_df["Churn Value"] == 1].index[0]
    low_risk_idx = test_df[test_df["Churn Value"] == 0].index[0]

    high_risk_cust = X_test.loc[[high_risk_idx]]
    low_risk_cust = X_test.loc[[low_risk_idx]]

    exp_high = explain_single_customer(
        pipeline,
        high_risk_cust,
        customer_id=f"High_Risk_Customer_{high_risk_idx}",
        save_path=FIGURES_DIR / "16_shap_local_high_risk.png",
    )
    exp_low = explain_single_customer(
        pipeline,
        low_risk_cust,
        customer_id=f"Low_Risk_Customer_{low_risk_idx}",
        save_path=FIGURES_DIR / "17_shap_local_low_risk.png",
    )

    report_data = {
        "sample_size": len(sample_df),
        "total_features_explained": len(feature_names),
        "high_risk_example": exp_high,
        "low_risk_example": exp_low,
    }
    with open(PHASE_6_REPORTS_DIR / "interpretability_summary.json", "w") as f:
        json.dump(report_data, f, indent=2)

    logger.info("Phase 6 interpretability analysis completed successfully.")


if __name__ == "__main__":
    main()
