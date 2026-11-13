"""Sample script demonstrating single and batch inference using ChurnPredictor."""

import json
import logging
import pandas as pd

from churn_prediction.model_inference import ChurnPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    predictor = ChurnPredictor()
    logger.info(f"Loaded ChurnPredictor with decision threshold: {predictor.default_threshold:.2f}")

    # 1. Single Customer Inference Example
    sample_customer = {
        "CustomerID": "CUST-9876",
        "Gender": "Female",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 3,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 89.85,
        "Total Charges": 269.55,
    }

    result = predictor.predict_single(sample_customer)
    print("\n--- Single Customer Prediction Result ---")
    print(json.dumps(result, indent=2))

    # 2. Batch Customer Inference Example
    batch_records = [
        sample_customer,
        {
            "CustomerID": "CUST-1234",
            "Gender": "Male",
            "Senior Citizen": "No",
            "Partner": "Yes",
            "Dependents": "Yes",
            "Tenure Months": 60,
            "Phone Service": "Yes",
            "Multiple Lines": "Yes",
            "Internet Service": "DSL",
            "Online Security": "Yes",
            "Online Backup": "Yes",
            "Device Protection": "Yes",
            "Tech Support": "Yes",
            "Streaming TV": "No",
            "Streaming Movies": "No",
            "Contract": "Two year",
            "Paperless Billing": "No",
            "Payment Method": "Credit card (automatic)",
            "Monthly Charges": 65.50,
            "Total Charges": 3930.00,
        },
    ]

    batch_df = pd.DataFrame(batch_records)
    scored_df, warnings = predictor.predict_batch(batch_df)
    print("\n--- Batch Prediction Output Table ---")
    display_cols = ["CustomerID", "Tenure Months", "Contract", "churn_probability", "churn_label", "risk_category"]
    print(scored_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
