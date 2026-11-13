"""
Data Loader & Feature Enrichment for Customer Churn Intelligence Dashboard.
Loads processed dataset (7,043 customer accounts) and enriches behavioral dimensions.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data(show_spinner=False)
def load_full_dataset() -> pd.DataFrame:
    """
    Load and concatenate train and test splits (total 7,043 records)
    with derived analytical dimensions.
    """
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed dataset splits not found in data/processed/")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # 1. Derive Living Situation
    def derive_living_situation(row):
        partner = str(row.get("Partner", "")).strip().lower() == "yes"
        dependents = str(row.get("Dependents", "")).strip().lower() == "yes"
        if not partner and not dependents:
            return "Living Alone"
        elif partner and not dependents:
            return "With Partner"
        elif not partner and dependents:
            return "With Dependents"
        else:
            return "Partner & Dependents"

    df["Living Situation"] = df.apply(derive_living_situation, axis=1)

    # 2. Derive Additional Services Count (0 to 6)
    services = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]
    existing_services = [s for s in services if s in df.columns]
    df["Service Count"] = (df[existing_services] == "Yes").sum(axis=1)

    # 3. Derive Standard Tenure Group
    df["Tenure Group"] = pd.cut(
        df["Tenure Months"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0–12 Mos", "13–24 Mos", "25–48 Mos", "49–72 Mos"],
    )

    # 4. Standard Labels
    df["Churn Label"] = df["Churn Value"].map({0: "Retained", 1: "Churned"}).fillna("Unknown")
    df["Senior Status"] = df["Senior Citizen"].map({"Yes": "Senior", "No": "Non-Senior", 1: "Senior", 0: "Non-Senior"}).fillna(df["Senior Citizen"])

    # 5. Customer Type / Persona proxy
    def derive_customer_type(row):
        tenure = row["Tenure Months"]
        charge = row["Monthly Charges"]
        if tenure <= 12 and charge < 50:
            return "New Budget"
        elif tenure <= 12 and charge >= 50:
            return "New Premium"
        elif tenure > 12 and charge >= 70:
            return "Tenured High-Value"
        else:
            return "Established Loyalist"

    df["Customer Type"] = df.apply(derive_customer_type, axis=1)

    return df


def apply_global_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply active global filters to customer dataframe."""
    filtered = df.copy()

    if filters.get("gender") and filters["gender"] != "All":
        filtered = filtered[filtered["Gender"] == filters["gender"]]

    if filters.get("senior") and filters["senior"] != "All":
        filtered = filtered[filtered["Senior Status"] == filters["senior"]]

    if filters.get("contract") and filters["contract"] != "All":
        filtered = filtered[filtered["Contract"] == filters["contract"]]

    if filters.get("internet") and filters["internet"] != "All":
        filtered = filtered[filtered["Internet Service"] == filters["internet"]]

    if filters.get("payment") and filters["payment"] != "All":
        filtered = filtered[filtered["Payment Method"] == filters["payment"]]

    if filters.get("tenure_group") and filters["tenure_group"] != "All":
        filtered = filtered[filtered["Tenure Group"] == filters["tenure_group"]]

    if filters.get("churn_status") and filters["churn_status"] != "All":
        filtered = filtered[filtered["Churn Label"] == filters["churn_status"]]

    return filtered
