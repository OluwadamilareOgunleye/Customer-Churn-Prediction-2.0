"""
Centralized Business Metric & KPI Calculation Engine.
Strictly calculates metrics from pandas dataframes without fabrication or synthetic values.
All monetary calculations are preserved in original USD ($).
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


def calculate_executive_kpis(df: pd.DataFrame, baseline_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Calculate primary enterprise analytics KPIs from filtered customer accounts.
    """
    total_customers = len(df)
    if total_customers == 0:
        return {
            "total_customers": 0,
            "churned_customers": 0,
            "retained_customers": 0,
            "churn_rate": 0.0,
            "mrr_total": 0.0,
            "revenue_lost": 0.0,
            "revenue_at_risk": 0.0,
            "avg_monthly_charges": 0.0,
            "avg_tenure": 0.0,
            "churn_rate_delta": None,
        }

    churned_mask = df["Churn Value"] == 1
    churned_customers = int(churned_mask.sum())
    retained_customers = total_customers - churned_customers
    churn_rate = (churned_customers / total_customers) * 100.0

    # Monthly Recurring Revenue (USD)
    mrr_total = float(df["Monthly Charges"].sum())

    # Historical Revenue Lost (MRR from churned customers)
    revenue_lost = float(df.loc[churned_mask, "Monthly Charges"].sum())

    # Revenue at Risk (Active customers on Month-to-Month contracts paying >= $70/mo)
    active_m2m_high_spend = (~churned_mask) & (df["Contract"] == "Month-to-month") & (df["Monthly Charges"] >= 70.0)
    revenue_at_risk = float(df.loc[active_m2m_high_spend, "Monthly Charges"].sum())

    avg_monthly_charges = float(df["Monthly Charges"].mean())
    avg_tenure = float(df["Tenure Months"].mean())

    # Baseline comparison delta (if filtering is active)
    churn_rate_delta = None
    if baseline_df is not None and len(baseline_df) > 0:
        base_rate = (baseline_df["Churn Value"] == 1).mean() * 100.0
        churn_rate_delta = round(churn_rate - base_rate, 2)

    return {
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "retained_customers": retained_customers,
        "churn_rate": round(churn_rate, 2),
        "mrr_total": round(mrr_total, 2),
        "revenue_lost": round(revenue_lost, 2),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "avg_monthly_charges": round(avg_monthly_charges, 2),
        "avg_tenure": round(avg_tenure, 1),
        "churn_rate_delta": churn_rate_delta,
    }


def compute_churn_by_dimension(df: pd.DataFrame, dimension_col: str) -> pd.DataFrame:
    """
    Compute customer count, churn count, churn rate, and total revenue across a category.
    """
    if len(df) == 0 or dimension_col not in df.columns:
        return pd.DataFrame()

    summary = df.groupby(dimension_col, observed=False).agg(
        total_customers=("Churn Value", "count"),
        churned_customers=("Churn Value", lambda x: (x == 1).sum()),
        total_revenue=("Monthly Charges", "sum"),
        avg_monthly_charges=("Monthly Charges", "mean"),
    ).reset_index()

    summary["churn_rate"] = (summary["churned_customers"] / summary["total_customers"]) * 100.0
    summary["retention_rate"] = 100.0 - summary["churn_rate"]
    return summary.sort_values(by="total_customers", ascending=False)
