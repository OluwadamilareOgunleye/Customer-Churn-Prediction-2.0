"""
Business Impact & Retention ROI Financial Framework.

Provides economic simulation of proactive retention interventions:
- Scenario analysis with explicit parameters
- Net campaign return on investment (ROI)
- Cost-benefit breakdown comparing default threshold (0.50) vs operational threshold (0.30)
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def calculate_retention_roi(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    customer_lifetime_value: float = 1200.0,
    campaign_cost_per_customer: float = 50.0,
    intervention_success_rate: float = 0.25,
    acquisition_cost: float = 400.0
) -> Dict[str, Any]:
    """
    Simulate financial returns from targeted retention intervention campaign.

    Parameters
    ----------
    tp : int
        True Positives (churners correctly identified and targeted).
    fp : int
        False Positives (non-churners targeted with retention offers).
    fn : int
        False Negatives (churners missed by the model).
    tn : int
        True Negatives (non-churners left alone).
    customer_lifetime_value : float
        Hypothetical Customer Lifetime Value (CLV) retained if churn is prevented.
    campaign_cost_per_customer : float
        Direct cost of retention outreach, discount, or bonus per targeted customer.
    intervention_success_rate : float
        Fraction of targeted churners who accept the offer and remain customers.
    acquisition_cost : float
        Cost to acquire a replacement customer.

    Returns
    -------
    Dict[str, Any]
        Financial breakdown including total spend, revenue saved, and net ROI.
    """
    total_targeted = tp + fp
    total_campaign_cost = total_targeted * campaign_cost_per_customer

    # Churners successfully saved
    saved_churners = int(tp * intervention_success_rate)
    gross_revenue_saved = saved_churners * customer_lifetime_value

    # Net business value
    net_profit = gross_revenue_saved - total_campaign_cost
    roi_percent = (net_profit / total_campaign_cost * 100.0) if total_campaign_cost > 0 else 0.0

    # Unmitigated loss from missed churners
    unmitigated_churn_loss = fn * customer_lifetime_value

    return {
        "targeted_customers": int(total_targeted),
        "true_positives_targeted": int(tp),
        "false_positives_targeted": int(fp),
        "saved_churners": saved_churners,
        "campaign_cost_usd": round(total_campaign_cost, 2),
        "gross_revenue_saved_usd": round(gross_revenue_saved, 2),
        "net_profit_usd": round(net_profit, 2),
        "roi_percent": round(roi_percent, 2),
        "unmitigated_loss_usd": round(unmitigated_churn_loss, 2),
        "assumptions": {
            "customer_lifetime_value_usd": customer_lifetime_value,
            "campaign_cost_per_customer_usd": campaign_cost_per_customer,
            "intervention_success_rate": intervention_success_rate,
            "acquisition_cost_usd": acquisition_cost
        }
    }


def compare_threshold_business_impact(
    metrics_at_thresholds: List[Dict[str, Any]],
    clv: float = 1200.0,
    campaign_cost: float = 50.0,
    success_rate: float = 0.25
) -> pd.DataFrame:
    """
    Compare business ROI across multiple candidate decision thresholds.
    """
    rows = []
    for item in metrics_at_thresholds:
        t = item["threshold"]
        tp = int(item["tp"])
        fp = int(item["fp"])
        fn = int(item["fn"])
        tn = int(item["tn"])

        roi_data = calculate_retention_roi(
            tp=tp, fp=fp, fn=fn, tn=tn,
            customer_lifetime_value=clv,
            campaign_cost_per_customer=campaign_cost,
            intervention_success_rate=success_rate
        )

        rows.append({
            "threshold": t,
            "targeted": roi_data["targeted_customers"],
            "saved_churners": roi_data["saved_churners"],
            "campaign_cost": roi_data["campaign_cost_usd"],
            "revenue_saved": roi_data["gross_revenue_saved_usd"],
            "net_profit": roi_data["net_profit_usd"],
            "roi_percent": roi_data["roi_percent"]
        })

    return pd.DataFrame(rows)
