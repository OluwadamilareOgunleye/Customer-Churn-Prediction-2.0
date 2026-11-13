"""Utility functions for scoring, risk tiering, and output formatting."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def assign_risk_category(probability: float) -> str:
    """Assign a justified risk category based on calibrated churn probability.
    
    Tiers are established based on the calibrated decision threshold (0.30):
    - Low Risk (< 0.30): Below the intervention threshold. Default retention stance: Monitor without intervention.
    - Medium Risk (0.30 - 0.59): Above decision threshold; moderate defection risk. High ROI for service/billing nudges.
    - High Risk (>= 0.60): Critical defection hazard. High priority for proactive retention campaigns (e.g. contract renewal incentives).
    
    Parameters
    ----------
    probability : float
        Predicted churn probability in [0, 1].
        
    Returns
    -------
    str
        'Low Risk', 'Medium Risk', or 'High Risk'.
    """
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"


def format_prediction_result(
    customer_id: Optional[str],
    probability: float,
    threshold: float,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Format a single customer's prediction result into a clean, standardized dictionary.
    
    Parameters
    ----------
    customer_id : str, optional
        Customer ID.
    probability : float
        Churn probability.
    threshold : float
        Decision threshold.
    warnings : list, optional
        List of warnings.
        
    Returns
    -------
    dict
        Structured prediction output.
    """
    prob_clean = float(np.round(probability, 4))
    pred = int(prob_clean >= threshold)
    label = "Yes" if pred == 1 else "No"
    risk_cat = assign_risk_category(prob_clean)

    return {
        "customer_id": customer_id,
        "churn_probability": prob_clean,
        "churn_prediction": pred,
        "churn_label": label,
        "risk_category": risk_cat,
        "decision_threshold": float(threshold),
        "warnings": warnings or [],
        "calibration_note": (
            "Probability represents estimated empirical likelihood from cross-validated gradient boosting. "
            "It is a risk propensity score, not a deterministic guarantee of departure."
        ),
    }
