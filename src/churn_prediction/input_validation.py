"""Input schema definition, type checking, and boundary validation for customer records."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from churn_prediction.config import (
    ALL_FEATURE_COLS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)

VALID_CATEGORIES = {
    "Gender": {"Male", "Female"},
    "Senior Citizen": {"Yes", "No"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "Phone Service": {"Yes", "No"},
    "Multiple Lines": {"Yes", "No", "No phone service"},
    "Internet Service": {"DSL", "Fiber optic", "No"},
    "Online Security": {"Yes", "No", "No internet service"},
    "Online Backup": {"Yes", "No", "No internet service"},
    "Device Protection": {"Yes", "No", "No internet service"},
    "Tech Support": {"Yes", "No", "No internet service"},
    "Streaming TV": {"Yes", "No", "No internet service"},
    "Streaming Movies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "Paperless Billing": {"Yes", "No"},
    "Payment Method": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


class ValidationError(ValueError):
    """Raised when customer input validation fails fatally."""
    pass


def validate_customer_record(
    record: Union[Dict[str, Any], pd.Series],
    strict: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """Validate a single customer record dictionary, casting types and checking boundaries.
    
    Parameters
    ----------
    record : dict or pd.Series
        Customer record.
    strict : bool, default=False
        If True, raises ValidationError on unseen categories. If False, emits warning.
        
    Returns
    -------
    Tuple[Dict[str, Any], List[str]]
        Validated and normalized dictionary and a list of warning strings.
    """
    data = dict(record)
    warnings: List[str] = []

    # Check missing required fields
    missing_fields = [f for f in ALL_FEATURE_COLS if f not in data or data[f] is None or data[f] == ""]
    if missing_fields:
        raise ValidationError(f"Missing required features: {missing_fields}")

    cleaned: Dict[str, Any] = {}

    # 1. Validate numericals
    for num_col in NUMERICAL_FEATURES:
        val = data[num_col]
        try:
            num_val = float(val)
        except (ValueError, TypeError):
            raise ValidationError(f"Feature '{num_col}' must be numeric; received {val!r}")

        if num_val < 0:
            raise ValidationError(f"Feature '{num_col}' cannot be negative; received {num_val}")
        cleaned[num_col] = num_val

    # 2. Validate categoricals
    for cat_col in CATEGORICAL_FEATURES:
        val_str = str(data[cat_col]).strip()
        expected_set = VALID_CATEGORIES.get(cat_col, set())

        if expected_set and val_str not in expected_set:
            msg = f"Feature '{cat_col}' received unexpected category '{val_str}'. Expected one of {sorted(expected_set)}."
            if strict:
                raise ValidationError(msg)
            else:
                warnings.append(msg + " Will be treated as unknown/imputed during encoding.")

        cleaned[cat_col] = val_str

    # Preserve customer_id if present
    if "CustomerID" in data:
        cleaned["CustomerID"] = str(data["CustomerID"])
    elif "customer_id" in data:
        cleaned["CustomerID"] = str(data["customer_id"])

    return cleaned, warnings


def validate_input_dataframe(
    df: pd.DataFrame,
    strict: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Validate a batch DataFrame of customer records.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing customer records.
    strict : bool, default=False
        Strict mode flag.
        
    Returns
    -------
    Tuple[pd.DataFrame, List[Dict[str, Any]]]
        Normalized DataFrame of features and list of per-row warnings.
    """
    if df is None or df.empty:
        raise ValidationError("Input DataFrame is empty or None.")

    missing_cols = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValidationError(f"Input DataFrame is missing required columns: {missing_cols}")

    validated_records = []
    all_warnings = []

    for idx, row in df.iterrows():
        try:
            clean_rec, warns = validate_customer_record(row, strict=strict)
            validated_records.append(clean_rec)
            if warns:
                all_warnings.append({"row_index": idx, "warnings": warns})
        except ValidationError as e:
            raise ValidationError(f"Error validating row {idx}: {e}") from e

    out_df = pd.DataFrame(validated_records)
    return out_df, all_warnings
