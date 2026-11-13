"""Unit and integration tests for data validation, cleaning, and splitting."""

import numpy as np
import pandas as pd
import pytest

from churn_prediction.config import ALL_FEATURE_COLS, TARGET_COL
from churn_prediction.data_cleaning import clean_total_charges, clean_raw_dataset
from churn_prediction.data_validation import (
    DataValidationError,
    validate_cleaned_data,
    validate_raw_data,
    validate_train_test_split,
)


@pytest.fixture
def sample_valid_df():
    """Create a minimal valid dataframe with all required columns."""
    data = {
        "CustomerID": ["1001-A", "1002-B", "1003-C", "1004-D"],
        "Tenure Months": [12, 0, 24, 6],
        "Monthly Charges": [50.0, 20.0, 85.5, 65.0],
        "Total Charges": ["600.0", " ", "2050.0", "390.0"],
        "Gender": ["Male", "Female", "Male", "Female"],
        "Senior Citizen": ["No", "No", "Yes", "No"],
        "Partner": ["Yes", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No"],
        "Phone Service": ["Yes", "Yes", "Yes", "Yes"],
        "Multiple Lines": ["No", "No", "Yes", "No"],
        "Internet Service": ["DSL", "No", "Fiber optic", "DSL"],
        "Online Security": ["Yes", "No internet service", "No", "Yes"],
        "Online Backup": ["No", "No internet service", "Yes", "No"],
        "Device Protection": ["No", "No internet service", "Yes", "No"],
        "Tech Support": ["Yes", "No internet service", "No", "No"],
        "Streaming TV": ["No", "No internet service", "Yes", "No"],
        "Streaming Movies": ["No", "No internet service", "Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "Paperless Billing": ["Yes", "No", "Yes", "Yes"],
        "Payment Method": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        "Churn Label": ["No", "No", "No", "Yes"],
        "Churn Value": [0, 0, 0, 1],
    }
    return pd.DataFrame(data)


def test_clean_total_charges_replaces_whitespace_for_zero_tenure(sample_valid_df):
    """Verify whitespace in Total Charges is converted to 0.0 when tenure is 0."""
    cleaned = clean_total_charges(sample_valid_df)
    assert np.issubdtype(cleaned["Total Charges"].dtype, np.floating)
    assert cleaned.loc[sample_valid_df["Tenure Months"] == 0, "Total Charges"].values[0] == 0.0
    assert cleaned.loc[sample_valid_df["Tenure Months"] == 12, "Total Charges"].values[0] == 600.0


def test_validate_raw_data_success(sample_valid_df):
    """Verify raw data validation passes with complete schema."""
    assert validate_raw_data(sample_valid_df) is True


def test_validate_raw_data_duplicate_id_fails(sample_valid_df):
    """Verify validation raises error if duplicate CustomerID exists."""
    df_dup = sample_valid_df.copy()
    df_dup.loc[1, "CustomerID"] = df_dup.loc[0, "CustomerID"]
    with pytest.raises(DataValidationError, match="duplicate CustomerID"):
        validate_raw_data(df_dup)


def test_validate_raw_data_missing_column_fails(sample_valid_df):
    """Verify validation raises error if required feature is absent."""
    df_missing = sample_valid_df.drop(columns=["Contract"])
    with pytest.raises(DataValidationError, match="Missing required feature columns"):
        validate_raw_data(df_missing)


def test_validate_cleaned_data(sample_valid_df):
    """Verify cleaned data validation passes after clean_raw_dataset."""
    cleaned = clean_raw_dataset(sample_valid_df)
    assert validate_cleaned_data(cleaned) is True


def test_validate_cleaned_data_invalid_target_fails(sample_valid_df):
    """Verify validation raises error if target column contains values outside {0, 1}."""
    cleaned = clean_raw_dataset(sample_valid_df)
    cleaned.loc[0, TARGET_COL] = 99
    with pytest.raises(DataValidationError, match="invalid values"):
        validate_cleaned_data(cleaned)


def test_validate_train_test_split_leakage_detection():
    """Verify validation fails if index leakage occurs between train and test."""
    X_train = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
    y_train = pd.Series([0, 1, 0], index=[0, 1, 2])
    X_test = pd.DataFrame({"a": [3, 4]}, index=[2, 3])  # index 2 overlaps!
    y_test = pd.Series([0, 1], index=[2, 3])

    with pytest.raises(DataValidationError, match="Data leakage detected"):
        validate_train_test_split(X_train, X_test, y_train, y_test)
