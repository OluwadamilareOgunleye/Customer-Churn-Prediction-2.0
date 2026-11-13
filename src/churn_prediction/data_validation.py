"""Data integrity and validation module."""

from typing import List, Optional
import numpy as np
import pandas as pd

from churn_prediction.config import (
    ALL_FEATURE_COLS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    ID_COL,
)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_raw_data(df: pd.DataFrame) -> bool:
    """Validate structure, identifiers, and required columns of raw dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
        
    Returns
    -------
    bool
        True if validation passes.
        
    Raises
    ------
    DataValidationError
        If validation checks fail.
    """
    if df is None or df.empty:
        raise DataValidationError("Raw dataset is empty or None.")

    # Check primary identifier
    if ID_COL in df.columns:
        duplicate_ids = df[ID_COL].duplicated().sum()
        if duplicate_ids > 0:
            raise DataValidationError(f"Found {duplicate_ids} duplicate CustomerID values.")

    # Check target column
    if TARGET_COL not in df.columns and "Churn Label" not in df.columns:
        raise DataValidationError(f"Neither '{TARGET_COL}' nor 'Churn Label' found in dataset.")

    # Check required features
    missing_features = [col for col in ALL_FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise DataValidationError(f"Missing required feature columns: {missing_features}")

    return True


def validate_cleaned_data(df: pd.DataFrame) -> bool:
    """Validate cleaned dataframe before splitting or pipeline execution.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
        
    Returns
    -------
    bool
        True if validation passes.
    """
    validate_raw_data(df)

    # Check Total Charges is numeric and has no NaNs
    if "Total Charges" in df.columns:
        if not np.issubdtype(df["Total Charges"].dtype, np.number):
            raise DataValidationError(f"'Total Charges' is not numeric (type: {df['Total Charges'].dtype}).")
        nan_tc = df["Total Charges"].isna().sum()
        if nan_tc > 0:
            raise DataValidationError(f"'Total Charges' contains {nan_tc} NaN values after cleaning.")

    # Check numerical feature ranges
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            if (df[col] < 0).any():
                raise DataValidationError(f"Numerical feature '{col}' contains negative values.")

    # Check target column values
    if TARGET_COL in df.columns:
        unique_targets = set(df[TARGET_COL].dropna().unique())
        if not unique_targets.issubset({0, 1}):
            raise DataValidationError(f"Target '{TARGET_COL}' contains invalid values: {unique_targets}")

    return True


def validate_train_test_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tolerance: float = 0.05,
) -> bool:
    """Validate train-test split for integrity, dimensions, index isolation, and class balance.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Testing target.
    tolerance : float, default=0.05
        Permitted percentage difference in class balance between train and test.
        
    Returns
    -------
    bool
        True if validation passes.
    """
    # Check shape alignment
    if len(X_train) != len(y_train):
        raise DataValidationError(f"Length mismatch: X_train ({len(X_train)}) vs y_train ({len(y_train)}).")
    if len(X_test) != len(y_test):
        raise DataValidationError(f"Length mismatch: X_test ({len(X_test)}) vs y_test ({len(y_test)}).")

    # Check feature dimension alignment
    if list(X_train.columns) != list(X_test.columns):
        raise DataValidationError("X_train and X_test column lists do not match.")

    # Check index isolation (no leakage across split)
    overlap = set(X_train.index).intersection(set(X_test.index))
    if len(overlap) > 0:
        raise DataValidationError(f"Data leakage detected! {len(overlap)} indices exist in both train and test sets.")

    # Check stratification balance
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    diff = abs(train_rate - test_rate)
    if diff > tolerance:
        raise DataValidationError(
            f"Class imbalance discrepancy across split exceeds tolerance ({diff:.4f} > {tolerance:.4f}): "
            f"Train={train_rate:.4f}, Test={test_rate:.4f}"
        )

    return True
