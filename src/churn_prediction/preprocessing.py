"""Preprocessing pipeline construction and data preparation."""

import logging
from typing import List, Optional, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_prediction.config import (
    ALL_FEATURE_COLS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from churn_prediction.data_cleaning import clean_raw_dataset
from churn_prediction.data_validation import validate_cleaned_data, validate_train_test_split

logger = logging.getLogger(__name__)


def build_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build a reusable Scikit-Learn ColumnTransformer for numeric and categorical features.
    
    Parameters
    ----------
    numerical_features : list, optional
        List of continuous feature names.
    categorical_features : list, optional
        List of nominal feature names.
        
    Returns
    -------
    ColumnTransformer
        Configured, unfitted ColumnTransformer.
    """
    num_cols = numerical_features if numerical_features is not None else NUMERICAL_FEATURES
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def prepare_and_split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_processed: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Clean dataset, extract predictive features and target, and perform stratified train-test split.
    
    Ensures that preprocessing transformations are never fitted prior to splitting,
    eliminating data leakage between train and test distributions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    test_size : float, default=0.20
        Proportion of observations reserved for test set.
    random_state : int, default=42
        Reproducibility seed.
    save_processed : bool, default=True
        Whether to save split datasets to data/processed/.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    # 1. Clean dataset
    df_clean = clean_raw_dataset(df)
    validate_cleaned_data(df_clean)

    # 2. Extract features and target
    X = df_clean[ALL_FEATURE_COLS].copy()
    y = df_clean[TARGET_COL].copy().astype(int)

    # 3. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 4. Validate split
    validate_train_test_split(X_train, X_test, y_train, y_test)
    logger.info(
        f"Data split successfully: Train={X_train.shape[0]} rows (Churn rate: {y_train.mean():.4f}), "
        f"Test={X_test.shape[0]} rows (Churn rate: {y_test.mean():.4f})"
    )

    # 5. Persist processed datasets if requested
    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
        logger.info(f"Saved train.csv and test.csv to {PROCESSED_DATA_DIR}")

    return X_train, X_test, y_train, y_test
