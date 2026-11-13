"""Data ingestion and loading module."""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from churn_prediction.config import (
    RAW_DATA_PATH,
    INTERIM_DATA_DIR,
    TARGET_COL,
    ALL_FEATURE_COLS,
    ID_COL,
)

logger = logging.getLogger(__name__)


def load_raw_data(filepath: Optional[Path] = None, use_cache: bool = True) -> pd.DataFrame:
    """Load raw dataset from Excel, caching an interim feather/csv for fast repeated loads.
    
    Parameters
    ----------
    filepath : Path, optional
        Path to raw data file. Defaults to RAW_DATA_PATH.
    use_cache : bool, default=True
        Whether to check and use interim cached CSV if available.
        
    Returns
    -------
    pd.DataFrame
        Raw dataset as DataFrame.
    """
    path = Path(filepath) if filepath else RAW_DATA_PATH
    cache_path = INTERIM_DATA_DIR / "raw_cache.csv"

    if use_cache and cache_path.exists():
        logger.info(f"Loading cached raw data from {cache_path}")
        return pd.read_csv(cache_path)

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at {path}")

    logger.info(f"Loading raw Excel dataset from {path}")
    df = pd.read_excel(path, sheet_name=0)

    # Save fast cache
    try:
        INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached raw dataset to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to cache dataset: {e}")

    return df


def split_features_target(
    df: pd.DataFrame, 
    feature_cols: Optional[list] = None,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing features and target.
    feature_cols : list, optional
        Specific feature columns to include. If None, uses ALL_FEATURE_COLS.
    target_col : str, default=TARGET_COL
        Target column name.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.
    """
    cols = feature_cols if feature_cols is not None else ALL_FEATURE_COLS
    
    missing_features = [col for col in cols if col not in df.columns]
    if missing_features:
        raise KeyError(f"Expected feature columns missing from DataFrame: {missing_features}")
        
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    X = df[cols].copy()
    y = df[target_col].copy()
    return X, y
