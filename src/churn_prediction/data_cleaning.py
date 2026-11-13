"""Data cleaning and quality remediation module."""

import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Total Charges column by handling whitespace strings and converting to float.
    
    In the raw dataset, when Tenure Months is 0, Total Charges contains a single whitespace ' '
    because new subscribers have not completed their first billing cycle. 
    This function replaces such blank values with 0.0 and converts the series to float64.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'Total Charges' and 'Tenure Months'.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned 'Total Charges' column.
    """
    df = df.copy()
    if "Total Charges" not in df.columns:
        return df

    # If column is object or string, strip and coerce
    if df["Total Charges"].dtype == object or isinstance(df["Total Charges"].dtype, pd.StringDtype):
        # Identify whitespace strings
        cleaned_tc = df["Total Charges"].astype(str).str.strip()
        is_blank = cleaned_tc == ""
        
        # Where tenure is 0 and total charges is blank, logically charges billed to date is 0.0
        if "Tenure Months" in df.columns:
            zero_tenure_mask = is_blank & (df["Tenure Months"] == 0)
            cleaned_tc = cleaned_tc.mask(zero_tenure_mask, "0.0")
        
        # Convert to numeric, replacing any remaining non-parsable strings with NaN
        df["Total Charges"] = pd.to_numeric(cleaned_tc, errors="coerce")
    else:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        if "Tenure Months" in df.columns:
            df.loc[(df["Total Charges"].isna()) & (df["Tenure Months"] == 0), "Total Charges"] = 0.0

    return df


def clean_categorical_columns(df: pd.DataFrame, categorical_cols: Optional[list] = None) -> pd.DataFrame:
    """Strip leading/trailing whitespace and ensure clean string representations for categoricals.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    categorical_cols : list, optional
        List of categorical column names. If None, operates on all object columns.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with stripped categorical strings.
    """
    df = df.copy()
    if categorical_cols is not None:
        cols = categorical_cols
    else:
        cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) in ("str", "string")]
    for col in cols:
        if col in df.columns and (df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype)):
            df[col] = df[col].astype(str).str.strip()
    return df


def clean_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run full data cleaning pipeline on raw dataframe without dropping records.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for exploratory analysis and preprocessing.
    """
    df_clean = df.copy()
    df_clean = clean_total_charges(df_clean)
    df_clean = clean_categorical_columns(df_clean)
    logger.info("Dataset cleaning complete: Total Charges cleaned to float64, categoricals stripped.")
    return df_clean
