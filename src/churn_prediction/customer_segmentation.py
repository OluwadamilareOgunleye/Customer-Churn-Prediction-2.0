"""
Customer Segmentation & Churn-Risk Stratification.

Provides K-Means customer persona segmentation and rule-based churn risk tiering
for actionable business intelligence and retention prioritization.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def segment_customers_kmeans(
    df: pd.DataFrame,
    features: List[str] = None,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[KMeans, pd.DataFrame, float]:
    """
    Cluster customers into behavioral cohorts using K-Means on scaled numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input customer dataframe.
    features : List[str]
        Continuous/ordinal features for clustering (default: Tenure Months, Monthly Charges, Total Charges).
    n_clusters : int
        Number of customer clusters (personas).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Tuple[KMeans, pd.DataFrame, float]
        Fitted model, DataFrame with assigned cluster labels, and silhouette score.
    """
    if features is None:
        features = ["Tenure Months", "Monthly Charges", "Total Charges"]

    # Filter available features
    valid_features = [f for f in features if f in df.columns]
    X = df[valid_features].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    sil_score = float(silhouette_score(X_scaled, cluster_labels))

    df_out = df.copy()
    df_out["customer_segment"] = cluster_labels

    # Map numeric cluster index to descriptive personas based on tenure and charges
    persona_map = {}
    summary = df_out.groupby("customer_segment")[["Tenure Months", "Monthly Charges"]].mean()
    for cluster_id, row in summary.iterrows():
        tenure = row["Tenure Months"]
        charge = row["Monthly Charges"]
        if tenure <= 15 and charge < 50:
            name = "New Budget Customers"
        elif tenure <= 15 and charge >= 50:
            name = "New High-Spend / Tech Customers"
        elif tenure > 15 and charge >= 70:
            name = "Tenured Premium Accounts"
        else:
            name = "Long-Term Value Loyalists"
        persona_map[cluster_id] = name

    df_out["persona_name"] = df_out["customer_segment"].map(persona_map)

    return kmeans, df_out, sil_score


def assign_churn_risk_tiers(
    probabilities: np.ndarray,
    low_cutoff: float = 0.15,
    med_cutoff: float = 0.30,
    elevated_cutoff: float = 0.60
) -> pd.Series:
    """
    Categorize continuous churn probabilities into actionable risk tiers.

    Tiers:
    - Low Risk: prob < 0.15
    - Moderate Risk: 0.15 <= prob < 0.30
    - Elevated Risk: 0.30 <= prob < 0.60
    - High Risk: prob >= 0.60
    """
    conditions = [
        probabilities < low_cutoff,
        (probabilities >= low_cutoff) & (probabilities < med_cutoff),
        (probabilities >= med_cutoff) & (probabilities < elevated_cutoff),
        probabilities >= elevated_cutoff
    ]
    choices = ["Low Risk", "Moderate Risk", "Elevated Risk", "High Risk"]
    return pd.Series(np.select(conditions, choices, default="Moderate Risk"))
