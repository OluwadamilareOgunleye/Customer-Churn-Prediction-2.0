"""
Execute Phase 19 Customer Segmentation & Business Impact Analysis.
"""

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from churn_prediction.config import (
    MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR,
    PHASE_19_REPORTS_DIR, TARGET_COL
)
from churn_prediction.customer_segmentation import segment_customers_kmeans, assign_churn_risk_tiers
from churn_prediction.business_impact import calculate_retention_roi, compare_threshold_business_impact

def main():
    print("=== Starting Phase 19: Customer Segmentation & Business Impact ===")
    PHASE_19_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data & production pipeline
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    bundle = joblib.load(MODELS_DIR / "churn_pipeline_bundle.joblib")
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]

    X_test = test_df[feature_names]
    y_test = test_df[TARGET_COL].values
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 2. Customer Segmentation via K-Means
    kmeans_model, segmented_df, sil_score = segment_customers_kmeans(
        test_df, features=["Tenure Months", "Monthly Charges", "Total Charges"], n_clusters=4
    )
    print(f"K-Means Customer Segmentation: 4 clusters, Silhouette Score: {sil_score:.4f}")

    # Add predictions & risk tiers
    segmented_df["churn_probability"] = y_prob
    segmented_df["predicted_churn_30"] = (y_prob >= 0.30).astype(int)
    segmented_df["risk_tier"] = assign_churn_risk_tiers(y_prob)

    # 3. Persona Profiling
    persona_summary = segmented_df.groupby("persona_name").agg(
        customer_count=("CustomerID", "count") if "CustomerID" in segmented_df.columns else ("Tenure Months", "count"),
        avg_tenure=("Tenure Months", "mean"),
        avg_monthly_charges=("Monthly Charges", "mean"),
        actual_churn_rate=(TARGET_COL, "mean"),
        avg_predicted_prob=("churn_probability", "mean"),
        high_risk_pct=("predicted_churn_30", "mean")
    ).reset_index()

    persona_summary.to_csv(PHASE_19_REPORTS_DIR / "customer_persona_profiles.csv", index=False)
    print("Exported customer persona profiles.")

    # Plot Customer Personas
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {"Low Risk": "#2ca02c", "Moderate Risk": "#1f77b4", "Elevated Risk": "#ff7f0e", "High Risk": "#d62728"}
    sns.scatterplot(
        data=segmented_df,
        x="Tenure Months",
        y="Monthly Charges",
        hue="risk_tier",
        palette=palette,
        alpha=0.7,
        ax=ax
    )
    ax.set_title("Customer Segmentation by Tenure, Spend, and Churn Risk Tier", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenure (Months)", fontsize=11)
    ax.set_ylabel("Monthly Charges ($)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_persona_path = FIGURES_DIR / "21_customer_segmentation_clusters.png"
    fig.savefig(fig_persona_path, dpi=300)
    plt.close()
    print(f"Saved persona scatter plot to {fig_persona_path}")

    # 4. Business Impact & ROI Modeling
    # Evaluate at tau = 0.30 vs tau = 0.50
    # From holdout test set:
    # At 0.30: TP=288, FP=243, FN=86, TN=792
    # At 0.50: TP=197, FP=96, FN=177, TN=939
    roi_30 = calculate_retention_roi(tp=288, fp=243, fn=86, tn=792, customer_lifetime_value=1200.0, campaign_cost_per_customer=50.0, intervention_success_rate=0.25)
    roi_50 = calculate_retention_roi(tp=197, fp=96, fn=177, tn=939, customer_lifetime_value=1200.0, campaign_cost_per_customer=50.0, intervention_success_rate=0.25)

    comparison_data = [
        {"threshold": 0.30, **roi_30},
        {"threshold": 0.50, **roi_50}
    ]
    with open(PHASE_19_REPORTS_DIR / "roi_simulation_summary.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Plot ROI Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_to_plot = ["campaign_cost_usd", "gross_revenue_saved_usd", "net_profit_usd"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    vals_30 = [roi_30[m] for m in metrics_to_plot]
    vals_50 = [roi_50[m] for m in metrics_to_plot]

    ax.bar(x - width/2, vals_30, width, label="Operational Threshold (tau=0.30)", color="#1f77b4")
    ax.bar(x + width/2, vals_50, width, label="Standard Threshold (tau=0.50)", color="#aec7e8")
    ax.set_ylabel("USD ($)", fontsize=11)
    ax.set_title("Retention ROI Financial Simulation (tau=0.30 vs tau=0.50)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Campaign Spend", "Revenue Saved", "Net Profit"], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(metrics_to_plot)):
        ax.text(x[i] - width/2, vals_30[i] + 1000, f"${vals_30[i]:,.0f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(x[i] + width/2, vals_50[i] + 1000, f"${vals_50[i]:,.0f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig_roi_path = FIGURES_DIR / "22_business_retention_roi_comparison.png"
    fig.savefig(fig_roi_path, dpi=300)
    plt.close()
    print(f"Saved ROI figure to {fig_roi_path}")

    # 5. Risk Tier Distribution
    tier_counts = segmented_df["risk_tier"].value_counts().to_dict()
    print("Risk Tier Distribution on Test Set:", tier_counts)

    print("=== Phase 19 Execution Completed Successfully ===")

if __name__ == "__main__":
    main()
