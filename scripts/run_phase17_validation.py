"""
Execute Phase 17 Advanced Model Validation, Calibration, and Robustness Analysis.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

from churn_prediction.config import (
    MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR,
    PHASE_17_REPORTS_DIR, TARGET_COL, ALL_FEATURE_COLS
)
from churn_prediction.calibration import compute_calibration_curve, evaluate_threshold_sweep, plot_calibration_comparison
from churn_prediction.robustness import (
    evaluate_subgroup_performance, run_missing_value_stress_test,
    run_numerical_extreme_stress_test, run_feature_sensitivity_analysis
)

def main():
    print("=== Starting Phase 17: Model Validation & Robustness ===")
    PHASE_17_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and production pipeline
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    bundle = joblib.load(MODELS_DIR / "churn_pipeline_bundle.joblib")
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]

    X_train = train_df[feature_names]
    y_train = train_df[TARGET_COL].values

    X_test = test_df[feature_names]
    y_test = test_df[TARGET_COL].values

    print(f"Loaded Train: {len(X_train)} samples, Test: {len(X_test)} samples.")

    # 2. Probability Calibration
    y_prob_uncal = pipeline.predict_proba(X_test)[:, 1]
    prob_true_uncal, prob_pred_uncal, brier_uncal = compute_calibration_curve(y_test, y_prob_uncal, n_bins=10)
    print(f"Uncalibrated HistGradientBoosting Brier Score: {brier_uncal:.4f}")

    # Fit CalibratedClassifierCV using 5-fold CV on TRAIN SET (to avoid test set data leakage)
    calibrated_pipeline = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=5)
    calibrated_pipeline.fit(X_train, y_train)
    y_prob_cal = calibrated_pipeline.predict_proba(X_test)[:, 1]
    prob_true_cal, prob_pred_cal, brier_cal = compute_calibration_curve(y_test, y_prob_cal, n_bins=10)
    print(f"Sigmoid-Calibrated Model Brier Score: {brier_cal:.4f}")

    # Plot Calibration Curve
    fig_cal_path = FIGURES_DIR / "18_calibration_curve_comparison.png"
    plot_calibration_comparison(
        uncalibrated_data=(prob_true_uncal, prob_pred_uncal, brier_uncal, "HistGradientBoosting (Raw)"),
        calibrated_data=(prob_true_cal, prob_pred_cal, brier_cal, "HistGradientBoosting (Platt Sigmoid)"),
        save_path=fig_cal_path
    )
    plt.close()
    print(f"Saved calibration figure to {fig_cal_path}")

    # 3. Decision Threshold Sweep
    threshold_df = evaluate_threshold_sweep(y_test, y_prob_uncal)
    threshold_df.to_csv(PHASE_17_REPORTS_DIR / "threshold_analysis.csv", index=False)

    # Plot Threshold Metrics Tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], "b-o", label="Precision", linewidth=2)
    ax.plot(threshold_df["threshold"], threshold_df["recall"], "g-s", label="Recall", linewidth=2)
    ax.plot(threshold_df["threshold"], threshold_df["f1_score"], "r-^", label="F1-Score", linewidth=2)
    ax.axvline(0.30, color="purple", linestyle="--", label="Operational Threshold (tau=0.30)", linewidth=2)
    ax.set_xlabel("Decision Threshold (tau)", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Precision, Recall, and F1-Score Tradeoff Across Decision Thresholds", fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(0.1, 1.0, 0.05))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="center right", fontsize=11)
    plt.tight_layout()
    fig_thresh_path = FIGURES_DIR / "19_threshold_tradeoff_curve.png"
    fig.savefig(fig_thresh_path, dpi=300)
    plt.close()
    print(f"Saved threshold tradeoff curve to {fig_thresh_path}")

    # 4. Subgroup Performance Analysis
    test_analysis_df = test_df.copy()
    test_analysis_df["y_true"] = y_test
    test_analysis_df["y_prob"] = y_prob_uncal

    # Create tenure buckets
    test_analysis_df["tenure_group"] = pd.cut(
        test_analysis_df["Tenure Months"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 Mos", "13-24 Mos", "25-48 Mos", "49-72 Mos"]
    )

    subgroups = ["Contract", "tenure_group", "Payment Method", "Internet Service"]
    all_subgroup_metrics = []
    for sg in subgroups:
        sg_metrics = evaluate_subgroup_performance(
            test_analysis_df, subgroup_col=sg, y_true_col="y_true", y_prob_col="y_prob", threshold=0.30
        )
        all_subgroup_metrics.append(sg_metrics)

    combined_subgroups = pd.concat(all_subgroup_metrics, ignore_index=True)
    combined_subgroups.to_csv(PHASE_17_REPORTS_DIR / "subgroup_performance.csv", index=False)
    print("Computed Subgroup Performance across Contract, Tenure, Payment, and InternetService.")

    # 5. Missing Value Stress Test
    features_to_test = ["Total Charges", "Monthly Charges", "Tenure Months", "Contract", "Payment Method", "Internet Service"]
    missing_stress_df = run_missing_value_stress_test(pipeline, X_test.head(50), features_to_test)
    missing_stress_df.to_csv(PHASE_17_REPORTS_DIR / "missing_value_stress_test.csv", index=False)
    print("Completed Missing Value Stress Test.")

    # 6. Numerical Extreme Stress Test
    sample_cust = X_test.iloc[0].to_dict()
    extreme_cases = [
        {"name": "Zero Tenure & Zero Charges", "updates": {"Tenure Months": 0, "Monthly Charges": 0.0, "Total Charges": 0.0}},
        {"name": "Ultra Long Tenure (120 mos)", "updates": {"Tenure Months": 120, "Monthly Charges": 80.0, "Total Charges": 9600.0}},
        {"name": "Very High Monthly Bill ($300)", "updates": {"Tenure Months": 12, "Monthly Charges": 300.0, "Total Charges": 3600.0}},
        {"name": "Negative Monthly Bill (-$50)", "updates": {"Tenure Months": 6, "Monthly Charges": -50.0, "Total Charges": 100.0}},
        {"name": "Extreme Total Charges ($50,000)", "updates": {"Tenure Months": 72, "Monthly Charges": 120.0, "Total Charges": 50000.0}}
    ]
    extreme_stress_df = run_numerical_extreme_stress_test(pipeline, sample_cust, extreme_cases)
    extreme_stress_df.to_csv(PHASE_17_REPORTS_DIR / "numerical_extreme_stress_test.csv", index=False)
    print("Completed Numerical Extreme Stress Test.")

    # 7. Feature Sensitivity Analysis (Tenure Months and Monthly Charges)
    tenure_sweep = run_feature_sensitivity_analysis(
        pipeline, sample_cust, "Tenure Months", list(range(0, 73, 6))
    )
    charges_sweep = run_feature_sensitivity_analysis(
        pipeline, sample_cust, "Monthly Charges", list(range(20, 130, 10))
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(tenure_sweep["value"], tenure_sweep["predicted_prob"], "b-o", linewidth=2)
    axes[0].axhline(0.30, color="r", linestyle="--", label="Threshold = 0.30")
    axes[0].set_xlabel("Tenure (Months)", fontsize=11)
    axes[0].set_ylabel("Predicted Churn Probability", fontsize=11)
    axes[0].set_title("Sensitivity: Tenure vs Churn Probability", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(charges_sweep["value"], charges_sweep["predicted_prob"], "m-s", linewidth=2)
    axes[1].axhline(0.30, color="r", linestyle="--", label="Threshold = 0.30")
    axes[1].set_xlabel("Monthly Charges ($)", fontsize=11)
    axes[1].set_ylabel("Predicted Churn Probability", fontsize=11)
    axes[1].set_title("Sensitivity: Monthly Charges vs Churn Probability", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    fig_sens_path = FIGURES_DIR / "20_feature_sensitivity_curves.png"
    fig.savefig(fig_sens_path, dpi=300)
    plt.close()
    print(f"Saved sensitivity curves to {fig_sens_path}")

    # 8. Summary JSON
    summary = {
        "calibration": {
            "uncalibrated_brier_score": round(brier_uncal, 4),
            "calibrated_sigmoid_brier_score": round(brier_cal, 4),
            "brier_improvement_pct": round(((brier_uncal - brier_cal) / brier_uncal) * 100, 2)
        },
        "operating_threshold": 0.30,
        "metrics_at_0_30": threshold_df[threshold_df["threshold"] == 0.30].iloc[0].to_dict(),
        "metrics_at_0_50": threshold_df[threshold_df["threshold"] == 0.50].iloc[0].to_dict(),
        "subgroups_evaluated": subgroups
    }
    with open(PHASE_17_REPORTS_DIR / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved validation summary JSON.")
    print("=== Phase 17 Execution Completed Successfully ===")

if __name__ == "__main__":
    main()
