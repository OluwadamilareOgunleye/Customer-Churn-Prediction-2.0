"""End-to-end pipeline execution script for Customer Churn Prediction (Phases 3, 4, 5)."""

import matplotlib
matplotlib.use("Agg")

import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from churn_prediction.config import (
    ALL_FEATURE_COLS,
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    PHASE_3_REPORTS_DIR,
    PHASE_4_REPORTS_DIR,
    PHASE_5_REPORTS_DIR,
    RANDOM_STATE,
    TARGET_COL,
)
from churn_prediction.data_cleaning import clean_raw_dataset
from churn_prediction.data_loader import load_raw_data
from churn_prediction.data_validation import validate_cleaned_data, validate_train_test_split
from churn_prediction.evaluation import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    plot_threshold_tradeoffs,
    sweep_thresholds,
)
from churn_prediction.model_inference import ChurnPredictor, save_model_bundle
from churn_prediction.model_training import (
    build_full_pipeline,
    cross_validate_models,
    get_candidate_models,
    tune_best_model,
)
from churn_prediction.preprocessing import build_preprocessor, prepare_and_split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set publication style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300


def run_phase_3_eda(df: pd.DataFrame):
    """Generate and save publication figures and summary statistics for Phase 3."""
    logger.info("--- Starting Phase 3: Exploratory Data Analysis ---")

    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    churn_counts = df["Churn Label"].value_counts()
    churn_pcts = df["Churn Label"].value_counts(normalize=True) * 100
    bars = ax.bar(churn_counts.index, churn_counts.values, color=["#1f77b4", "#d62728"], width=0.5)
    for bar, pct in zip(bars, churn_pcts.values):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 50, f"{yval:,} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Customer Churn Target Distribution", fontsize=12, pad=10)
    ax.set_xlabel("Churn Status", fontsize=10)
    ax.set_ylabel("Customer Count", fontsize=10)
    ax.set_ylim(0, churn_counts.max() * 1.15)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_target_distribution.png", dpi=300)
    plt.close()

    # 2. Tenure vs Churn (Histogram & KDE)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(
        data=df,
        x="Tenure Months",
        hue="Churn Label",
        multiple="stack",
        palette={"No": "#1f77b4", "Yes": "#d62728"},
        bins=36,
        ax=ax,
    )
    ax.set_title("Tenure Months Distribution by Churn Status", fontsize=12, pad=10)
    ax.set_xlabel("Tenure (Months)", fontsize=10)
    ax.set_ylabel("Customer Count", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_tenure_distribution_by_churn.png", dpi=300)
    plt.close()

    # 3. Monthly Charges vs Churn (Boxplot & KDE)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(
        data=df,
        x="Churn Label",
        y="Monthly Charges",
        palette={"No": "#1f77b4", "Yes": "#d62728"},
        ax=ax,
        width=0.4,
    )
    ax.set_title("Monthly Charges ($) Distribution by Churn Status", fontsize=12, pad=10)
    ax.set_xlabel("Churn Status", fontsize=10)
    ax.set_ylabel("Monthly Charges ($)", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_monthly_charges_boxplot.png", dpi=300)
    plt.close()

    # 4. Total Charges vs Churn
    # Temporary numeric conversion for EDA plot
    tc_clean = pd.to_numeric(df["Total Charges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
    df_temp = df.copy()
    df_temp["Total Charges Clean"] = tc_clean

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(
        data=df_temp,
        x="Churn Label",
        y="Total Charges Clean",
        palette={"No": "#1f77b4", "Yes": "#d62728"},
        ax=ax,
        width=0.4,
    )
    ax.set_title("Total Charges ($) Distribution by Churn Status", fontsize=12, pad=10)
    ax.set_xlabel("Churn Status", fontsize=10)
    ax.set_ylabel("Total Charges ($)", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_total_charges_boxplot.png", dpi=300)
    plt.close()

    # 5. Contract Type Churn Rates
    contract_churn = df.groupby("Contract")["Churn Value"].agg(["count", "mean"]).reset_index()
    contract_churn["mean"] = contract_churn["mean"] * 100
    contract_churn = contract_churn.sort_values(by="mean", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(contract_churn["Contract"], contract_churn["mean"], color="#d62728", width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Churn Rate by Contract Commitment Type", fontsize=12, pad=10)
    ax.set_xlabel("Contract Type", fontsize=10)
    ax.set_ylabel("Churn Rate (%)", fontsize=10)
    ax.set_ylim(0, contract_churn["mean"].max() * 1.2)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_contract_churn_rate.png", dpi=300)
    plt.close()

    # 6. Internet Service & Tech Support Interaction
    fig, ax = plt.subplots(figsize=(8, 4.5))
    service_churn = df.groupby(["Internet Service", "Tech Support"])["Churn Value"].mean().reset_index()
    service_churn["Churn Rate (%)"] = service_churn["Churn Value"] * 100
    sns.barplot(
        data=service_churn,
        x="Internet Service",
        y="Churn Rate (%)",
        hue="Tech Support",
        palette="Blues_r",
        ax=ax,
    )
    ax.set_title("Churn Rate by Internet Service and Tech Support", fontsize=12, pad=10)
    ax.set_xlabel("Internet Service Type", fontsize=10)
    ax.set_ylabel("Churn Rate (%)", fontsize=10)
    ax.legend(title="Tech Support", loc="upper right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_internet_and_tech_support_churn.png", dpi=300)
    plt.close()

    # 7. Payment Method Churn Rate
    pm_churn = df.groupby("Payment Method")["Churn Value"].agg(["count", "mean"]).reset_index()
    pm_churn["Churn Rate (%)"] = pm_churn["mean"] * 100
    pm_churn = pm_churn.sort_values(by="Churn Rate (%)", ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(pm_churn["Payment Method"], pm_churn["Churn Rate (%)"], color="#ff7f0e", height=0.5)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", ha="left", va="center", fontweight="bold")
    ax.set_title("Churn Rate by Payment Method", fontsize=12, pad=10)
    ax.set_xlabel("Churn Rate (%)", fontsize=10)
    ax.set_xlim(0, pm_churn["Churn Rate (%)"].max() * 1.2)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "07_payment_method_churn.png", dpi=300)
    plt.close()

    # 8. Numerical Correlation Matrix
    corr_cols = ["Tenure Months", "Monthly Charges", "Total Charges Clean", "Churn Value"]
    corr_df = df_temp[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap: Numerical Features & Churn", fontsize=12, pad=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "08_numerical_correlation_heatmap.png", dpi=300)
    plt.close()

    # Save summary stats
    eda_stats = {
        "total_records": len(df),
        "churn_rate": float(df["Churn Value"].mean()),
        "tenure_median_non_churn": float(df[df["Churn Value"] == 0]["Tenure Months"].median()),
        "tenure_median_churn": float(df[df["Churn Value"] == 1]["Tenure Months"].median()),
        "monthly_charges_median_non_churn": float(df[df["Churn Value"] == 0]["Monthly Charges"].median()),
        "monthly_charges_median_churn": float(df[df["Churn Value"] == 1]["Monthly Charges"].median()),
        "month_to_month_churn_rate": float(df[df["Contract"] == "Month-to-month"]["Churn Value"].mean()),
        "one_year_churn_rate": float(df[df["Contract"] == "One year"]["Churn Value"].mean()),
        "two_year_churn_rate": float(df[df["Contract"] == "Two year"]["Churn Value"].mean()),
        "electronic_check_churn_rate": float(df[df["Payment Method"] == "Electronic check"]["Churn Value"].mean()),
        "fiber_optic_churn_rate": float(df[df["Internet Service"] == "Fiber optic"]["Churn Value"].mean()),
    }
    with open(PHASE_3_REPORTS_DIR / "eda_summary.json", "w") as f:
        json.dump(eda_stats, f, indent=2)

    logger.info("Phase 3 EDA complete. Figures and summary saved.")
    return eda_stats


def run_phase_4_preprocessing(df: pd.DataFrame):
    """Run Phase 4: clean dataset, perform stratified train-test split, and save processed splits."""
    logger.info("--- Starting Phase 4: Data Cleaning & Preprocessing ---")

    X_train, X_test, y_train, y_test = prepare_and_split_data(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        save_processed=True,
    )

    # Document quality stats
    quality_report = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features_count": len(ALL_FEATURE_COLS),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "missing_values_post_split": int(X_train.isna().sum().sum() + X_test.isna().sum().sum()),
    }
    with open(PHASE_4_REPORTS_DIR / "quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)

    logger.info(f"Phase 4 complete. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def run_phase_5_modeling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """Run Phase 5: baseline, model benchmark, tuning, threshold optimization, and final test evaluation."""
    logger.info("--- Starting Phase 5: Model Development & Evaluation ---")

    # 1. Benchmark candidate models via Stratified 5-Fold Cross-Validation
    logger.info("Evaluating candidate models with 5-fold Stratified CV...")
    cv_summary_df = cross_validate_models(X_train, y_train, cv_folds=5, random_state=RANDOM_STATE)
    cv_summary_df.to_csv(PHASE_5_REPORTS_DIR / "cv_model_comparison.csv", index=False)
    print("\n=== Model Benchmark Cross-Validation Summary ===")
    print(cv_summary_df.to_string(index=False))

    # 2. Select and tune best tree ensemble model
    # We compare HistGradientBoosting and Logistic Regression
    logger.info("Tuning HistGradientBoosting Classifier...")
    best_pipeline, best_params = tune_best_model(
        X_train, y_train, model_type="HistGradientBoosting", cv_folds=5, random_state=RANDOM_STATE
    )

    # 3. Fit best pipeline on full training dataset
    logger.info("Fitting best pipeline on full training set...")
    best_pipeline.fit(X_train, y_train)

    # 4. Out-of-fold / Training Probabilities for Threshold Optimization
    # We calibrate threshold on training set predictions (using cross_val_predict or internal calibration)
    from sklearn.model_selection import cross_val_predict
    train_probs = cross_val_predict(
        best_pipeline, X_train, y_train, cv=5, method="predict_proba"
    )[:, 1]

    sweep_df = sweep_thresholds(y_train.values, train_probs)
    sweep_df.to_csv(PHASE_5_REPORTS_DIR / "threshold_sweep_train.csv", index=False)

    # Optimal threshold selection: Maximize F1 or balance Recall with Precision
    # In churn management, missing churners is costly, so threshold around 0.35-0.45 boosts recall significantly while keeping precision stable
    best_f1_idx = sweep_df["f1"].idxmax()
    f1_threshold = float(sweep_df.loc[best_f1_idx, "threshold"])
    
    # Let's inspect threshold balancing recall >= 0.75 or optimal F1
    # We select optimal F1 threshold
    optimal_threshold = f1_threshold
    logger.info(f"Optimal Decision Threshold identified from CV: {optimal_threshold:.2f} (F1 = {sweep_df.loc[best_f1_idx, 'f1']:.4f}, Recall = {sweep_df.loc[best_f1_idx, 'recall']:.4f})")

    # Plot threshold trade-offs
    plot_threshold_tradeoffs(
        sweep_df,
        optimal_threshold=optimal_threshold,
        save_path=FIGURES_DIR / "09_threshold_tradeoffs.png",
    )
    plt.close()

    # 5. Final Evaluation on Untouched Test Set
    logger.info("Evaluating final model on untouched test set...")
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]

    # Metrics at standard 0.50 threshold
    metrics_std = evaluate_predictions(y_test.values, test_probs, threshold=0.50)
    # Metrics at calibrated threshold
    metrics_opt = evaluate_predictions(y_test.values, test_probs, threshold=optimal_threshold)

    final_metrics = {
        "model_name": "HistGradientBoostingClassifier",
        "best_hyperparameters": best_params,
        "standard_threshold_0.50": metrics_std,
        "calibrated_threshold": metrics_opt,
    }
    with open(PHASE_5_REPORTS_DIR / "final_test_evaluation.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n=== Final Test Set Performance ===")
    print(f"ROC-AUC: {metrics_opt['roc_auc']:.4f} | PR-AUC: {metrics_opt['pr_auc']:.4f}")
    print(f"At Calibrated Threshold ({optimal_threshold:.2f}):")
    print(f"  Recall:    {metrics_opt['recall']:.4f}  (TP={metrics_opt['tp']}, FN={metrics_opt['fn']})")
    print(f"  Precision: {metrics_opt['precision']:.4f}  (FP={metrics_opt['fp']})")
    print(f"  F1-Score:  {metrics_opt['f1']:.4f}")
    print(f"  Accuracy:  {metrics_opt['accuracy']:.4f}")

    # Plot final confusion matrices (Standard vs Calibrated)
    cm_opt = np.array([[metrics_opt["tn"], metrics_opt["fp"]], [metrics_opt["fn"], metrics_opt["tp"]]])
    plot_confusion_matrix(
        cm_opt,
        model_name=f"HistGradientBoosting (Threshold = {optimal_threshold:.2f})",
        save_path=FIGURES_DIR / "10_test_confusion_matrix_calibrated.png",
    )
    plt.close()

    cm_std = np.array([[metrics_std["tn"], metrics_std["fp"]], [metrics_std["fn"], metrics_std["tp"]]])
    plot_confusion_matrix(
        cm_std,
        model_name="HistGradientBoosting (Threshold = 0.50)",
        save_path=FIGURES_DIR / "11_test_confusion_matrix_standard.png",
    )
    plt.close()

    # Plot final ROC and PR curves
    plot_roc_and_pr_curves(
        y_test.values,
        test_probs,
        model_name="HistGradientBoosting (Test Set)",
        save_path=FIGURES_DIR / "12_test_roc_pr_curves.png",
    )
    plt.close()

    # 6. Feature Importance via Permutation Importance on Test Set
    logger.info("Computing Permutation Importance on test set...")
    perm_result = permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )

    perm_df = pd.DataFrame({
        "Feature": ALL_FEATURE_COLS,
        "Importance Mean": perm_result.importances_mean,
        "Importance Std": perm_result.importances_std,
    }).sort_values(by="Importance Mean", ascending=False)
    perm_df.to_csv(PHASE_5_REPORTS_DIR / "permutation_importance.csv", index=False)

    # Plot top features
    fig, ax = plt.subplots(figsize=(8.5, 6))
    top_perm = perm_df.head(12).sort_values(by="Importance Mean", ascending=True)
    ax.barh(top_perm["Feature"], top_perm["Importance Mean"], xerr=top_perm["Importance Std"], color="#1f77b4", height=0.5, capsize=3)
    ax.set_title("Permutation Feature Importance (ROC-AUC Impact on Test Set)", fontsize=12, pad=10)
    ax.set_xlabel("Mean Decrease in ROC-AUC", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "13_permutation_feature_importance.png", dpi=300)
    plt.close()

    # 7. Persist complete model bundle
    bundle_path = save_model_bundle(
        pipeline=best_pipeline,
        threshold=optimal_threshold,
        metrics=final_metrics,
        metadata={
            "description": "Customer Churn Prediction Model (HistGradientBoosting)",
            "train_observations": len(X_train),
            "test_observations": len(X_test),
            "best_params": best_params,
        },
        save_path=MODELS_DIR / "churn_pipeline_bundle.joblib",
    )

    # 8. Test inference wrapper
    logger.info("Testing ChurnPredictor inference wrapper on sample records...")
    predictor = ChurnPredictor(bundle_path)
    sample_test = X_test.head(5).copy()
    scored_sample = predictor.score_records(sample_test)
    print("\n=== Sample Model Inference Output ===")
    print(scored_sample[["Tenure Months", "Contract", "Monthly Charges", "churn_probability", "churn_prediction", "churn_label", "risk_tier"]])

    logger.info("Phase 5 completed successfully.")
    return final_metrics


def main():
    raw_df = load_raw_data()
    run_phase_3_eda(raw_df)
    X_train, X_test, y_train, y_test = run_phase_4_preprocessing(raw_df)
    run_phase_5_modeling(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
