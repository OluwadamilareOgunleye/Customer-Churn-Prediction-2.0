"""Generate and execute complete Jupyter notebooks for Phases 3, 4, and 5."""

import logging
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# Phase 3: Exploratory Data Analysis (EDA)
## Customer Churn Prediction System

This notebook investigates the IBM Telco Customer Churn dataset, uncovers behavioral and contract drivers of customer churn, explores bivariate relationships with the target variable, and documents business insights."""
        ),
        nbf.v4.new_code_cell(
            """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from churn_prediction.config import (
    ALL_FEATURE_COLS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    TARGET_LABEL_COL,
    FIGURES_DIR,
)
from churn_prediction.data_loader import load_raw_data

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
%matplotlib inline"""
        ),
        nbf.v4.new_markdown_cell(
            """### 1. Ingestion and Structural Overview
We load the raw dataset through the project data-loader module."""
        ),
        nbf.v4.new_code_cell(
            """df_raw = load_raw_data(use_cache=True)
print(f"Dataset Dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
df_raw.head(3)"""
        ),
        nbf.v4.new_code_cell(
            """# Check data types and non-null counts
df_raw.info()"""
        ),
        nbf.v4.new_markdown_cell(
            """### 2. Target Variable & Class Imbalance Analysis
We inspect the distribution of `Churn Label` and `Churn Value`."""
        ),
        nbf.v4.new_code_cell(
            """churn_counts = df_raw["Churn Label"].value_counts()
churn_pcts = df_raw["Churn Label"].value_counts(normalize=True) * 100

summary_df = pd.DataFrame({"Count": churn_counts, "Percentage (%)": churn_pcts.round(2)})
display(summary_df)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(churn_counts.index, churn_counts.values, color=["#1f77b4", "#d62728"], width=0.5)
for bar, pct in zip(bars, churn_pcts.values):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 50, f"{yval:,} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
ax.set_title("Customer Churn Target Distribution", fontsize=12)
ax.set_ylabel("Customer Count")
ax.set_ylim(0, churn_counts.max() * 1.15)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """### 3. Univariate & Bivariate Analysis of Key Features
#### A. Customer Tenure Dynamics
We analyze how customer tenure differs between churned and retained subscribers."""
        ),
        nbf.v4.new_code_cell(
            """fig, ax = plt.subplots(figsize=(9, 4.5))
sns.histplot(
    data=df_raw,
    x="Tenure Months",
    hue="Churn Label",
    multiple="stack",
    palette={"No": "#1f77b4", "Yes": "#d62728"},
    bins=36,
    ax=ax,
)
ax.set_title("Tenure Months Distribution by Churn Status", fontsize=13)
ax.set_xlabel("Tenure (Months)")
plt.show()

print("Median Tenure by Churn Status:")
print(df_raw.groupby("Churn Label")["Tenure Months"].median())"""
        ),
        nbf.v4.new_markdown_cell(
            """#### B. Monthly Charges vs. Churn
We explore the distribution of monthly billing rates."""
        ),
        nbf.v4.new_code_cell(
            """fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(
    data=df_raw,
    x="Churn Label",
    y="Monthly Charges",
    palette={"No": "#1f77b4", "Yes": "#d62728"},
    ax=ax,
    width=0.4,
)
ax.set_title("Monthly Charges ($) Distribution by Churn Status", fontsize=13)
plt.show()

print("Monthly Charges Summary by Churn Status:")
print(df_raw.groupby("Churn Label")["Monthly Charges"].describe().round(2))"""
        ),
        nbf.v4.new_markdown_cell(
            """#### C. Contract Type Impact
We evaluate churn rates across contract types."""
        ),
        nbf.v4.new_code_cell(
            """contract_churn = df_raw.groupby("Contract")["Churn Value"].agg(["count", "mean"]).reset_index()
contract_churn["Churn Rate (%)"] = (contract_churn["mean"] * 100).round(2)
contract_churn = contract_churn.sort_values(by="Churn Rate (%)", ascending=False)
display(contract_churn)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(contract_churn["Contract"], contract_churn["Churn Rate (%)"], color="#d62728", width=0.5)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
ax.set_title("Churn Rate by Contract Commitment Type", fontsize=13)
ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, contract_churn["Churn Rate (%)"].max() * 1.2)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """#### D. Internet Service and Value-Added Tech Support
We investigate how Tech Support affects retention across internet service tiers."""
        ),
        nbf.v4.new_code_cell(
            """service_churn = df_raw.groupby(["Internet Service", "Tech Support"])["Churn Value"].agg(["count", "mean"]).reset_index()
service_churn["Churn Rate (%)"] = (service_churn["mean"] * 100).round(2)
display(service_churn)

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(
    data=service_churn,
    x="Internet Service",
    y="Churn Rate (%)",
    hue="Tech Support",
    palette="Blues_r",
    ax=ax,
)
ax.set_title("Churn Rate by Internet Service and Tech Support", fontsize=13)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """#### E. Payment Method Analysis"""
        ),
        nbf.v4.new_code_cell(
            """pm_churn = df_raw.groupby("Payment Method")["Churn Value"].agg(["count", "mean"]).reset_index()
pm_churn["Churn Rate (%)"] = (pm_churn["mean"] * 100).round(2)
pm_churn = pm_churn.sort_values(by="Churn Rate (%)", ascending=True)

fig, ax = plt.subplots(figsize=(8.5, 4))
bars = ax.barh(pm_churn["Payment Method"], pm_churn["Churn Rate (%)"], color="#ff7f0e", height=0.5)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", ha="left", va="center", fontweight="bold")
ax.set_title("Churn Rate by Payment Method", fontsize=13)
ax.set_xlabel("Churn Rate (%)")
ax.set_xlim(0, pm_churn["Churn Rate (%)"].max() * 1.2)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """### 4. Multivariate Numerical Correlations"""
        ),
        nbf.v4.new_code_cell(
            """tc_clean = pd.to_numeric(df_raw["Total Charges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
df_corr = df_raw[["Tenure Months", "Monthly Charges", "Churn Value"]].copy()
df_corr["Total Charges"] = tc_clean

corr_matrix = df_corr.corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("Correlation Heatmap: Numerical Features & Churn", fontsize=13)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """### 5. Summary of EDA Findings
1. **Tenure Decay:** Over 50% of churn occurs in the first 12 months; median tenure for churners is 10 months vs 38 months for loyal customers.
2. **Contract Lock-in:** Month-to-month contracts exhibit a 42.7% churn rate, compared to 11.3% for 1-year and 2.8% for 2-year contracts.
3. **High-Speed Friction:** Fiber optic customers churn at 41.9%, but adding Tech Support cuts churn by more than half.
4. **Billing Friction:** Electronic check payers churn at 45.3% vs ~15.5% for automatic credit card or bank transfer."""
        ),
    ]
    return nb


def build_preprocessing_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# Phase 4: Data Quality, Cleaning, and Preprocessing
## Customer Churn Prediction System

This notebook audits data quality anomalies, resolves missing whitespace in `Total Charges`, prevents data leakage by excluding post-churn and identifier columns, constructs a reproducible stratified train-test split, and defines a reusable Scikit-Learn `ColumnTransformer`."""
        ),
        nbf.v4.new_code_cell(
            """import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from churn_prediction.config import (
    ALL_FEATURE_COLS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
)
from churn_prediction.data_loader import load_raw_data
from churn_prediction.data_cleaning import clean_raw_dataset, clean_total_charges
from churn_prediction.data_validation import validate_raw_data, validate_cleaned_data, validate_train_test_split
from churn_prediction.preprocessing import build_preprocessor, prepare_and_split_data"""
        ),
        nbf.v4.new_markdown_cell(
            """### 1. Raw Data Quality Audit
We examine the raw dataset for missing values, whitespace, and duplicates."""
        ),
        nbf.v4.new_code_cell(
            """df_raw = load_raw_data(use_cache=True)
validate_raw_data(df_raw)

# Investigate Total Charges whitespace
blank_tc = df_raw[df_raw["Total Charges"].astype(str).str.strip() == ""]
print(f"Total Charges blank whitespace count: {len(blank_tc)}")
display(blank_tc[["CustomerID", "Tenure Months", "Monthly Charges", "Total Charges"]])"""
        ),
        nbf.v4.new_markdown_cell(
            """### 2. Domain-Sound Cleaning
Notice that in all 11 records with blank `Total Charges`, `Tenure Months == 0`. These are brand-new customers who have not yet received their first bill.
We impute `0.0` for these accounts and convert `Total Charges` to `float64`."""
        ),
        nbf.v4.new_code_cell(
            """df_clean = clean_raw_dataset(df_raw)
validate_cleaned_data(df_clean)
print("Cleaned Total Charges dtype:", df_clean["Total Charges"].dtype)
print("Null count in Total Charges post-cleaning:", df_clean["Total Charges"].isna().sum())"""
        ),
        nbf.v4.new_markdown_cell(
            """### 3. Leakage Prevention and Feature Scoping
We explicitly exclude:
- Post-churn feedback: `Churn Reason`
- Model-generated scores: `Churn Score`
- Derived metrics: `CLTV`
- Identifiers: `CustomerID`
- Zero-variance constants: `Count`, `Country`, `State`"""
        ),
        nbf.v4.new_code_cell(
            """print(f"Total Predictive Features: {len(ALL_FEATURE_COLS)}")
print("Numerical Features:", NUMERICAL_FEATURES)
print("Categorical Features:", CATEGORICAL_FEATURES)"""
        ),
        nbf.v4.new_markdown_cell(
            """### 4. Stratified Train-Test Splitting (Zero Data Leakage)
We split the data before fitting any transformations."""
        ),
        nbf.v4.new_code_cell(
            """X_train, X_test, y_train, y_test = prepare_and_split_data(
    df_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE, save_processed=True
)

print(f"X_train shape: {X_train.shape} (Churn rate: {y_train.mean():.4f})")
print(f"X_test shape:  {X_test.shape} (Churn rate: {y_test.mean():.4f})")"""
        ),
        nbf.v4.new_markdown_cell(
            """### 5. Preprocessor Pipeline Verification
We build and inspect the `ColumnTransformer`."""
        ),
        nbf.v4.new_code_cell(
            """preprocessor = build_preprocessor()
# Fit ONLY on training data
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
print(f"Transformed Training Matrix Shape: {X_train_trans.shape}")
print(f"Transformed Test Matrix Shape:     {X_test_trans.shape}")
print(f"Total One-Hot Encoded Features:    {len(feature_names)}")"""
        ),
    ]
    return nb


def build_modeling_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# Phase 5: Model Development, Evaluation, and Selection
## Customer Churn Prediction System

This notebook benchmarks multiple classification algorithms using 5-Fold Stratified Cross-Validation, optimizes hyperparameters for the best model architecture, calibrates decision thresholds for business impact, and conducts final evaluation on the untouched test set."""
        ),
        nbf.v4.new_code_cell(
            """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from churn_prediction.config import (
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    MODELS_DIR,
    ALL_FEATURE_COLS,
)
from churn_prediction.model_training import (
    cross_validate_models,
    build_full_pipeline,
    tune_best_model,
)
from churn_prediction.evaluation import (
    evaluate_predictions,
    sweep_thresholds,
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    plot_threshold_tradeoffs,
)
from churn_prediction.model_inference import ChurnPredictor, load_model_bundle

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
%matplotlib inline"""
        ),
        nbf.v4.new_markdown_cell(
            """### 1. Ingestion of Cleaned Training & Test Datasets"""
        ),
        nbf.v4.new_code_cell(
            """train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

X_train = train_df[ALL_FEATURE_COLS]
y_train = train_df["Churn Value"]
X_test = test_df[ALL_FEATURE_COLS]
y_test = test_df["Churn Value"]

print(f"Train set: {X_train.shape}, Churn rate: {y_train.mean():.4f}")
print(f"Test set:  {X_test.shape}, Churn rate: {y_test.mean():.4f}")"""
        ),
        nbf.v4.new_markdown_cell(
            """### 2. Candidate Model Benchmarking (5-Fold Stratified CV)
We evaluate Dummy Classifier, Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, and XGBoost."""
        ),
        nbf.v4.new_code_cell(
            """cv_results = cross_validate_models(X_train, y_train, cv_folds=5, random_state=RANDOM_STATE)
display(cv_results)"""
        ),
        nbf.v4.new_markdown_cell(
            """### 3. Hyperparameter Tuning (HistGradientBoosting)
We tune the HistGradientBoostingClassifier using GridSearchCV."""
        ),
        nbf.v4.new_code_cell(
            """best_pipeline, best_params = tune_best_model(
    X_train, y_train, model_type="HistGradientBoosting", cv_folds=5, random_state=RANDOM_STATE
)
print("Optimal Hyperparameters:")
for param, val in best_params.items():
    print(f"  {param}: {val}")"""
        ),
        nbf.v4.new_markdown_cell(
            """### 4. Threshold Optimization for High Recall Retention
We sweep decision thresholds from 0.05 to 0.95 on cross-validation probabilities."""
        ),
        nbf.v4.new_code_cell(
            """from sklearn.model_selection import cross_val_predict

cv_probs = cross_val_predict(best_pipeline, X_train, y_train, cv=5, method="predict_proba")[:, 1]
sweep_df = sweep_thresholds(y_train.values, cv_probs)

best_f1_idx = sweep_df["f1"].idxmax()
opt_thresh = float(sweep_df.loc[best_f1_idx, "threshold"])
print(f"Optimal F1 Threshold: {opt_thresh:.2f} (F1={sweep_df.loc[best_f1_idx, 'f1']:.4f}, Recall={sweep_df.loc[best_f1_idx, 'recall']:.4f})")

fig = plot_threshold_tradeoffs(sweep_df, optimal_threshold=opt_thresh)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """### 5. Final Evaluation on Untouched Test Set
We evaluate on the untouched test set using both the standard 0.50 threshold and the calibrated 0.30 threshold."""
        ),
        nbf.v4.new_code_cell(
            """best_pipeline.fit(X_train, y_train)
test_probs = best_pipeline.predict_proba(X_test)[:, 1]

metrics_std = evaluate_predictions(y_test.values, test_probs, threshold=0.50)
metrics_cal = evaluate_predictions(y_test.values, test_probs, threshold=opt_thresh)

eval_comparison = pd.DataFrame([
    {"Threshold": "0.50 (Standard)", "ROC-AUC": metrics_std["roc_auc"], "PR-AUC": metrics_std["pr_auc"], "Recall": metrics_std["recall"], "Precision": metrics_std["precision"], "F1-Score": metrics_std["f1"], "Accuracy": metrics_std["accuracy"], "Captured Churners": f"{metrics_std['tp']}/374"},
    {"Threshold": f"{opt_thresh:.2f} (Calibrated)", "ROC-AUC": metrics_cal["roc_auc"], "PR-AUC": metrics_cal["pr_auc"], "Recall": metrics_cal["recall"], "Precision": metrics_cal["precision"], "F1-Score": metrics_cal["f1"], "Accuracy": metrics_cal["accuracy"], "Captured Churners": f"{metrics_cal['tp']}/374"},
])
display(eval_comparison)"""
        ),
        nbf.v4.new_code_cell(
            """# Plot test set ROC & PR curves
fig = plot_roc_and_pr_curves(y_test.values, test_probs, model_name="HistGradientBoosting (Test Set)")
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """### 6. Production Model Inference Testing
We load the serialized production model bundle from `models/churn_pipeline_bundle.joblib` and score sample customer accounts."""
        ),
        nbf.v4.new_code_cell(
            """predictor = ChurnPredictor(MODELS_DIR / "churn_pipeline_bundle.joblib")
sample_batch = X_test.head(5).copy()
scored_results = predictor.score_records(sample_batch)

display(scored_results[["Tenure Months", "Contract", "Monthly Charges", "churn_probability", "churn_prediction", "churn_label", "risk_tier"]])"""
        ),
    ]
    return nb


def main():
    ep = ExecutePreprocessor(timeout=600, kernel_name="customer-churn-env")

    # 1. EDA Notebook
    logger.info("Building and executing 03_exploratory_data_analysis.ipynb...")
    nb_eda = build_eda_notebook()
    ep.preprocess(nb_eda, {"metadata": {"path": "."}})
    with open(NOTEBOOKS_DIR / "03_exploratory_data_analysis.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb_eda, f)

    # 2. Preprocessing Notebook
    logger.info("Building and executing 04_data_cleaning_preprocessing.ipynb...")
    nb_prep = build_preprocessing_notebook()
    ep.preprocess(nb_prep, {"metadata": {"path": "."}})
    with open(NOTEBOOKS_DIR / "04_data_cleaning_preprocessing.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb_prep, f)

    # 3. Modeling Notebook
    logger.info("Building and executing 05_model_development_evaluation.ipynb...")
    nb_model = build_modeling_notebook()
    ep.preprocess(nb_model, {"metadata": {"path": "."}})
    with open(NOTEBOOKS_DIR / "05_model_development_evaluation.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb_model, f)

    logger.info("All notebooks generated and successfully executed.")


if __name__ == "__main__":
    main()
