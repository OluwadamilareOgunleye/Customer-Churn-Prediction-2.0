# Technical Case Study: Production-Grade Customer Churn Prediction & Retention Optimization

## 1. Problem Context & Business Stakes
Subscription churn directly threatens recurring revenue in telecommunications. In the IBM Telco benchmark, $26.5\%$ of customers churned. For an enterprise with 100,000 subscribers and an average Customer Lifetime Value (CLV) of \$1,200, an unmitigated churn rate represents \$31.8M in annual revenue erosion.

Standard machine learning models evaluated at default $\tau = 0.50$ decision boundaries miss nearly half of all churners because binary classifiers minimize overall error rather than the asymmetric cost of false negatives.

---

## 2. Key Engineering & Methodological Decisions

### Decision 1: Target Leakage Remediation
- **Problem:** Raw dataset contains exit-interview feedback (`Churn Reason`), IBM's internal model predictions (`Churn Score`), and post-hoc `CLTV`.
- **Engineering Solution:** Strictly pruned all leakage columns in `src/churn_prediction/config.py` before cross-validation split, ensuring true real-world generalizability.

### Decision 2: Model Architecture Selection
- **Evaluated Algorithms:** Baseline Dummy, Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, and XGBoost.
- **Winner:** `HistGradientBoostingClassifier` achieved highest 5-fold CV ROC-AUC ($0.8639$) and test set ROC-AUC ($0.8570$) with native binning, fast CPU convergence ($<40$s complete run), and zero native C++ compilation overhead on diverse platforms.

### Decision 3: Operating Threshold Calibration ($\tau = 0.30$)
- **Tradeoff:** Shifting the threshold from $0.50$ to $0.30$ increases recall from $52.7\%$ to $77.01\%$ (capturing 288 out of 374 true churners in the holdout test set).
- **Economic Value:** Reduces unmitigated churn losses by $\$109,200$ and yields a **$+\$15,700$ net profit advantage** in targeted campaign ROI over default classification.

### Decision 4: Experiment Tracking & Model Registry
- **Implementation:** Embedded MLflow with SQLite backend (`sqlite:///mlflow.db`) avoiding external server dependencies.
- **Governance:** JSON model registry (`models/registry.json`) enforcing strict release states (`Staging` vs `Production`) and prohibiting unreviewed automatic model replacement.

### Decision 5: Stateless Microservice Serving
- **Backend:** FastAPI with Pydantic schema validation.
- **Performance:** Measured p50 latency of **$36.4$ ms** for single predictions, and **$1.11$ s for 1,409 batch records** ($0.79$ ms/record).
- **Resilience:** Fallback engine in client allowing direct inference if HTTP connection drops.

---

## 3. Verified Metrics & Quantitative Scorecard

| Dimension | Measured Benchmark | Status |
| :--- | :--- | :--- |
| **Discriminative Ability** | Test Set ROC-AUC: **$0.8570$** | Exceeds Target ($>0.82$) |
| **Precision-Recall Area** | Test Set PR-AUC: **$0.6740$** | Exceeds Target ($>0.60$) |
| **Sensitivity at Operating Threshold** | Recall: **$77.01\%$** ($\tau = 0.30$) | Exceeds Target ($>75\%$) |
| **Probability Calibration** | Brier Score: **$0.1322$** | Near-optimal calibration |
| **Inference Latency** | Single: **$36.4$ ms** (p50); Batch: **$0.79$ ms/rec** | Low latency |
| **Test Suite Quality** | **34 / 34 passing** ($100\%$ pass rate) | Fully Verified |
| **Data Drift (PSI)** | Max feature PSI: **$0.0124$** ($< 0.10$ threshold) | Stable |
