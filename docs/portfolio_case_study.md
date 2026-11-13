# Machine Learning Engineering Case Study: Telecom Customer Churn Prediction System

## 1. Executive Summary & Business Context

In the subscription telecommunications sector, acquiring a new customer costs 5 to 7 times more than retaining an existing subscriber. When customers defect early in their lifecycle, upfront equipment subsidies, onboarding costs, and sales commissions fail to amortize, leading to a direct erosion of operating margin and Customer Lifetime Value (CLV).

Historically, retention departments have operated reactively—reaching out only after a subscriber submits a cancellation request, which is often too late, or extending broad, uncalibrated discounts across entire subscriber cohorts, which burns margin on customers who never planned to leave.

This project delivers a **production-ready Customer Churn Prediction and Retention Optimization System** built on the IBM Telco Customer Churn dataset (7,043 customer accounts). Rather than treating model training as a kaggle-style exercise in metric chasing, this system implements an end-to-end engineering workflow: from leakage-free feature engineering and stratified cross-validation to business-driven threshold optimization, SHAP interpretability, a FastAPI REST microservice, a Streamlit dashboard, Docker containerization, and population drift monitoring.

---

## 2. Key Exploratory Findings & Retention Dynamics

Exploratory Data Analysis revealed four foundational business dynamics governing customer churn:

1. **The 12-Month Hazard Window:**
   Over 50% of customer churn is concentrated during the first year of service. The median tenure for churned subscribers is **10 months**, compared to **38 months** for retained accounts. After 24 months, defection rates decay sharply.
2. **Contract Type as the Strongest Retention Anchor:**
   Subscribers on month-to-month contracts churn at **42.71%**, compared to **11.27%** for one-year contracts and **2.83%** for two-year contracts—a 15-fold difference. Over 88.5% of all churners held month-to-month agreements.
3. **The Fiber Optic Support Shield:**
   Fiber optic broadband users exhibited an unexpectedly high churn rate of **41.89%** despite paying premium rates ($70–$100/mo). However, when Fiber Optic customers had **Tech Support**, churn dropped by more than half to ~22%. Technical assistance serves as an essential operational buffer against service complexity and connection friction.
4. **Billing Friction:**
   Subscribers paying via manual electronic checks churned at **45.29%**, compared to **15.24%** for customers enrolled in automated credit card or bank transfer billing.

---

## 3. Data Cleaning & Zero-Leakage Preprocessing Design

### Data Quality Remediation
During ingestion, an audit of `Total Charges` revealed 11 blank whitespace values (`" "`). Cross-referencing these accounts showed that in every case, `Tenure Months == 0`. These represented newly activated subscribers who joined during the current billing cycle and had not yet received their first bill. Rather than dropping valid rows or imputing an arbitrary mean, we assigned `Total Charges = 0.0` and coerced the feature to `float64`.

### Strict Leakage Prevention
To guarantee that the model cannot cheat by exploiting future information:
- **`Churn Reason`** was excluded: Only collected during exit interviews post-churn (5,174 missing values for active customers).
- **`Churn Score`** was excluded: IBM's pre-computed model risk score.
- **`CustomerID`, `Count`, `Country`, `State`** were dropped due to zero predictive variance or non-generalizable high cardinality.
- **`ColumnTransformer` Isolation:** Numerical transformations (`SimpleImputer` + `StandardScaler`) and nominal transformations (`OneHotEncoder(handle_unknown="ignore")`) were wrapped in a single Scikit-Learn pipeline and fitted **only** on training splits, eliminating test-to-train data leakage.

---

## 4. Model Benchmarking, Selection & Hyperparameter Tuning

We benchmarked 6 candidate architectures using 5-Fold Stratified Cross-Validation on the training set (5,634 rows):

| Model Architecture | Cross-Val ROC-AUC | Cross-Val PR-AUC | Cross-Val Recall | Cross-Val F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 0.8624 | 0.6805 | 0.8047 | 0.6494 |
| **XGBoost** | 0.8624 | 0.6878 | 0.8127 | 0.6433 |
| **Logistic Regression** | 0.8588 | 0.6789 | 0.8100 | 0.6401 |
| **HistGradientBoosting** | 0.8494 | 0.6576 | 0.7338 | 0.6370 |
| **Decision Tree** | 0.8453 | 0.6254 | 0.8127 | 0.6295 |
| **Dummy (Majority Class)**| 0.5000 | 0.2654 | 0.0000 | 0.0000 |

### Why HistGradientBoosting Was Selected:
1. **Generalization:** After tuning via `GridSearchCV` (`max_iter=100`, `learning_rate=0.03`, `max_leaf_nodes=15`, `min_samples_leaf=50`), its cross-validation ROC-AUC rose to **0.8639** with the lowest standard deviation across folds ($\sigma = 0.0061$).
2. **Speed & Scalability:** Native C-level histogram binning enables microsecond CPU inference without external runtime dependencies.
3. **Probability Calibration:** Leaving `class_weight=None` preserved smooth, well-calibrated posterior probabilities essential for business threshold tuning.

---

## 5. Threshold Optimization: The Economics of Retention

In churn prediction, the default classification threshold of $\tau = 0.50$ is rarely optimal because false negatives (losing a customer worth hundreds of dollars in lifetime margin) are far more costly than false positives (sending a retention email or discount voucher).

We ran a 91-point threshold sweep on out-of-fold cross-validation probabilities and identified **$\tau = 0.30$** as the optimal operating point:

```
Test Set Confusion Matrix Comparison (1,409 Test Accounts, 374 Actual Churners)

Standard Threshold (τ = 0.50)           Calibrated Threshold (τ = 0.30)
        Actual: No   Actual: Yes                Actual: No   Actual: Yes
Pred No    939 (TN)    167 (FN)        Pred No    792 (TN)     86 (FN)
Pred Yes    96 (FP)    207 (TP)        Pred Yes   243 (FP)    288 (TP)
```

- **Recall rose from 55.35% to 77.01%**: Captured **81 additional churners** (reducing missed churners from 167 down to 86, a 48.5% reduction).
- **Final Test Metrics:** **ROC-AUC: 0.8570**, **PR-AUC: 0.6740**, **F1-Score: 0.6365**, **Accuracy: 76.65%**.

---

## 6. Model Explainability with SHAP

Using cooperative game theory via TreeSHAP:
- **Global Drivers:** `Tenure Months` (negative log-odds impact), `Contract: Month-to-month` (positive impact), `Monthly Charges` (positive impact), and `Internet Service: Fiber optic` (positive impact) dominate model decisions.
- **Demographic Parity:** `Gender` and `Senior Citizen` features ranked in the bottom 3 features with near-zero attribution, verifying algorithmic fairness and ensuring retention programs do not discriminate based on protected demographic traits.
- **Local Attribution:** The system outputs waterfall attributions for individual accounts, allowing customer success representatives to identify specific churn risk drivers before contacting a subscriber.

---

## 7. Engineering Architecture & Production Stack

- **Decoupled Backend:** Built with **FastAPI**, featuring Pydantic request validation, CORS middleware, health probes, and batch inference endpoints.
- **Interactive UI:** Built with **Streamlit**, featuring single-customer simulation, batch CSV file upload/scoring with one-click export, and performance diagnostic curve viewing.
- **Containerization:** Multi-stage **Docker** and **Docker Compose** configuration with non-root user setup and container-to-container healthcheck dependency orchestration.
- **Testing & Quality Assurance:** Comprehensive test suite (28 passing tests in pytest) covering schema validation, leakage prevention, API contracts, and inference fallbacks with 92% coverage across active serving modules.
- **Observability:** Telemetry logger capturing non-PII prediction metadata, coupled with a Population Stability Index (PSI) drift engine capable of alerting on feature distribution shifts.

---

## 8. Technical Challenges & Lessons Learned

1. **Handling 0-Tenure Blank Total Charges:** Inexperienced data practitioners often delete rows with missing values or fill them with median charges, corrupting the tenure distribution. Investigating the business context revealed that 0 tenure logically equates to $0 total spend.
2. **The Accuracy Paradox:** The Dummy classifier scored 73.46% accuracy with zero predictive utility. Prioritizing ROC-AUC, PR-AUC, and calibrated Recall ensured the business problem was addressed effectively.
3. **Threshold Calibration as a Value Multiplier:** Machine learning models output continuous probabilities, but business systems take discrete actions. Calibrating the decision threshold to 0.30 halved the number of lost customers compared to off-the-shelf classification defaults.

---

## 9. Future Roadmap

1. **Real-Time Event Streaming:** Integrate Apache Kafka to score customer telemetry in real time based on recent dropped calls or bandwidth drops.
2. **Causal ML / Uplift Modeling:** Transition from predicting churn probability to predicting *intervention uplift*—identifying which customers are positively influenced by a retention offer versus those who would renew anyway.
3. **Continuous Retraining with Feature Store:** Deploy Feast to serve feature values with automated drift-triggered retraining pipelines in Kubeflow.
