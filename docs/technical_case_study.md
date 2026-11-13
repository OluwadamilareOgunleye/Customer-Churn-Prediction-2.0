# Comprehensive Technical Case Study: Production Customer Churn System

## 1. Executive Summary & Business Stakes

In subscription telecommunications, recurring monthly revenue is the core driver of enterprise valuation. When a subscriber disconnects service, the operator loses future cash flow and incurs the steep cost of customer re-acquisition—frequently estimated at 5 to 7 times the cost of proactive retention outreach.

In the IBM Telco benchmark ($N = 7,043$), $26.54\%$ of subscribers churned. In an enterprise context with $100,000$ accounts and an annual Customer Lifetime Value (CLV) of $\$1,200$, an unmanaged churn rate represents $\$31.8\text{M}$ in recurring revenue erosion.

Standard off-the-shelf classification algorithms evaluated at default $\tau = 0.50$ thresholds optimize for symmetrical accuracy rather than asymmetric business costs. At $\tau = 0.50$, models miss nearly half of all departing subscribers. This project built a calibrated, production-grade machine learning system that cut missed churners by **$51.4\%$** while generating an estimated **$+\$15,700$ net profit advantage** per campaign over default classification.

---

## 2. Data Science Methodology

### Exploratory Data Analysis & Behavioral Drivers
Exploratory analysis revealed four critical hazard patterns:
1. **The 12-Month Hazard Peak:** Over $50\%$ of all churn occurs during months 1–12 (median tenure of churners is 10 months vs 38 months for retained accounts).
2. **Contract Lock-in Dynamics:** Month-to-month contracts experience a **$42.56\%$** churn rate, compared to **$12.00\%$** for 1-year and **$2.68\%$** for 2-year contracts.
3. **The Fiber Optic Support Shield:** Fiber optic internet subscribers churn at **$41.11\%$** despite paying premium monthly fees (\$70–\$100). However, bundling Tech Support cuts churn by more than half (~22%), demonstrating that technical assistance anchors customer retention.
4. **Payment Friction:** Customers paying via electronic check churn at **$43.46\%$**, whereas customers on automated bank transfers churn at only **$19.00\%$**.

### Leakage-Free Preprocessing & Integrity
- **Cryptographic Provenance:** Raw Excel data verified with SHA-256 fingerprint: `1BCBC0CCC9B352175216979102628E579CFBDE2C3FF57B005DE168A433122640`.
- **Target Leakage Remediation:** Three columns were strictly pruned before cross-validation splitting:
  - `Churn Reason`: Post-churn exit interview feedback collected only after a customer leaves.
  - `Churn Score`: Output of IBM's pre-computed internal model.
  - `CLTV`: Post-hoc metric calculated from churn formula.
- **Pipeline Architecture:** Scikit-learn `ColumnTransformer` with `StandardScaler` for continuous features (`Tenure Months`, `Monthly Charges`, `Total Charges`) and `OneHotEncoder(handle_unknown='ignore')` for categorical attributes.

---

## 3. Machine Learning Engineering & Model Selection

### 5-Fold Stratified Cross-Validation Benchmark
| Estimator Algorithm | CV Mean ROC-AUC | Out-of-Fold Generalization | Selection Verdict |
| :--- | :--- | :--- | :--- |
| **Dummy (Majority Class)** | $0.5000$ | Baseline zero skill | Baseline |
| **Logistic Regression (L2)** | $0.8412$ | Fast, linear baseline | Benchmark |
| **Decision Tree (CART)** | $0.6580$ | Severe leaf overfitting | Rejected |
| **Random Forest (100 Trees)** | $0.8450$ | Robust ensemble | Candidate |
| **XGBoost Classifier** | $0.8520$ | Strong gradient booster | Strong Candidate |
| **HistGradientBoosting** | **$0.8639$** | Fast native binning, superior log-loss | **Selected Production Model** |

### Holdout Test Set Performance (1,409 Records)
- **ROC-AUC:** **`0.8570`**
- **Precision-Recall AUC:** **`0.6740`**
- **Brier Calibration Score:** **`0.1322`** (near-optimal alignment along $y=x$)
- **Operating Threshold ($\tau = 0.30$):**
  - Recall (Sensitivity): **$77.01\%$** ($288$ of $374$ true churners captured)
  - Precision: **$54.24\%$**
  - F1-Score: **$0.6365$**
  - Missed Churners: Cut from $177$ (at 0.50) down to $86$ (a **$51.4\%$ reduction in churn leakage**).

---

## 4. Advanced Engineering & Production Architecture

1. **Stateless REST Serving:** FastAPI backend with lifespan-managed model singleton, sub-20ms median latency, and automatic fallback engine in the Streamlit UI.
2. **Experiment Tracking & Governance:** MLflow with embedded SQLite backend (`sqlite:///mlflow.db`) and local Model Registry (`models/registry.json`) enforcing human approval gates before production promotion.
3. **Application Security:** In-memory sliding-window rate limiter (180 req/min), 2,500 record batch ceiling, 10 MB payload limit, strict PII redaction, and standard HTTP security headers.
4. **Performance Engineering:** Empirical benchmarks confirmed $18.26$ ms p50 latency and $5,104$ records/second throughput on multi-record batch scoring.
5. **System Reliability:** Health monitoring distinguishing liveness (`/health`) from readiness (`/ready`), accompanied by 1-click model rollback procedures.
6. **Customer Personas & ROI Modeling:** Unsupervised K-Means clustering ($k=4$, Silhouette Score = $0.4768$) coupled with interactive financial ROI simulations in the UI.

---

## 5. Technical Difficulties, Trade-offs & Lessons Learned

| Challenge Encountered | Engineering Trade-off | Resolution Implemented |
| :--- | :--- | :--- |
| **MLflow 3.x Deprecations** | File-based stores (`mlruns/`) deprecated in modern MLflow releases. | Adopted embedded SQLite database (`sqlite:///mlflow.db`), providing ACID transactions without requiring an external server. |
| **Target Leakage Risks** | Retaining post-churn attributes inflates validation scores spuriously. | Implemented automated Pandera schema assertions and configuration-level exclusion lists. |
| **Threshold Asymmetry** | Default $\tau = 0.50$ yields high precision but leaves half of all churners undetected. | Calibrated threshold to $\tau = 0.30$ on cross-validation, capturing $81$ additional churners with verified positive campaign ROI. |
| **Headless Script Threading** | Matplotlib GUI backend crashes when run in non-interactive terminal tasks. | Enforced non-interactive Agg backend (`matplotlib.use('Agg')`) globally across all plotting modules. |
| **PII Protection vs Diagnostics**| Tracking individual accounts without violating privacy boundaries. | Masked IDs to prefix format (`7590-****`) and logged only aggregate distribution statistics. |
