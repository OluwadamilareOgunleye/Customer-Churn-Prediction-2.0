# Phase 1: Problem Definition and Business Understanding

## 1. Executive Summary & Business Context

In the telecommunications industry, customer churn—the rate at which existing subscribers discontinue their service subscriptions—represents one of the most critical operational and financial challenges. Telecommunications providers operate in mature, highly competitive, and largely saturated markets where market expansion through new customer acquisition is significantly more expensive than retaining existing customers (industry research routinely indicates acquisition costs are 5 to 7 times higher than retention costs).

### Financial Impact of Churn
- **Recurring Revenue Erosion:** Telecom business models rely on predictable, high-margin monthly recurring revenue (MRR) and annual contract commitments. Losing a customer results not only in an immediate loss of monthly billings but also permanently diminishes Customer Lifetime Value (CLV).
- **Customer Acquisition Cost (CAC) Amortization:** Telecom companies often heavily subsidize initial hardware, installation, and onboarding costs. When customers churn early in their lifecycle (low tenure), the company fails to amortize these upfront acquisition costs, resulting in a net negative return on investment for that customer account.
- **Retention Cost Efficiency:** Proactive retention programs (e.g., tailored plan upgrades, proactive technical support, targeted discounts) are substantially cheaper to execute than winning back lapsed subscribers or hunting for net new customers in saturated demographics.

### Strategic Value of Machine Learning in Churn Management
Without predictive modeling, retention teams typically act **reactively**—intervening only after a customer calls customer care to cancel, which is often too late, or offering blanket discounts across broad customer segments, which wastes margin on customers who were never at risk of leaving. 

A predictive machine learning system allows telecom operators to:
1. Identify high-risk customers before they defect.
2. Uncover the underlying root causes and risk factors driving dissatisfaction (e.g., specific contract structures, high monthly charges, technical support issues).
3. Enable targeted, personalized, and cost-effective retention interventions.

---

## 2. Problem Statement

Develop a machine learning system capable of predicting whether an active telecommunications customer will churn based on historical demographic profile data, account information, service subscription configurations, and billing details.

The goal is to provide reliable churn risk probabilities for individual customer accounts to enable customer retention specialists and automated marketing systems to prioritize and personalize intervention campaigns.

---

## 3. Machine Learning Formulation

| Dimension | Specification | Notes / Rationale |
| :--- | :--- | :--- |
| **Learning Paradigm** | Supervised Learning | Ground-truth labels indicating customer churn status are available in historical records. |
| **Task Type** | Binary Classification | Each customer observation is classified into one of two mutually exclusive outcomes. |
| **Target Variable** | `Churn` (or `Churn Value` / `Churn Label`) | Binary indicator of customer departure. |
| **Positive Class ($y = 1$)** | `"Yes"` (or `1`) | Customer discontinued their service (Churned). |
| **Negative Class ($y = 0$)** | `"No"` (or `0`) | Customer retained their service (Active / Retained). |
| **Primary Model Output** | Calibrated Probability $\hat{p} \in [0, 1]$ | The probability that $y = 1$, allowing customizable decision thresholds based on campaign economics. |

*(Note: Target column naming and encoding will be formally verified against the dataset in Phase 2).*

---

## 4. Key Business Questions & Research Hypotheses

This project systematically investigates the following analytical and business questions without presuming predetermined outcomes:

1. **Customer Tenacity & Tenure Dynamics:**
   - *Hypothesis:* Newer customers exhibit higher churn vulnerability compared to long-standing customers who have established usage habits and switching friction.
   - *Investigation:* How does churn probability vary across tenure tiers? Is there an inflection point where customer retention stabilizes?

2. **Contract & Commitment Structure:**
   - *Hypothesis:* Month-to-month contracts lack long-term commitment mechanisms and show significantly higher churn rates than one-year or two-year contracts.
   - *Investigation:* What proportion of churners are on month-to-month contracts versus long-term commitments?

3. **Pricing & Billing Sensitivity:**
   - *Hypothesis:* Higher monthly charges without perceived value-add drive customer dissatisfaction and defections.
   - *Investigation:* How do monthly charges correlate with churn across service tiers? Is paperless billing or electronic check payment associated with higher churn?

4. **Service Bundle & Ecosystem Stickiness:**
   - *Hypothesis:* Customers subscribed to bundled services (e.g., Online Security, Tech Support, Streaming, Backup) experience higher switching costs and churn less.
   - *Investigation:* Which individual services or bundle combinations are most strongly associated with high customer retention?

5. **Demographic & Household Factors:**
   - *Hypothesis:* Senior citizens or customers without dependents/partners have distinct churn patterns requiring tailored retention approaches.
   - *Investigation:* Are demographic attributes predictive on their own, or are their effects mediated by contract and service choices?

6. **Model Identification & Operational Utility:**
   - *Hypothesis:* Non-linear gradient boosting models (e.g., XGBoost, LightGBM) or ensemble methods will outperform linear baselines due to complex feature interactions (e.g., tenure $\times$ monthly charges $\times$ contract type).

---

## 5. Prediction Framing & Realistic Operational Constraints

### Historical Snapshot Limitations
The dataset represents a single cross-sectional observational snapshot of customer accounts at a point in time. It is crucial to recognize the inherent operational constraints of this data structure:
- **No Explicit Time Horizon:** The dataset does not track longitudinal event timestamps or survival time-to-event windows (e.g., churn within 30 days vs 90 days). Churn is observed as a static status. Therefore, the model predicts the likelihood of being a churned customer based on snapshot attributes, not time-to-event survival hazard rates.
- **Lookahead Bias Prevention:** In real-world deployment, all input features fed into the model must strictly represent information known **prior** to the customer's decision to churn. Attributes generated post-churn (such as exit surveys or churn reason codes) cannot be used as predictive inputs.
- **Data Availability at Scoring Time:** Only features routinely tracked in CRM, billing, and provisioning databases (account age, subscribed products, billing method, payment status) should be included in the inference feature set.

---

## 6. Business Success Criteria & Evaluation Strategy

Evaluating a churn model requires aligning technical classification metrics with business trade-offs. Relying solely on **Accuracy** is flawed because customer churn datasets are typically imbalanced (e.g., 70–80% non-churners vs 20–30% churners); a trivial model predicting "No Churn" for every customer could achieve 75%+ accuracy while identifying zero at-risk customers.

### Metric Hierarchy & Trade-off Analysis

```
                              Confusion Matrix
                       Actual Churn (1)       Actual Retained (0)
Predicted Churn (1)    True Positive (TP)     False Positive (FP)
Predicted Retained (0) False Negative (FN)    True Negative (TN)
```

1. **Recall (Sensitivity / True Positive Rate = $\frac{TP}{TP + FN}$):**
   - **Business Meaning:** What proportion of all actual churners did the system successfully catch?
   - **Cost of False Negatives (FN):** High. A false negative means an at-risk customer went undetected, received no retention effort, and churned. The cost is the lost Customer Lifetime Value (CLV) and subsequent acquisition replacement cost.

2. **Precision ($\frac{TP}{TP + FP}$):**
   - **Business Meaning:** When the model alerts that a customer is at risk, how often is it right?
   - **Cost of False Positives (FP):** Moderate. A false positive triggers an unnecessary retention intervention (e.g., discount, customer service call). This expends operational resources and risks unnecessarily reducing profit margins for customers who intended to stay anyway.

3. **$F_1$-Score & $F_\beta$-Score:**
   - Harmonic balance between precision and recall. Where missing a churner is considered more costly than sending an unnecessary retention offer, an $F_\beta$ score (such as $F_2$, weighting recall twice as heavily as precision) serves as a robust composite metric.

4. **ROC-AUC & PR-AUC:**
   - **ROC-AUC:** Measures the model's overall discriminative ability across all possible classification thresholds.
   - **PR-AUC (Precision-Recall Area Under Curve):** Particularly vital in imbalanced datasets, as it directly reflects precision and recall trade-offs across minority class predictions without being inflated by a large volume of true negatives.

5. **Decision Threshold Tuning:**
   - The default classification threshold of 0.5 is rarely optimal in churn management. By adjusting the probability threshold based on the specific cost-benefit matrix of the retention campaign (Intervention Cost vs. Churn Value Saved), the business can maximize net ROI.

---

## 7. Project Scope & Roadmap

### In Scope (Phases 1–4)
- Comprehensive problem definition and business impact modeling.
- Raw dataset acquisition, structural auditing, and data dictionary compilation.
- Rigorous exploratory data analysis (EDA): univariate, bivariate, and multivariate investigations.
- Identification and treatment of data quality anomalies (missing values, whitespace, data types).
- Professional, leakage-free preprocessing pipeline design using Scikit-Learn (`ColumnTransformer`, `Pipeline`).
- Unit testing and validation test suites for data integrity.

### Future Scope (Phases 5+)
- Baseline and advanced model training (Logistic Regression, Decision Trees, Random Forest, XGBoost).
- Class imbalance remediation strategies (class weighting, SMOTE, PR-curve threshold optimization).
- Systematic hyperparameter tuning (GridSearchCV, RandomizedSearchCV, or Optuna).
- Model explainability via SHAP (SHapley Additive exPlanations) for global and local feature attribution.
- Production deployment (Interactive Streamlit dashboard, modular scoring script, containerization with Docker).
