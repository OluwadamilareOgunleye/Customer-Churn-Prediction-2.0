# Customer Segmentation & Persona Analysis

## Overview

While supervised binary classification predicts individual customer churn propensity, **unsupervised customer segmentation** reveals behavioral clusters and spend patterns across the subscriber base. Combining segmentation with churn risk stratification enables marketing and retention teams to tailor interventions to distinct subscriber cohorts.

---

## 1. Methodology & Model Architecture

- **Algorithm:** K-Means Clustering (`KMeans(n_clusters=4, random_state=42, n_init=10)`).
- **Feature Space:** Numerical attributes capturing the subscriber financial footprint:
  - `Tenure Months`: Length of customer relationship.
  - `Monthly Charges`: Recurring subscription revenue.
  - `Total Charges`: Cumulative lifetime spend.
- **Preprocessing:** Median imputation for missing total charges and `StandardScaler` normalization.
- **Clustering Quality:**
  - **Silhouette Score:** **0.4768** (indicating dense, well-separated clusters without arbitrary forcing).

---

## 2. Identified Customer Personas

| Persona Cohort | Cluster Size ($N$) | Avg Tenure | Avg Monthly Bill | Actual Churn Rate | Model Churn Rate ($\tau=0.30$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **New Budget Subscribers** | 428 | $13.1$ mos | $\$33.80$ | $23.1\%$ | $24.8\%$ |
| **New High-Spend / Tech Subscribers** | 443 | $14.2$ mos | $\$82.40$ | **$51.5\%$** | **$68.2\%$** |
| **Tenured Premium Accounts** | 352 | $57.8$ mos | $\$98.60$ | $14.2\%$ | $16.5\%$ |
| **Long-Term Value Loyalists** | 186 | $54.2$ mos | $\$24.10$ | $4.3\%$ | $3.2\%$ |

*(Artifact: `reports/figures/21_customer_segmentation_clusters.png`)*

---

## 3. Churn-Risk Stratification Tiers

Customer accounts are categorized into four operational risk tiers based on calibrated churn probability:

1. **Low Risk ($p < 0.15$):** $45.6\%$ of test subscribers ($N=642$). Loyal or long-tenured accounts; no retention intervention required.
2. **Moderate Risk ($0.15 \le p < 0.30$):** $16.8\%$ of test subscribers ($N=236$). Standard service monitoring and automated engagement.
3. **Elevated Risk ($0.30 \le p < 0.60$):** $25.1\%$ of test subscribers ($N=353$). Target candidates for proactive contract incentives and customer success check-ins.
4. **High Risk ($p \ge 0.60$):** $12.6\%$ of test subscribers ($N=178$). Immediate intervention required: senior retention specialist outreach, fee waivers, or discounted hardware upgrades.
