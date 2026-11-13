# Phase 13: Model Monitoring, Data Drift, and Maintenance Strategy

## 1. Production Monitoring Architecture

Machine learning models deployed in dynamic business environments inevitably face performance degradation due to changes in consumer behavior, macroeconomic shifts, or competitor promotions.

Our monitoring framework tracks three distinct tiers:
1. **System & Operational Health:** API uptime, throughput (requests/sec), latency percentiles (p50, p95, p99), HTTP 4xx/5xx error rates.
2. **Data & Covariate Drift:** Statistical evaluation of production feature distributions against baseline training distributions.
3. **Prediction & Ground-Truth Performance Drift:** Tracking the proportion of flagged accounts, average predicted probabilities, and realized classification metrics once true ground-truth customer retention labels become available (typically after 30–60 days).

---

## 2. Statistical Data Drift Methodology

To detect feature distribution drift without requiring true labels, we combine two complementary statistical metrics:

### A. Population Stability Index (PSI)
Used across both numerical and categorical features:
$$\text{PSI} = \sum_{i=1}^{k} \left( \text{Actual}_i - \text{Expected}_i \right) \times \ln\left( \frac{\text{Actual}_i}{\text{Expected}_i} \right)$$

- **$\text{PSI} < 0.10$ (Green / Stable):** No significant distribution change. Standard model serving continues.
- **$0.10 \le \text{PSI} < 0.25$ (Yellow / Moderate Drift):** Warning threshold. Increase monitoring frequency; analyze sub-populations.
- **$\text{PSI} \ge 0.25$ (Red / Significant Drift):** Critical distribution shift. Triggers an automated alert for pipeline retraining.

### B. Kolmogorov-Smirnov (KS) Test
Employed on continuous numerical variables (`Tenure Months`, `Monthly Charges`, `Total Charges`) as a non-parametric test comparing cumulative empirical distribution functions ($H_0$: the two samples are drawn from the same underlying distribution; reject if $p < 0.05$).

---

## 3. Verified Drift Audit Results (Baseline vs Test Sample)

| Feature | Feature Type | PSI Score | KS Statistic | KS $p$-value | Drift Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Tenure Months`** | Numerical | **0.0067** | 0.0270 | 0.6835 | **Stable** |
| **`Monthly Charges`** | Numerical | **0.0124** | 0.0390 | 0.1912 | **Stable** |
| **`Total Charges`** | Numerical | **0.0076** | 0.0360 | 0.2710 | **Stable** |
| **`Contract`** | Categorical | **0.0000** | — | — | **Stable** |
| **`Internet Service`** | Categorical | **0.0000** | — | — | **Stable** |
| **`Payment Method`** | Categorical | **0.0001** | — | — | **Stable** |

*Audit Conclusion:* All feature distributions in the test set remain within standard stability bounds ($\text{PSI} \ll 0.10$).

---

## 4. Retraining Cadence & Champion-Challenger Workflow

```
                        Retraining Pipeline
                        
[ New Billing / CRM Data ] ──> [ Data Quality & Schema Gate ]
                                           │
                                           ▼
[ Candidate Challenger Model ] <── [ Stratified Retraining ]
            │
            ▼
[ Offline Evaluation Gate ] (Requires ROC-AUC >= Champion & PR-AUC >= Champion)
            │
            ├── Pass ──> [ Shadow Deployment (10% Traffic) ] ──> [ Promote to Champion ]
            └── Fail ──> [ Reject & Alert ML Engineer ]
```

### Retraining Triggers
1. **Scheduled Cadence:** Quarterly retraining incorporating the most recent 90-day cohort.
2. **Drift-Triggered:** Any primary predictor exhibiting $\text{PSI} \ge 0.25$ for more than 14 consecutive days.
3. **Performance-Triggered:** Realized Recall dropping below 70.0% or PR-AUC dropping below 0.60 on validated ground truth.

### Promotion Criteria
A challenger model will replace the active champion **only** if:
- It passes all automated schema and boundary unit tests.
- Cross-validation ROC-AUC improves or matches the champion without increasing variance across folds.
- Recall at the calibrated threshold meets or exceeds 75.0%.
- Latency benchmarks remain under 25 milliseconds per record.
