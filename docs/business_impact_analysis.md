# Business Impact & Retention Campaign Financial ROI Framework

## Executive Summary

Machine learning classification metrics (such as ROC-AUC, Precision, and Recall) must translate directly into corporate profit and loss impact. This financial evaluation models the economic value of deploying the churn prediction model at the operational threshold ($\tau = 0.30$) compared against the standard default threshold ($\tau = 0.50$).

---

## 1. Simulation Parameters & Explicit Assumptions

Because the Telco dataset does not contain historical retention marketing costs or post-intervention retention rates, the financial model applies explicitly labeled baseline assumptions:

- **Customer Lifetime Value (CLV):** **\$1,200.00** (average annual revenue per subscriber).
- **Retention Campaign Cost per Targeted Account:** **\$50.00** (cost of direct outreach, temporary discounts, or promotional perks).
- **Intervention Success Rate:** **25%** (1 in 4 targeted churners accepts the offer and remains retained).
- **Customer Acquisition Cost (CAC):** **\$400.00** (cost to acquire a replacement subscriber).

---

## 2. Empirical Holdout Test Set Performance Comparison

Evaluated on the held-out test dataset ($N = 1,409$ total customers, containing $374$ true churners):

| Metric | Operational Threshold ($\tau = 0.30$) | Default Threshold ($\tau = 0.50$) | Business Variance ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Total Customers Targeted** | 531 accounts | 293 accounts | $+238$ accounts |
| **True Churners Targeted ($TP$)** | **288 churners** | 197 churners | **$+91$ churners captured** |
| **False Alarms ($FP$)** | 243 accounts | 96 accounts | $+147$ offers sent |
| **Missed Churners ($FN$)** | **86 churners** | 177 churners | **$-91$ churners missed (51.4% reduction)** |
| **Saved Churners ($TP \times 25\%$)** | **72 subscribers** | 49 subscribers | **$+23$ subscribers saved** |
| **Campaign Spend ($Targeted \times \$50$)**| \$26,550.00 | \$14,650.00 | $+\$11,900.00$ spend |
| **Gross Revenue Saved ($Saved \times \$1,200$)**| **\$86,400.00** | \$58,800.00 | **+\$27,600.00 revenue saved** |
| **Net Financial Profit** | **\$59,850.00** | \$44,150.00 | **+\$15,700.00 Net Profit** |
| **Net Campaign ROI** | **225.4%** | 301.4% | Strategic tradeoff |
| **Unmitigated Churn Loss ($FN \times \$1,200$)**| **\$103,200.00** | \$212,400.00 | **-\$109,200.00 loss averted** |

*(Artifact: `reports/figures/22_business_retention_roi_comparison.png`)*

---

## 3. Key Strategic Takeaways

1. **Maximizing Bottom-Line Profit:** While the default $\tau = 0.50$ threshold yields a slightly higher percentage ROI ($301.4\%$ vs $225.4\%$) because it sends fewer offers, it results in **\$15,700 less net absolute dollar profit** (\$44,150 vs \$59,850). In enterprise operations, dollar profits outweigh percentage efficiency.
2. **Drastic Reduction in Churn Leakage:** Setting $\tau = 0.30$ cuts unmitigated churn losses from **\$212,400 down to \$103,200**, averting **\$109,200 in churn damage**.
3. **Interactive Control:** The Streamlit application (`frontend/app.py`) provides an interactive ROI slider allowing executive leadership to test sensitivity against dynamic CLV and intervention cost assumptions.
