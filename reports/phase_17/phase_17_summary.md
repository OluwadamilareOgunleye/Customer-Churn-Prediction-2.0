# Phase 17 Summary: Advanced Model Validation, Calibration, and Robustness

## Summary of Findings
1. **Calibration:**
   - Evaluated using 10-bin calibration curves and Brier score.
   - Raw HistGradientBoosting Brier score: `0.1322`.
   - Sigmoid Platt scaling via 5-fold CV Brier score: `0.1321` (minimal 0.08% improvement).
   - Finding: The raw tree booster is already well-calibrated; post-hoc recalibration is unnecessary.
2. **Threshold Tradeoff:**
   - Tested thresholds from 0.10 to 0.90.
   - Confirmed operational threshold $\tau = 0.30$ provides the ideal balance between Recall ($77.01\%$) and Precision ($54.24\%$) with maximum F1 ($0.6365$).
   - Captures 288 out of 374 true churners in the holdout test set.
3. **Subgroup Disaggregation:**
   - High sensitivity verified for core high-churn customer groups:
     - Month-to-month contracts: Recall = $86.02\%$
     - First-year tenure (0-12 mos): Recall = $86.05\%$
     - Electronic check: Recall = $88.83\%$
     - Fiber optic internet: Recall = $86.51\%$
4. **Stress & Sensitivity Testing:**
   - Missing values handled via pipeline imputers without runtime failure.
   - Numerical extreme stress testing completed and saved to `reports/phase_17/numerical_extreme_stress_test.csv`.
   - Sensitivity curves generated and plotted for tenure and monthly charges in `reports/figures/20_feature_sensitivity_curves.png`.
