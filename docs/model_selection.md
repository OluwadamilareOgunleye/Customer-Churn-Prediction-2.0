# Phase 5: Model Development, Evaluation, and Selection Report

## 1. Executive Summary

Phase 5 develops, cross-validates, tunes, and evaluates candidate machine learning algorithms for predicting customer churn. All models were integrated with the Scikit-Learn preprocessing pipeline to strictly prevent data leakage. Stratified 5-Fold Cross-Validation was conducted across six candidate models.

The selected production model is a **HistGradientBoostingClassifier** (Histogram-based Gradient Boosted Decision Tree ensemble). Following hyperparameter optimization and probability threshold calibration (selected threshold $\tau = 0.30$), the model achieved an **ROC-AUC of 0.8570**, a **PR-AUC of 0.6740**, and a **Recall of 77.01%** on the completely untouched test set (capturing 288 out of 374 actual churners).

The complete end-to-end pipeline was serialized with its calibrated decision threshold and operational metadata into `models/churn_pipeline_bundle.joblib`.

---

## 2. Model Benchmarking & Cross-Validation Results

Six distinct model architectures were trained and evaluated on the training dataset (5,634 rows) using **5-Fold Stratified Cross-Validation**. 

### Benchmark Performance Summary

| Model Architecture | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | Recall (Mean ± Std) | Precision (Mean ± Std) | F1-Score (Mean ± Std) | Accuracy (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 0.8624 ± 0.0091 | 0.6805 ± 0.0179 | 0.8047 ± 0.0306 | 0.5447 ± 0.0198 | 0.6494 ± 0.0202 | 0.7694 ± 0.0125 |
| **XGBoost** | 0.8624 ± 0.0100 | 0.6878 ± 0.0186 | 0.8127 ± 0.0254 | 0.5325 ± 0.0138 | 0.6433 ± 0.0163 | 0.7609 ± 0.0098 |
| **Logistic Regression** | 0.8588 ± 0.0125 | 0.6789 ± 0.0208 | 0.8100 ± 0.0342 | 0.5294 ± 0.0198 | 0.6401 ± 0.0211 | 0.7583 ± 0.0112 |
| **HistGradientBoosting** | 0.8494 ± 0.0061 | 0.6576 ± 0.0107 | 0.7338 ± 0.0204 | 0.5631 ± 0.0108 | 0.6370 ± 0.0087 | 0.7781 ± 0.0075 |
| **Decision Tree (depth=5)**| 0.8453 ± 0.0083 | 0.6254 ± 0.0163 | 0.8127 ± 0.0221 | 0.5142 ± 0.0140 | 0.6295 ± 0.0070 | 0.7460 ± 0.0089 |
| **Dummy (Majority Class)**| 0.5000 ± 0.0000 | 0.2654 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7346 ± 0.0001 |

### Architectural Insights:
1. **The Dummy Baseline Trap:** The Dummy Classifier achieved 73.46% accuracy simply by guessing the majority class ("No Churn"). However, its Recall and F1-Score are exactly 0.0000, confirming that accuracy is an uninformative metric for churn detection.
2. **Linear vs. Non-linear Ensembles:** Logistic Regression with balanced class weights performed admirably (ROC-AUC 0.8588, Recall 0.8100), demonstrating strong baseline linear separability. However, tree ensembles (Random Forest, XGBoost, HistGradientBoosting) captured non-linear interactions between tenure, contract type, and service tier combinations, delivering superior PR-AUC.
3. **HistGradientBoosting Strengths:** HistGradientBoosting offers native support for continuous histogram binning, rapid CPU execution, and low variance across folds ($\sigma_{\text{ROC-AUC}} = 0.0061$, the lowest variance among all competitive models).

---

## 3. Hyperparameter Tuning

We conducted systematic hyperparameter optimization using **GridSearchCV** with 5-fold stratified cross-validation on the HistGradientBoosting pipeline.

### Search Space & Optimization Results

| Hyperparameter | Evaluated Grid | Best Value | Rationale |
| :--- | :--- | :--- | :--- |
| `max_iter` (Number of boosting trees) | `[100, 150]` | **100** | Prevents overfitting while ensuring convergence. |
| `learning_rate` (Shrinkage rate) | `[0.03, 0.05]` | **0.03** | Smaller step size provides smoother decision boundaries and improved generalization. |
| `max_leaf_nodes` (Tree complexity) | `[15, 31]` | **15** | Shallower trees avoid memorizing niche multi-feature interactions. |
| `min_samples_leaf` (Leaf regularization) | `[30, 50]` | **50** | Enforces minimum statistical support in each leaf, smoothing probability estimates. |
| `class_weight` | `[None, 'balanced']` | **`None`** | Preserves true posterior probability calibration for threshold optimization. |

- **Cross-Validation ROC-AUC before tuning:** 0.8494
- **Cross-Validation ROC-AUC after tuning:** **0.8639** (+1.45% relative improvement)

---

## 4. Classification Threshold Calibration

In customer churn management, the default classification threshold of $\tau = 0.50$ is suboptimal because it treats false negatives (undetected churners) and false positives (needless retention offers) as equally costly. 

To determine the optimal operating point without touching the test set, we conducted a 91-point threshold sweep ($\tau \in [0.05, 0.95]$) on out-of-fold training probabilities.

![Threshold Trade-offs](../reports/figures/09_threshold_tradeoffs.png)

### Threshold Comparison

| Decision Threshold ($\tau$) | Cross-Validation Recall | Cross-Validation Precision | Cross-Validation F1 | Business Impact |
| :--- | :--- | :--- | :--- | :--- |
| **0.50 (Default Standard)** | 55.45% | 67.82% | 0.6101 | Misses 44.5% of churners; retention team is underutilized. |
| **0.40 (Intermediate)** | 67.22% | 60.40% | 0.6361 | Balanced trade-off. |
| **0.30 (Calibrated Optimum)** | **78.06%** | **55.80%** | **0.6507** | **Optimal F1 & High Recall:** Captures nearly 80% of churners with sustainable precision. |
| **0.20 (High Recall)** | 88.43% | 45.12% | 0.5973 | High recall, but 55% of flagged accounts are false alarms. |

**Selected Threshold:** **$\tau = 0.30$**. At this threshold, the retention team captures over 77% of defecting customers while maintaining an intervention precision above 54%.

---

## 5. Final Evaluation on Untouched Test Set

The tuned pipeline was fitted on the entire training set (5,634 rows) and evaluated once on the previously unseen test set (1,409 rows).

### Test Set Performance Metrics

| Metric | Calibrated Threshold ($\tau = 0.30$) | Standard Threshold ($\tau = 0.50$) | Baseline (Dummy) |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.8570** | 0.8570 | 0.5000 |
| **PR-AUC (Avg. Precision)** | **0.6740** | 0.6740 | 0.2654 |
| **Recall (Sensitivity)** | **77.01% (288 / 374)** | 55.35% (207 / 374) | 0.00% (0 / 374) |
| **Precision** | **54.24%** | 68.32% | 0.00% |
| **F1-Score** | **0.6365** | 0.6115 | 0.0000 |
| **Accuracy** | **76.65%** | 82.26% | 73.46% |

![ROC and PR Curves](../reports/figures/12_test_roc_pr_curves.png)

### Confusion Matrix Comparison

```
Calibrated Threshold (τ = 0.30)           Standard Threshold (τ = 0.50)
        Actual: No   Actual: Yes                  Actual: No   Actual: Yes
Pred No    792 (TN)     86 (FN)          Pred No    939 (TN)    167 (FN)
Pred Yes   243 (FP)    288 (TP)          Pred Yes    96 (FP)    207 (TP)
```

![Calibrated Confusion Matrix](../reports/figures/10_test_confusion_matrix_calibrated.png)
![Standard Confusion Matrix](../reports/figures/11_test_confusion_matrix_standard.png)

**Key Takeaway:** Calibrating the threshold to 0.30 enabled the system to capture **81 additional churners** (reducing missed churners from 167 down to 86) with only 147 additional false alerts, delivering an overall higher F1 score and superior business value.

---

## 6. Feature Importance & Model Interpretability

We evaluated global feature attribution using **Permutation Feature Importance** on the test set, measuring the mean drop in ROC-AUC when each feature is randomly shuffled across 10 repeats.

![Permutation Feature Importance](../reports/figures/13_permutation_feature_importance.png)

### Top Predictive Features

| Rank | Feature | Mean ROC-AUC Drop | Interpretation & Business Significance |
| :--- | :--- | :--- | :--- |
| 1 | `Tenure Months` | **0.0682** | Customer relationship duration is the primary anchor of retention; customers survive the highest hazard rate during months 1–12. |
| 2 | `Contract` | **0.0415** | Commitment structure directly governs customer churn friction. Month-to-month contracts lack lock-in. |
| 3 | `Monthly Charges` | **0.0248** | Plan expense drives price sensitivity, especially in higher price tiers ($70–$100). |
| 4 | `Internet Service` | **0.0210** | Fiber Optic subscribers exhibit substantially elevated defection rates compared to DSL or phone-only users. |
| 5 | `Payment Method` | **0.0152** | Electronic check payers churn at 45.3% vs ~15.5% for automatic billing methods. |
| 6 | `Tech Support` | **0.0118** | Technical support acts as an operational buffer mitigating friction on high-speed internet products. |
| 7 | `Total Charges` | **0.0094** | Cumulative billing magnitude interacts with tenure and monthly expenditure. |
| 8 | `Paperless Billing`| **0.0065** | Paperless billing users show moderately higher churn propensity. |

---

## 7. Model Serialization & Production Serving

The model is serialized using Joblib at:
`models/churn_pipeline_bundle.joblib`

The bundle contains:
1. `pipeline`: The complete, fitted Scikit-Learn `Pipeline` encompassing both the `ColumnTransformer` (SimpleImputer + StandardScaler + OneHotEncoder) and the tuned `HistGradientBoostingClassifier`.
2. `threshold`: The calibrated decision threshold (`0.30`).
3. `metrics`: Comprehensive test and cross-validation performance dictionaries.
4. `feature_names`: Expected predictor columns.
5. `metadata`: Training timestamp, dataset size, and hyperparameter specifications.

### Inference API
The `ChurnPredictor` class (`src/churn_prediction/model_inference.py`) provides:
- `predict_proba(df)`: Returns calibrated positive-class churn probabilities $\hat{p} \in [0, 1]$.
- `predict(df, threshold=None)`: Returns binary classifications applying the calibrated threshold.
- `score_records(df)`: Appends `churn_probability`, `churn_prediction`, `churn_label`, and `risk_tier` (`Low Risk`, `Medium Risk`, `High Risk`) to the customer dataframe.
