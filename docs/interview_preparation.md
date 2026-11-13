# Technical Interview Preparation Guide: Customer Churn Prediction & MLOps

This guide provides technical questions and answers based on the implementation of this project.

---

## 1. Machine Learning & Modeling Decisions

### Q1: Why formulate customer churn as a binary classification problem rather than survival analysis or regression?
**Answer:** Binary classification directly answers the operational question: *"Is this subscriber at elevated risk of terminating their contract during the upcoming billing cycle?"* This matches marketing workflow schedules (e.g. monthly campaigns). While survival analysis is well suited for time-to-event estimation, gradient boosted binary classification directly optimizes ranking metrics (ROC-AUC and PR-AUC) and outputs calibrated probabilities that feed directly into cost-benefit threshold optimization.

### Q2: Why did you choose HistGradientBoosting over XGBoost or Random Forest?
**Answer:** In 5-fold cross-validation benchmarking:
- Decision Tree overfitted drastically (train AUC ~0.98 vs validation ~0.65).
- Random Forest achieved 0.8450 AUC but had large serialized artifact sizes and slower CPU inference.
- XGBoost achieved 0.8520 AUC.
- **HistGradientBoosting** achieved the highest cross-validation ROC-AUC (**0.8639**) and test ROC-AUC (**0.8570**). Its histogram-based integer binning provides 10x faster training, native handling of missing values, sub-20ms inference latency, and zero external C++ runtime dependencies.

### Q3: What is data leakage, and where was it present in this dataset?
**Answer:** Data leakage occurs when information unavailable at inference time is included in training features, producing unrealistically optimistic validation metrics that fail in production. In this dataset, three columns were identified as severe leakage:
1. `Churn Reason`: Qualitative customer exit interview notes collected only *after* churn occurred.
2. `Churn Score`: Output of IBM's pre-computed churn model.
3. `CLTV`: Calculated using post-hoc tenure and churn status.
All three were strictly excluded before training.

### Q4: Why calibrate the decision threshold to 0.30 instead of using the standard 0.50?
**Answer:** Standard classification assumes symmetric misclassification costs ($\text{Cost}(FN) = \text{Cost}(FP)$). In customer retention, the costs are asymmetric:
- $\text{Cost}(FN)$: Losing a customer worth $\$1,200$ CLV to churn.
- $\text{Cost}(FP)$: Sending a $\$50$ promotional perk or outreach to a customer who wasn't going to leave.
At $\tau = 0.50$, the model missed $177$ churners (Recall: $52.67\%$). At $\tau = 0.30$, Recall rises to **$77.01\%$**, capturing $91$ additional churners and cutting missed churners by **$51.4\%$**. Financial simulation confirms this produces **$+\$15,700$ higher net profit** than $\tau = 0.50$.

---

## 2. Software Architecture & Serving

### Q5: How is the API serving architecture designed?
**Answer:** The serving architecture is a stateless microservice built on FastAPI:
- The trained Scikit-learn Pipeline bundle is loaded once into container memory at startup via Starlette's `lifespan` handler.
- Incoming payloads are validated against strict Pydantic v2 schemas.
- In-memory rate limiting (180 req/min) and batch caps (2,500 records) prevent DoS attacks.
- Standard HTTP security headers (`nosniff`, `DENY`, `HSTS`) are injected on every response.
- Median inference latency is **$18.26$ ms**, with multi-record batch throughput exceeding **$5,100$ records / second**.

### Q6: How does the system distinguish between Liveness and Readiness?
**Answer:**
- **Liveness (`/health`):** Verifies the process is up and the event loop is responsive. Orchestrators use this to detect deadlocks.
- **Readiness (`/ready`):** Deep check verifying that the model bundle is deserialized in RAM, the calibrated threshold ($\tau = 0.30$) is active, and the host has at least $50$ MB of available RAM headroom. If unready, it returns `HTTP 503 Service Unavailable`, preventing traffic from hitting an uninitialized pod.

---

## 3. MLOps, Governance & Reliability

### Q7: How do you handle experiment tracking and model governance?
**Answer:**
- **Experiment Tracking:** Uses **MLflow** with an embedded SQLite backend (`sqlite:///mlflow.db`) tracking dataset hashes, hyperparameters, 5-fold CV metrics, holdout test metrics, and serialized artifacts.
- **Model Registry:** A JSON-backed registry (`models/registry.json`) manages versions across four lifecycle states: `Candidate` $\rightarrow$ `Staging` $\rightarrow$ `Production` $\rightarrow$ `Archived`.
- **Governance Rule:** A newly trained candidate **never replaces production automatically**. When a verified model is promoted to `Production`, the previous version is demoted to `Archived`, enabling 1-click rollback.

### Q8: What happens when the live data distribution drifts?
**Answer:** The system implements a drift detection module (`src/churn_prediction/drift_detection.py`) computing the **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests between reference training distributions and incoming production telemetry. If $\text{PSI} < 0.10$, features are stable. If $\text{PSI} \ge 0.25$, an alert is triggered, prompting evaluation in Staging and retraining via `scripts/train.py`.
