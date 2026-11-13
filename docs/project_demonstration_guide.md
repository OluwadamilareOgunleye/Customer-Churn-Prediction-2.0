# Project Demonstration & Walkthrough Guide

## 1. Introduction for Demonstrators

This guide outlines the recommended live demonstration narrative for presenting this project to hiring managers, interview panels, and technical stakeholders in 5 to 10 minutes.

---

## 2. Live Demonstration Script

### Step 1: System Overview & Live Health Check (1 min)
1. Start the API server:
   ```powershell
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
2. In browser, navigate to Swagger documentation: `http://localhost:8000/docs`
3. Point out the dual probes:
   - Execute `GET /health` $\rightarrow$ Returns `status: "healthy"`
   - Execute `GET /ready` $\rightarrow$ Returns `ready: true`, `decision_threshold: 0.30`, and memory metrics.

### Step 2: Interactive Streamlit Dashboard (3 min)
1. In a second terminal, launch the Streamlit frontend:
   ```powershell
   streamlit run frontend/app.py --server.port 8501
   ```
2. Navigate to `http://localhost:8501`
3. **Tab 1: Overview & Performance:**
   - Highlight the test set benchmark metrics (ROC-AUC: 0.8570, Recall: 77.01%).
   - Explain why threshold $\tau = 0.30$ reduces missed churners by $51.4\%$.
4. **Tab 2: Single Customer Scoring:**
   - Load sample profile (e.g. Month-to-month, Fiber optic, Electronic check).
   - Click **Run Prediction** $\rightarrow$ Show probability ($72.5\%$), risk category (*High Risk*), and SHAP attribution bars explaining why the customer is likely to leave.
5. **Tab 3: Batch Prediction:**
   - Click "Score Sample Batch" or upload a CSV $\rightarrow$ Displays interactive risk-stratified customer table with immediate export.

### Step 3: Business Intelligence & Strategic Retention ROI (3 min)
1. Navigate to **"5. Business Insights & Retention"**:
   - Tab 1: Show the **Retention Playbook** (First-Year Shield, Annual Incentives, Tech Support Bundles).
   - Tab 2: Show the **K-Means Personas** scatter plot (Silhouette = 0.4768) and the high churn risk of *New High-Spend / Tech Subscribers* ($51.5\%$).
   - Tab 3: Open the **Interactive Retention ROI Simulator**:
     - Adjust the CLV slider (\$1,200) and campaign cost (\$50).
     - Show the live metric card: **+$15,700 Net Profit Advantage** at $\tau = 0.30$ vs $\tau = 0.50$.

### Step 4: Engineering Rigor & MLOps Infrastructure (2 min)
1. Show test suite execution:
   ```powershell
   pytest -v
   ```
   Highlight: **34 passing tests** across unit, API, integration, and reliability.
2. Open MLflow tracking and Model Registry:
   - Show `models/registry.json` tracking versions and staging gates.
3. Show `infrastructure/main.tf` and GitHub Actions workflows in `.github/workflows/`.
