# Phase 12: Cloud Deployment Guide

## 1. Cloud Platform Evaluation & Selection

To deploy this dual-service application (FastAPI backend + Streamlit frontend) reliably and cost-effectively, the following hosting platforms were evaluated:

| Platform | Free / Low-Cost Tier | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Render.com** | Free tier available (750 free instance hours/month) | Native Docker support, automatic SSL, zero-config web services. | Free tier spins down after 15 minutes of inactivity (50s cold start). | **Recommended Option 1** (Best for portfolio presentation) |
| **Hugging Face Spaces** | 100% Free (Community CPU: 2 vCPU, 16 GB RAM) | Highly stable, no sleep on basic spaces, ideal for Streamlit UI. | Designed for monolithic single-container apps rather than split REST + UI. | **Recommended Option 2** (For standalone Streamlit demo) |
| **Railway.app** | $5 free trial credit, usage-based ($5/mo) | Excellent multi-container orchestration, fast builds, persistent domains. | Requires credit card verification after trial credit. | **Recommended for Production** |
| **Fly.io** | Free tier allowances (up to 3 shared-cpu-1x VMs) | Global edge hosting, low latency. | More complex CLI configuration (`flyctl`). | Advanced |

---

## 2. Step-by-Step Deployment on Render.com (Recommended Free-Tier)

Render allows deploying both the API and Streamlit app as independent Web Services connected via environment variables.

### A. Deploy FastAPI Backend
1. Sign in to [Render.com](https://render.com).
2. Click **New +** → **Web Service**.
3. Connect your Git repository.
4. Set the configuration:
   - **Name:** `telco-churn-api`
   - **Environment:** `Docker`
   - **Dockerfile Path:** `./Dockerfile.api`
   - **Region:** Closest region (e.g. Frankfurt / Ohio)
   - **Instance Type:** `Free`
5. Under **Environment Variables**, add:
   - `PORT`: `8000`
   - `API_DEBUG`: `false`
6. Click **Create Web Service**. Once built, note your backend URL:
   `https://telco-churn-api.onrender.com`

### B. Deploy Streamlit Frontend
1. Click **New +** → **Web Service**.
2. Connect the same Git repository.
3. Set the configuration:
   - **Name:** `telco-churn-ui`
   - **Environment:** `Docker`
   - **Dockerfile Path:** `./Dockerfile.frontend`
   - **Instance Type:** `Free`
4. Under **Environment Variables**, configure the API target:
   - `API_BASE_URL`: `https://telco-churn-api.onrender.com`
   - `STREAMLIT_SERVER_PORT`: `8501`
5. Click **Create Web Service**. Render will build and deploy your Streamlit dashboard at:
   `https://telco-churn-ui.onrender.com`

---

## 3. Alternative: Deploying Standalone Streamlit on Hugging Face Spaces

1. Create a free account at [Hugging Face](https://huggingface.co).
2. Create a new Space:
   - **Space Name:** `telco-churn-predictor`
   - **SDK:** `Streamlit`
   - **License:** `MIT`
3. Push the repository contents to the Hugging Face Space git remote.
4. Because `frontend/utils/api_client.py` includes automatic fallback to the local `ChurnPredictor`, the Streamlit application will function immediately without needing an external REST API!

---

## 4. Post-Deployment Verification Checklist

Once deployed to your chosen platform, execute this checklist to verify production health:
- [ ] Access `https://your-api.com/health` $\rightarrow$ returns `{"status": "healthy", "model_loaded": true}`.
- [ ] Access `https://your-api.com/docs` $\rightarrow$ Swagger interactive API documentation loads properly.
- [ ] Submit a sample prediction via `/predict` $\rightarrow$ returns `churn_probability` and `risk_category`.
- [ ] Open the Streamlit frontend $\rightarrow$ dashboard displays live metrics and green health indicator.
- [ ] Submit single prediction form $\rightarrow$ risk gauge and recommendations render cleanly.
- [ ] Run batch prediction test $\rightarrow$ scores 100 sample records and exports CSV correctly.
