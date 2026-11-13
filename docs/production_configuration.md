# Phase 12: Production Environment Configuration Guide

## 1. Overview

Deploying machine learning applications to production requires strict segregation between development and production configurations:
- **Zero Hardcoded Credentials:** All environment-specific variables, API URLs, and secret keys must be supplied via runtime environment variables.
- **Stateless Serving:** Both the FastAPI backend and Streamlit frontend are stateless services. The model artifact (`models/churn_pipeline_bundle.joblib`) is baked into the immutable deployment image or mounted from persistent storage.
- **Fail-Safe Health Checks:** Production load balancers depend on `/health` to direct incoming traffic only to healthy worker processes.

---

## 2. Environment Variables Specification

| Variable | Scope | Development Default | Production Recommended | Description |
| :--- | :--- | :--- | :--- | :--- |
| `PORT` | API | `8000` | Assigned by cloud provider ($PORT) | Port for Uvicorn worker |
| `API_DEBUG` | API | `false` | `false` | Disables internal debug exception tracebacks |
| `API_BASE_URL` | Frontend | `http://127.0.0.1:8000` | `https://api.yourdomain.com` | Target URL for backend inference API |
| `CORS_ORIGINS` | API | `*` | `https://your-frontend-domain.com` | Strict whitelist of allowed web origins |
| `STREAMLIT_SERVER_PORT` | Frontend | `8501` | Assigned by cloud provider | Port for Streamlit server |
| `STREAMLIT_SERVER_HEADLESS`| Frontend | `true` | `true` | Suppresses browser pop-ups on launch |
| `STREAMLIT_SERVER_ENABLE_CORS`| Frontend | `false` | `false` | Disables Streamlit internal CORS checks |

---

## 3. Production Hardening Checklist

1. **Uvicorn Concurrency:** In production environments with multi-core CPUs, launch Uvicorn with multiple workers:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 4
   ```
2. **Reverse Proxy (Nginx / Cloudflare):** Terminate SSL/TLS at the reverse proxy layer to protect customer data in transit.
3. **Structured Logging:** Centralize API logs in JSON format for automated ingestion by Datadog, Grafana Loki, or AWS CloudWatch.
4. **Memory Allocation:** Ensure the container has at least 1 GB RAM to comfortably host the Scikit-Learn tree ensemble in memory without triggering OOM (Out Of Memory) restarts.
