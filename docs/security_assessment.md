# Application Security Assessment & Threat Model

## 1. Executive Summary

This security audit conducted a comprehensive threat modeling and vulnerability assessment of the Customer Churn Prediction platform across the FastAPI backend, Streamlit frontend, inference endpoints, environment management, and containerization layers.

---

## 2. Threat Modeling & Vulnerability Findings

| Component | Potential Threat / Vector | Inherent Risk | Mitigating Countermeasure Implemented | Residual Risk |
| :--- | :--- | :--- | :--- | :--- |
| **REST API (`/predict/batch`)** | Denial of Service (DoS) via massive memory exhaustion batch payload | **High** | Hard cap at **2,500 records** (`HTTP 413 Payload Too Large`) enforced in `api/routes/prediction.py`. | **Low** |
| **REST API (All Endpoints)** | Automated request flooding / scraping | **Medium** | Sliding-window in-memory rate limiter enforcing **180 requests/minute per client IP** (`HTTP 429 Too Many Requests`). | **Low** |
| **HTTP Transport** | Clickjacking & MIME-confusion attacks | **Medium** | Standard production HTTP security headers injected via middleware: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Strict-Transport-Security`, `CSP`. | **Low** |
| **CORS Configuration** | Unauthorized cross-origin resource access | **Medium** | Restricted from wildcard (`*`) to explicit local and trusted frontend origins via `CORS_ORIGINS` environment variable. | **Low** |
| **Error Handling** | Information disclosure via leaked stack traces | **Medium** | Global exception handler catches unhandled exceptions and returns sanitized JSON with HTTP 500 without exposing code paths. | **Low** |
| **Model Ingestion** | Deserialization attacks on arbitrary code execution | **High** | Pipelines are serialized with `joblib` from verified local paths or read-only S3 buckets with restricted IAM roles. Never deserialize untrusted public files. | **Low** |
| **Secrets Management** | Accidental credential leakage in public repositories | **Critical** | `.gitignore` strictly excludes `.env`, `credentials.json`, `mlflow.db`, and `*.sqlite`. `.env.example` contains only non-sensitive template keys. | **Negligible** |

---

## 3. Production Authentication Architecture

### Current Status
For internal microservice deployment and portfolio demonstration, endpoints operate without authentication overhead to facilitate immediate local testing and seamless frontend-to-backend communication.

### Enterprise Production Gateway Pattern
In an enterprise telecommunications perimeter, the API should sit behind an API Gateway (e.g. AWS API Gateway, Kong, or Traefik) enforcing:
1. **OAuth 2.0 / OIDC with JWT:** All `/predict` calls require `Authorization: Bearer <token>` signed by corporate Okta / Azure AD.
2. **mTLS (Mutual TLS):** Between Streamlit frontend containers and FastAPI inference pods.
3. **Role-Based Access Control (RBAC):** Restricting single predictions to Customer Support agents and batch predictions to Data Science / Marketing Analysts.
