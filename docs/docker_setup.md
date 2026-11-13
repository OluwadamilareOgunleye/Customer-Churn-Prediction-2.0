# Phase 11: Containerization and Docker Deployment Guide

## 1. Overview

The customer churn prediction system is containerized as a multi-service architecture using Docker and Docker Compose:
- **Backend Service (`backend`):** FastAPI REST application exposing prediction and health endpoints on port `8000`.
- **Frontend Service (`frontend`):** Streamlit web dashboard on port `8501`, networked directly with the backend container.

---

## 2. Docker Architecture & Optimization

- **Multi-Stage Builds:** Uses `python:3.11-slim` with a compilation stage (`builder`) to compile wheels and a minimal final stage (`runner`) without build toolchains, minimizing image size and attack surface.
- **Embedded Healthchecks:** Both containers define native Python-based HTTP health probes, preventing dependencies on external tools like `curl`.
- **Isolated Bridge Network:** Inter-container communication is managed over a private virtual bridge network (`churn-network`).

---

## 3. Quickstart: Running via Docker Compose

### Step 1: Build and Launch Containers
```bash
docker compose up --build -d
```

### Step 2: Verify Running Containers & Health
```bash
docker compose ps
```

### Step 3: Access Services
- **Streamlit Dashboard:** `http://localhost:8501`
- **FastAPI Documentation:** `http://localhost:8000/docs`
- **Health Endpoint:** `http://localhost:8000/health`

### Step 4: Stop Containers
```bash
docker compose down
```

---

## 4. Building and Running Individual Images Manually

### FastAPI Backend
```bash
docker build -f Dockerfile.api -t churn-prediction-api:latest .
docker run -d -p 8000:8000 --name churn-api churn-prediction-api:latest
```

### Streamlit Frontend
```bash
docker build -f Dockerfile.frontend -t churn-prediction-ui:latest .
docker run -d -p 8501:8501 -e API_BASE_URL=http://localhost:8000 --name churn-ui churn-prediction-ui:latest
```
