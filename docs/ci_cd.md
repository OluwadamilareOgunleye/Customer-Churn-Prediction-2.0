# CI/CD & Automated Machine Learning Workflows

## Overview

This project includes a continuous integration and delivery architecture powered by **GitHub Actions**. The pipeline enforces code quality, executes unit and integration test suites, validates Docker container builds, automates controlled model retraining workflows, and establishes gated deployment pathways.

---

## Workflow Suite

### 1. Continuous Integration (`.github/workflows/ci.yml`)
- **Triggers:** Push to `main`, `master`, `develop`, or any Pull Request.
- **Tasks:**
  - Python 3.11 environment initialization with pip dependency caching.
  - Flake8 linting and syntax error detection.
  - Pytest execution across unit and integration suites (`tests/`).
  - Code coverage reporting enforcing $\ge 50\%$ overall codebase coverage threshold.
  - API smoke test verifying FastAPI application initialization.

### 2. Container Build Validation (`.github/workflows/docker-build.yml`)
- **Triggers:** Push or Pull Request touching core application files.
- **Tasks:**
  - Validates `Dockerfile.api` build with Docker Buildx.
  - Validates `Dockerfile.frontend` build.
  - Validates `docker-compose.yml` multi-container specification via `docker compose config`.
  - **Security Rule:** Images are tagged for local test verification only and are **never** pushed to public registries automatically.

### 3. Model Training & Evaluation Workflow (`.github/workflows/model-training.yml`)
- **Triggers:** Manual execution via `workflow_dispatch`.
- **Tasks:**
  - Runs reproducible training script `python scripts/train.py --config configs/train_config.json`.
  - Computes candidate validation metrics.
  - **Quality Gate:** Candidate must achieve holdout $\text{ROC-AUC} \ge 0.80$ and $\text{Recall} \ge 0.70$.
  - **Governance Rule:** A newly trained candidate is **never** promoted directly to `Production`. It is recorded in the model registry as `Candidate` or `Staging`, requiring manual review before promotion.
  - Uploads `training_run_summary.json` as an artifact.

### 4. Controlled Deployment (`.github/workflows/deployment.yml`)
- **Triggers:** Published GitHub release tag or manual dispatch with target environment (`staging` or `production`).
- **Tasks:**
  - Requires GitHub Environment Approval gates for production promotion.
  - Verifies presence of serialized model bundle and service entry points.
  - Zero plain-text credentials in logs.

---

## Local Git Repository Setup

When configuring a remote repository:
1. Initialize git (if not already done): `git init`
2. Add remote: `git remote add origin <user-repo-url>`
3. Configure repository secrets if cloud deployment tokens are needed:
   - `RENDER_API_KEY` or `RENDER_DEPLOY_HOOK_URL`
   - `HF_TOKEN`
4. Establish branch protection rules on `main` requiring the `Lint, Test & Validate` check to pass before merging.
