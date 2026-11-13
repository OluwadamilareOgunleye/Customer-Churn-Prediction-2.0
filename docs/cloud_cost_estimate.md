# Cloud Cost Estimation & Multi-Cloud Comparison

## Overview

This analysis provides realistic cost breakdowns for deploying the Customer Churn Prediction system across major cloud providers (AWS, Google Cloud, Microsoft Azure) as well as free/low-cost alternatives for portfolio presentation.

---

## 1. Estimated Workload Profile
- **Traffic Volume:** 50,000 inference requests / month (approx. 1 request per minute average, 10 req/sec peak).
- **Batch Predictions:** 1 weekly batch scoring run (7,000 customers).
- **Model Bundle Size:** ~200 KB (`churn_pipeline_bundle.joblib`).
- **Memory Footprint:** API container requires ~350 MB RAM; Frontend container requires ~400 MB RAM.

---

## 2. Multi-Cloud Cost Breakdown

### A. Amazon Web Services (AWS) — ECS Fargate & S3
| Resource Component | Specifications | Monthly Cost (USD) |
| :--- | :--- | :--- |
| **ECS Fargate (API)** | 1 task, 0.25 vCPU, 0.5 GB RAM (24/7) | \$7.23 |
| **ECS Fargate (Frontend)** | 1 task, 0.25 vCPU, 0.5 GB RAM (24/7) | \$7.23 |
| **Application Load Balancer**| 1 ALB + 1 LCU | \$16.20 |
| **Amazon S3 Standard** | 5 GB storage + 1,000 PUT/GET requests | \$0.15 |
| **Amazon CloudWatch** | 5 GB log ingestion + 3 dashboards | \$2.50 |
| **Data Transfer Out** | 10 GB egress / month | \$0.90 |
| **Total AWS Monthly Cost** | | **\$34.21 / month** |

### B. Google Cloud Platform (GCP) — Cloud Run & Cloud Storage
| Resource Component | Specifications | Monthly Cost (USD) |
| :--- | :--- | :--- |
| **Cloud Run (API)** | 1 vCPU, 512 MB RAM, scale-to-zero when idle | \$3.10 |
| **Cloud Run (Frontend)** | 1 vCPU, 512 MB RAM, scale-to-zero when idle | \$3.10 |
| **Cloud Load Balancing** | Serverless Network Endpoint Group | \$18.00 |
| **Google Cloud Storage** | Standard regional bucket | \$0.12 |
| **Cloud Logging** | First 50 GB free tier | \$0.00 |
| **Total GCP Monthly Cost** | | **\$24.32 / month** |

### C. Zero-Cost / Portfolio Tier (Recommended for Personal Demo)
| Service | Role | Cost |
| :--- | :--- | :--- |
| **Render.com Free Tier** | FastAPI container (512 MB RAM, sleeps after 15m inactivity) | \$0.00 |
| **Hugging Face Spaces** | Streamlit application container (16 GB RAM, 2 vCPU) | \$0.00 |
| **GitHub Actions** | Free public runner minutes (2,000 min/mo) | \$0.00 |
| **SQLite / Local Storage**| Embedded MLflow and telemetry tracking | \$0.00 |
| **Total Portfolio Tier** | | **\$0.00 / month** |

---

## 3. Cost Optimization Strategies
1. **Scale-to-Zero:** Use container engines supporting scale-to-zero (GCP Cloud Run or Render Free) during periods of non-use.
2. **Stateless API Design:** Compute workers can be terminated immediately without persistent disk attach fees.
3. **Model Artifact Caching:** Cache the 200 KB model bundle in container RAM upon startup rather than re-downloading from S3 on every prediction call.
