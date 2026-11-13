# Performance Engineering & Benchmarking Analysis

## 1. Executive Summary

This performance analysis rigorously evaluated the computational efficiency, latency distribution, memory footprint, and horizontal batch scaling characteristics of the Customer Churn Prediction engine. All metrics are empirically measured on the project benchmark test suite (`tests/performance/benchmark.py`).

---

## 2. System Hardware & Benchmark Environment
- **Operating System:** Windows 64-bit
- **Python Runtime:** Python 3.11.9 (64-bit)
- **Serving Architecture:** FastAPI (Uvicorn) with lifespan-managed Scikit-learn Pipeline singleton
- **Model Bundle Size:** ~201 KB (`models/churn_pipeline_bundle.joblib`)

---

## 3. Measured Performance Benchmarks

### A. Memory Footprint & Initialization
- **Model Bundle Loading Duration:** **$0.8281$ seconds**
- **Model Pipeline Memory Overhead:** **$14.36$ MB**
- **Active Serving Engine RSS:** **$196.54$ MB** (well within standard 512 MB microservice container boundaries).

### B. Single Inference Latency Distribution (200 Sequential Trials)
| Percentile Metric | Measured Latency (ms) | Target SLA | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Minimum** | $13.61$ ms | $< 50$ ms | Meets SLA |
| **Median (p50)** | **$18.26$ ms** | $< 50$ ms | Exceeds Target |
| **90th Percentile (p90)**| $19.23$ ms | $< 75$ ms | Exceeds Target |
| **95th Percentile (p95)**| **$19.74$ ms** | $< 100$ ms | Exceeds Target |
| **99th Percentile (p99)**| $20.64$ ms | $< 150$ ms | Exceeds Target |
| **Maximum** | $23.95$ ms | $< 250$ ms | Meets SLA |

*(Artifact: `reports/figures/24_single_prediction_latency_distribution.png`)*

---

## 4. Batch Inference Scaling & Throughput

To evaluate vectorization and ColumnTransformer batch efficiency, inference was evaluated across increasing customer batch sizes:

| Batch Size ($N$) | Total Latency (s) | Per-Record Latency (ms) | Throughput (Records / Sec) |
| :--- | :--- | :--- | :--- |
| 10 records | $0.0257$ s | $2.569$ ms | $389.3$ rec/s |
| 50 records | $0.0286$ s | $0.571$ ms | $1,750.8$ rec/s |
| 100 records | $0.0462$ s | $0.462$ ms | $2,163.2$ rec/s |
| 250 records | $0.0864$ s | $0.346$ ms | $2,893.1$ rec/s |
| 500 records | $0.1252$ s | $0.250$ ms | $3,993.2$ rec/s |
| 1,000 records | $0.2344$ s | $0.234$ ms | $4,266.2$ rec/s |
| **1,409 records (Full Test)**| **$0.2761$ s** | **$0.196$ ms** | **$5,104.1$ rec/s** |

*(Artifact: `reports/figures/23_batch_latency_throughput.png`)*

### Key Engineering Insights
- **Sub-Linear Latency Scaling:** Processing 1,409 customer records takes only $0.276$ seconds—less than 11x the duration of scoring 10 records ($0.025$s).
- **Peak Throughput:** Vectorized NumPy / Pandas array operations inside the pipeline achieve over **5,100 predictions per second** on standard CPU hardware without GPU acceleration.

---

## 5. Load Testing & Locust Scenario
A load-testing scenario is established in [`tests/performance/locustfile.py`](file:///c:/Users/USER/Desktop/Data%20science%20projects/Customer%20Churn%20prediction/tests/performance/locustfile.py), generating realistic traffic distributions:
- $30\%$ GET `/health` probes
- $50\%$ POST `/predict` single inference
- $20\%$ POST `/predict/batch` small batch inference

To execute headless load tests locally against a running API server:
```bash
locust -f tests/performance/locustfile.py --headless -u 20 -r 5 --run-time 30s --host http://localhost:8000
```
