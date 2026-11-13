"""
Comprehensive Performance Benchmark Suite for Customer Churn Prediction Engine.
Measures:
- Model bundle loading time
- Memory consumption (RSS)
- Single prediction latency distribution (min, mean, p50, p90, p95, p99, max)
- Batch scaling throughput across varied batch sizes
- Generates publication figures 23 and 24
"""

import json
import os
import time
import psutil
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from churn_prediction.config import (
    MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR,
    REPORTS_DIR
)
from churn_prediction.model_inference import ChurnPredictor

PHASE_22_DIR = REPORTS_DIR / "phase_22"

def main():
    print("=== Starting Phase 22 Performance Engineering Benchmarking ===")
    PHASE_22_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    process = psutil.Process(os.getpid())

    # 1. Baseline Memory
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    # 2. Model Loading Benchmark
    t0 = time.perf_counter()
    bundle_path = MODELS_DIR / "churn_pipeline_bundle.joblib"
    bundle = joblib.load(bundle_path)
    predictor = ChurnPredictor(bundle_path)
    model_load_time_sec = time.perf_counter() - t0

    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    model_memory_footprint_mb = mem_after_mb - mem_before_mb

    print(f"Model Load Time: {model_load_time_sec:.4f} s")
    print(f"Memory Footprint: {model_memory_footprint_mb:.2f} MB (Total Process RSS: {mem_after_mb:.2f} MB)")

    # 3. Single Prediction Latency Distribution (200 trials)
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    sample_records = test_df.head(50).to_dict(orient="records")

    single_latencies_ms = []
    # Warmup
    for _ in range(20):
        predictor.predict_single(sample_records[0])

    for i in range(200):
        rec = sample_records[i % len(sample_records)]
        t_start = time.perf_counter()
        _ = predictor.predict_single(rec)
        single_latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

    p_series = pd.Series(single_latencies_ms)
    single_stats = {
        "trials": len(single_latencies_ms),
        "mean_ms": round(float(p_series.mean()), 2),
        "min_ms": round(float(p_series.min()), 2),
        "p50_ms": round(float(p_series.median()), 2),
        "p90_ms": round(float(p_series.quantile(0.90)), 2),
        "p95_ms": round(float(p_series.quantile(0.95)), 2),
        "p99_ms": round(float(p_series.quantile(0.99)), 2),
        "max_ms": round(float(p_series.max()), 2)
    }
    print("Single Prediction Latency (ms):", single_stats)

    # Plot Single Latency Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(single_latencies_ms, kde=True, color="#1f77b4", bins=25, ax=ax)
    ax.axvline(single_stats["p50_ms"], color="green", linestyle="--", label=f"p50: {single_stats['p50_ms']:.1f}ms")
    ax.axvline(single_stats["p95_ms"], color="red", linestyle="--", label=f"p95: {single_stats['p95_ms']:.1f}ms")
    ax.set_title("Single Prediction Latency Distribution (200 Trials)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Latency (milliseconds)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig24_path = FIGURES_DIR / "24_single_prediction_latency_distribution.png"
    fig.savefig(fig24_path, dpi=300)
    plt.close()
    print(f"Saved latency distribution figure to {fig24_path}")

    # 4. Batch Scaling Benchmark
    batch_sizes = [10, 50, 100, 250, 500, 1000, len(test_df)]
    batch_results = []

    for b_size in batch_sizes:
        sub_df = test_df.head(b_size)
        t_b_start = time.perf_counter()
        _ = predictor.predict_batch(sub_df)
        elapsed_sec = time.perf_counter() - t_b_start
        throughput_rec_per_sec = len(sub_df) / elapsed_sec if elapsed_sec > 0 else 0.0

        batch_results.append({
            "batch_size": int(b_size),
            "elapsed_seconds": round(float(elapsed_sec), 4),
            "ms_per_record": round(float((elapsed_sec / len(sub_df)) * 1000.0), 3),
            "throughput_records_per_sec": round(float(throughput_rec_per_sec), 1)
        })

    batch_df = pd.DataFrame(batch_results)
    print("\nBatch Scaling Benchmark:\n", batch_df)

    # Plot Batch Latency & Throughput
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color = "tab:blue"
    ax1.set_xlabel("Batch Size (Number of Customer Records)", fontsize=11)
    ax1.set_ylabel("Total Duration (seconds)", color=color, fontsize=11)
    ax1.plot(batch_df["batch_size"], batch_df["elapsed_seconds"], "o-", color=color, linewidth=2, label="Latency (s)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Throughput (records / sec)", color=color, fontsize=11)
    ax2.plot(batch_df["batch_size"], batch_df["throughput_records_per_sec"], "s--", color=color, linewidth=2, label="Throughput (rec/s)")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Batch Inference Latency and Throughput Scaling", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig23_path = FIGURES_DIR / "23_batch_latency_throughput.png"
    fig.savefig(fig23_path, dpi=300)
    plt.close()
    print(f"Saved batch throughput figure to {fig23_path}")

    # 5. Export Summary JSON
    benchmark_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_environment": {
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        },
        "model_loading": {
            "load_time_seconds": round(model_load_time_sec, 4),
            "process_rss_mb": round(mem_after_mb, 2),
            "model_memory_delta_mb": round(model_memory_footprint_mb, 2)
        },
        "single_prediction_latency_ms": single_stats,
        "batch_scaling": batch_results
    }

    with open(PHASE_22_DIR / "performance_benchmark_results.json", "w") as f:
        json.dump(benchmark_report, f, indent=2)

    print(f"Saved benchmark results to {PHASE_22_DIR / 'performance_benchmark_results.json'}")
    print("=== Phase 22 Performance Benchmarking Completed Successfully ===")

if __name__ == "__main__":
    main()
