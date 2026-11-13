"""Production telemetry logging and serving performance monitoring."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from churn_prediction.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

MONITORING_LOG_FILE = PROJECT_ROOT / "reports" / "monitoring_events.jsonl"


class PredictionLogger:
    """Non-PII audit logger recording inference events for observability and governance."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = Path(log_path) if log_path else MONITORING_LOG_FILE
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        customer_id: Optional[str],
        probability: float,
        prediction: int,
        risk_category: str,
        threshold: float,
        latency_ms: float = 0.0,
        model_version: str = "1.0.0",
    ) -> Dict[str, Any]:
        """Record an inference event without storing raw personal information."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customer_id": customer_id or "anonymous",
            "model_version": model_version,
            "decision_threshold": float(threshold),
            "churn_probability": float(np.round(probability, 4)),
            "churn_prediction": int(prediction),
            "risk_category": risk_category,
            "latency_ms": float(np.round(latency_ms, 2)),
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append to prediction audit log: {e}")

        return event

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Read recent logged events."""
        if not self.log_path.exists():
            return []
        events = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line.strip()))
        except Exception as e:
            logger.warning(f"Failed to read monitoring log: {e}")
        return events[-limit:]

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Compute serving telemetry summary from log."""
        events = self.get_recent_events(limit=1000)
        if not events:
            return {"total_predictions": 0, "status": "No inference events logged yet"}

        probs = [e["churn_probability"] for e in events]
        preds = [e["churn_prediction"] for e in events]
        latencies = [e["latency_ms"] for e in events]
        tiers = [e["risk_category"] for e in events]

        return {
            "total_predictions": len(events),
            "mean_churn_probability": float(np.round(np.mean(probs), 4)),
            "flagged_churn_ratio": float(np.round(np.mean(preds), 4)),
            "mean_latency_ms": float(np.round(np.mean(latencies), 2)),
            "risk_tier_counts": {
                "High Risk": tiers.count("High Risk"),
                "Medium Risk": tiers.count("Medium Risk"),
                "Low Risk": tiers.count("Low Risk"),
            },
        }
