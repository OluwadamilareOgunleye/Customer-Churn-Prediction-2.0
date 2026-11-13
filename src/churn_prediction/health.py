"""
System Health Monitoring and Liveness/Readiness Diagnostics.

Differentiates:
- Liveness: Process responsiveness and basic runtime availability
- Readiness: Deep dependency checks, pipeline bundle availability, memory health
"""

from typing import Dict, Any
from pathlib import Path
import psutil
import joblib


class HealthChecker:
    """Provides system diagnostic status and readiness evaluation."""

    @staticmethod
    def check_liveness() -> Dict[str, Any]:
        """
        Liveness probe: verifies process is alive and responsive.
        """
        return {
            "status": "alive",
            "uptime_seconds": round(psutil.time.time() - psutil.boot_time(), 1),
            "cpu_percent": psutil.cpu_percent(interval=None)
        }

    @staticmethod
    def check_readiness(model_path: Path, predictor_instance: Any = None) -> Dict[str, Any]:
        """
        Readiness probe: verifies model bundle exists, is non-corrupt, and loaded in memory.
        """
        issues = []
        model_ready = False
        threshold = None

        if predictor_instance is not None:
            model_ready = getattr(predictor_instance, "pipeline", None) is not None
            threshold = getattr(predictor_instance, "default_threshold", getattr(predictor_instance, "threshold", 0.30))
        elif model_path and Path(model_path).exists():
            try:
                bundle = joblib.load(model_path)
                model_ready = "pipeline" in bundle
                threshold = bundle.get("threshold", 0.30)
            except Exception as e:
                issues.append(f"Corrupted model bundle at {model_path}: {str(e)}")
        else:
            issues.append(f"Model artifact not found at {model_path}")

        memory_info = psutil.virtual_memory()
        mem_available_mb = memory_info.available / (1024 * 1024)

        if mem_available_mb < 50.0:
            issues.append(f"Critically low system memory: {mem_available_mb:.1f} MB available")

        is_ready = len(issues) == 0 and model_ready

        return {
            "ready": is_ready,
            "status": "ready" if is_ready else "not_ready",
            "model_loaded": model_ready,
            "decision_threshold": threshold,
            "available_memory_mb": round(mem_available_mb, 1),
            "issues": issues
        }
