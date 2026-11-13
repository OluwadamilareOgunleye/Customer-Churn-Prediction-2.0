"""Local and lightweight Model Registry managing model lifecycle and versioning."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from churn_prediction.config import MODELS_DIR

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_FILE = MODELS_DIR / "registry.json"


class ModelMetadata(BaseModel):
    """Metadata schema for a registered model version."""

    model_name: str
    version: int
    model_type: str
    status: str  # "Development", "Staging", "Production", "Archived"
    created_at: str
    artifact_path: str
    decision_threshold: float
    metrics: Dict[str, Any]
    run_id: Optional[str] = None
    notes: Optional[str] = None


class ModelRegistry:
    """Manages versioned model artifacts and deployment stage transitions."""

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_FILE
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, List[Dict[str, Any]]] = self._load()

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.registry_path.exists():
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading registry at {self.registry_path}: {e}")
            return {}

    def _save(self):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._models, f, indent=2)

    def register_model(
        self,
        model_name: str,
        artifact_path: Path,
        model_type: str,
        decision_threshold: float,
        metrics: Dict[str, Any],
        stage: str = "Staging",
        run_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> ModelMetadata:
        """Register a new version of a trained model."""
        if model_name not in self._models:
            self._models[model_name] = []

        new_version = len(self._models[model_name]) + 1
        meta = ModelMetadata(
            model_name=model_name,
            version=new_version,
            model_type=model_type,
            status=stage,
            created_at=datetime.now(timezone.utc).isoformat(),
            artifact_path=str(artifact_path),
            decision_threshold=float(decision_threshold),
            metrics=metrics,
            run_id=run_id,
            notes=notes,
        )

        # If registering directly as Production, archive previous production versions
        if stage == "Production":
            for m in self._models[model_name]:
                if m["status"] == "Production":
                    m["status"] = "Archived"

        self._models[model_name].append(meta.model_dump())
        self._save()
        logger.info(f"Registered model '{model_name}' version {new_version} with status '{stage}'")
        return meta

    def transition_stage(
        self,
        model_name: str,
        version: int,
        new_stage: str,
    ) -> Optional[ModelMetadata]:
        """Transition a specific model version to a new stage."""
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not found in registry.")

        target = None
        for m in self._models[model_name]:
            if m["version"] == version:
                target = m
                break

        if not target:
            raise KeyError(f"Version {version} of model '{model_name}' not found.")

        # If promoting to Production, archive existing Production model
        if new_stage == "Production":
            for m in self._models[model_name]:
                if m["status"] == "Production" and m["version"] != version:
                    m["status"] = "Archived"

        target["status"] = new_stage
        self._save()
        logger.info(f"Model '{model_name}' version {version} transitioned to '{new_stage}'")
        return ModelMetadata(**target)

    def get_production_model(self, model_name: str) -> Optional[ModelMetadata]:
        """Retrieve the current active Production model."""
        if model_name not in self._models:
            return None
        for m in reversed(self._models[model_name]):
            if m["status"] == "Production":
                return ModelMetadata(**m)
        return None

    def list_versions(self, model_name: str) -> List[ModelMetadata]:
        """List all registered versions for a model."""
        if model_name not in self._models:
            return []
        return [ModelMetadata(**m) for m in self._models[model_name]]
