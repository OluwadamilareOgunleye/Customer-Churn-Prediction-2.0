"""
Cloud & Local Artifact Storage Abstraction.

Supports loading and saving model bundles and artifacts across:
- Local filesystem (development / staging / CI)
- Cloud Object Storage (AWS S3, GCP Cloud Storage, Azure Blob Storage) via uniform URI schemes
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any, Optional
import joblib


class BaseArtifactStore(ABC):
    """Abstract base class for model artifact storage backends."""

    @abstractmethod
    def load_artifact(self, uri: str) -> Any:
        """Load an artifact from the given URI."""
        pass

    @abstractmethod
    def save_artifact(self, obj: Any, uri: str) -> str:
        """Save an artifact to the given URI and return the resolved path."""
        pass


class LocalArtifactStore(BaseArtifactStore):
    """Local filesystem artifact store."""

    def load_artifact(self, uri: str) -> Any:
        # Strip file:// prefix if present
        clean_path = uri.replace("file://", "")
        path = Path(clean_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local artifact not found at: {path}")
        return joblib.load(path)

    def save_artifact(self, obj: Any, uri: str) -> str:
        clean_path = uri.replace("file://", "")
        path = Path(clean_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)
        return str(path)


class S3ArtifactStore(BaseArtifactStore):
    """AWS S3 artifact store abstraction (boto3 optional dependency)."""

    def __init__(self, boto3_client: Optional[Any] = None):
        self._client = boto3_client

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("s3")
            except ImportError:
                raise ImportError("boto3 is required for s3:// artifact URIs. Install via pip install boto3.")
        return self._client

    def _parse_s3_uri(self, uri: str):
        # Format: s3://bucket-name/key/path/to/artifact.joblib
        stripped = uri.replace("s3://", "")
        parts = stripped.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def load_artifact(self, uri: str) -> Any:
        import io
        client = self._get_client()
        bucket, key = self._parse_s3_uri(uri)
        buffer = io.BytesIO()
        client.download_fileobj(Bucket=bucket, Key=key, Fileobj=buffer)
        buffer.seek(0)
        return joblib.load(buffer)

    def save_artifact(self, obj: Any, uri: str) -> str:
        import io
        client = self._get_client()
        bucket, key = self._parse_s3_uri(uri)
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        buffer.seek(0)
        client.upload_fileobj(Fileobj=buffer, Bucket=bucket, Key=key)
        return uri


def get_artifact_store(uri: str) -> BaseArtifactStore:
    """
    Factory function resolving the appropriate store implementation by URI scheme.
    """
    if uri.startswith("s3://"):
        return S3ArtifactStore()
    else:
        return LocalArtifactStore()


def load_model_bundle_from_uri(uri: str) -> Any:
    """Convenience helper to load model bundles from any supported URI scheme."""
    store = get_artifact_store(uri)
    return store.load_artifact(uri)
