"""Artifact service implementations for codin agents.

This module provides artifact management services including storage, retrieval,
and metadata tracking for task artifacts, files, and generated content.
"""

import abc
import typing as _t

from datetime import datetime


__all__ = [
    'ArtifactService',
    'InMemoryArtifactService',
]


class ArtifactService(abc.ABC):
    """Service for managing code artifacts and files."""

    @abc.abstractmethod
    async def get_artifact(self, artifact_id: str) -> _t.Any:
        """Get artifact by ID."""
        ...

    @abc.abstractmethod
    async def save_artifact(self, content: _t.Any, metadata: dict) -> str:
        """Save artifact and return ID."""
        ...


class InMemoryArtifactService(ArtifactService):
    """In-memory implementation of ArtifactService."""

    def __init__(self):
        self._artifacts: dict[str, _t.Any] = {}
        self._metadata: dict[str, dict] = {}
        self._counter = 0

    async def get_artifact(self, artifact_id: str) -> _t.Any:
        """Get artifact by ID."""
        return self._artifacts.get(artifact_id)

    async def save_artifact(self, content: _t.Any, metadata: dict) -> str:
        """Save artifact and return ID."""
        self._counter += 1
        artifact_id = f'artifact_{self._counter}'

        self._artifacts[artifact_id] = content
        self._metadata[artifact_id] = {**metadata, 'created_at': datetime.now().isoformat(), 'id': artifact_id}

        return artifact_id

    async def get_artifact_metadata(self, artifact_id: str) -> dict:
        """Get artifact metadata."""
        return self._metadata.get(artifact_id, {})
