"""Replay service for recording execution replay logs.

This module provides replay functionality for recording and analyzing
agent execution steps, enabling debugging and performance analysis.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseReplay(ABC):
    """Base class for replay services."""

    @abstractmethod
    async def record_message_exchange(self, client_message: Any, agent_message: Any, session_id: str, **kwargs) -> None:
        """Records a client message and the corresponding agent message for a session."""
        pass

    def _serialize_message(self, message: Any) -> Any:
        """Serializes a message. Can be overridden by subclasses if needed."""
        if hasattr(message, 'dict'):
            return message.dict()
        if isinstance(message, str | int | float | bool | list | dict) or message is None:
            return message
        return str(message) # Fallback to string representation


class ReplayService:
    """Simple in-memory replay service used for tests."""

    def __init__(self) -> None:
        self._log: Dict[str, List[Dict[str, Any]]] = {}

    async def record_step(self, session_id: str, step: Any, result: Any) -> None:
        """Record a step execution result for a session."""
        entries = self._log.setdefault(session_id, [])
        entries.append(
            {
                "step_id": getattr(step, "step_id", ""),
                "step_type": getattr(step, "step_type", ""),
                "step_data": {"type": type(step).__name__, "data": str(step)},
                "result": {"type": type(result).__name__, "data": str(result)},
            }
        )

    async def get_replay_log(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve recorded log for a session."""
        return list(self._log.get(session_id, []))
