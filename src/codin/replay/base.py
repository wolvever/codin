"""Replay service for recording execution replay logs.

This module provides replay functionality for recording and analyzing
agent execution steps, enabling debugging and performance analysis.
"""

from abc import ABC, abstractmethod
from typing import Any

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
        if isinstance(message, (str, int, float, bool, list, dict)) or message is None:
            return message
        return str(message) # Fallback to string representation
