"""CoDIN Agent-to-Agent (A2A) / Actor-to-Actor Communication Package.

This package contains types and utilities related to structured task definitions
and message formats for inter-actor communication, particularly for A2A_TASK
envelope kinds.
"""

from .types import A2AParam, A2AResult, A2ATaskPayload

__all__ = [
    "A2AParam",
    "A2AResult",
    "A2ATaskPayload",
]
