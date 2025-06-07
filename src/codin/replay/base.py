"""Replay service for recording execution replay logs.

This module provides replay functionality for recording and analyzing
agent execution steps, enabling debugging and performance analysis.
"""

import typing as _t

from datetime import datetime


class ReplayService:
    """Service for recording execution replay logs."""

    def __init__(self):
        self._replay_logs: dict[str, list[dict]] = {}

    async def record_step(self, session_id: str, step: _t.Any, result: _t.Any) -> None:
        """Record step execution for replay."""
        if session_id not in self._replay_logs:
            self._replay_logs[session_id] = []

        self._replay_logs[session_id].append(
            {
                'timestamp': datetime.now().isoformat(),
                'step_id': getattr(step, 'step_id', 'unknown'),
                'step_type': getattr(step, 'step_type', 'unknown'),
                'step_data': self._serialize_step(step),
                'result': self._serialize_result(result),
            }
        )

    async def get_replay_log(self, session_id: str) -> list[dict]:
        """Get replay log for session."""
        return self._replay_logs.get(session_id, [])

    def _serialize_step(self, step: _t.Any) -> dict:
        """Serialize step for logging."""
        # Basic serialization - could be enhanced
        return {'type': type(step).__name__, 'data': str(step)}

    def _serialize_result(self, result: _t.Any) -> dict:
        """Serialize result for logging."""
        # Basic serialization - could be enhanced
        return {'type': type(result).__name__, 'data': str(result)}
