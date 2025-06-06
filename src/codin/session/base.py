"""Session service implementations."""

import asyncio
import typing as _t
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict

from ..agent.types import State


__all__ = [
    "SessionService",
    "ReplayService",
    "Session",
    "SessionManager",
]


class SessionService:
    """Service for managing agent sessions."""
    
    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._counter = 0
    
    async def get_or_create_session(self, session_id: str | None) -> dict:
        """Get existing or create new session."""
        if session_id is None:
            self._counter += 1
            session_id = f"session_{self._counter}"
        
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "session_id": session_id,
                "created_at": datetime.now(),
                "iteration_count": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "elapsed_time": 0.0,
                "metadata": {}
            }
        
        return self._sessions[session_id]
    
    async def update_session(self, session_id: str, updates: dict) -> None:
        """Update session data."""
        if session_id in self._sessions:
            self._sessions[session_id].update(updates)
    
    async def get_session(self, session_id: str) -> dict | None:
        """Get session by ID."""
        return self._sessions.get(session_id)


# =============================================================================
# Data-oriented Session classes (merged from agent/session.py)
# =============================================================================

# Import needed for the merged classes
from a2a.types import Message
from ..memory.base import MemoryService, MemMemoryService
from codin.replay.base import ReplayService


class Session(BaseModel):
    """Data-oriented session that holds conversation state and manages recording."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Core conversation data
    messages: list[dict] = Field(default_factory=list)  # Changed from Message to dict for now
    turn_count: int = 0
    task_list: dict[str, list[str]] = Field(default_factory=lambda: {"completed": [], "pending": []})
    
    # Execution metrics
    metrics: dict[str, _t.Any] = Field(default_factory=dict)
    context: dict[str, _t.Any] = Field(default_factory=dict)
    
    # Optional external systems
    memory_system: _t.Any | None = None  # Changed to Any to avoid import issues
    rollout_recorder: _t.Any | None = None  # For audit trail - type depends on implementation
    
    def __post_init__(self):
        """Initialize default metrics."""
        if not self.metrics:
            self.metrics = {
                "start_time": self.created_at.timestamp(),
                "total_tool_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "last_activity": self.created_at.timestamp()
            }
    
    async def record(self, message: dict) -> None:  # Changed from Message to dict
        """Record a message to both internal state and external systems (Codex-inspired)."""
        # Update internal state
        self.messages.append(message)
        self.metrics["last_activity"] = datetime.now().timestamp()
        
        # Record to memory system if available
        if self.memory_system:
            try:
                await self.memory_system.add_message(message)
            except Exception as e:
                # Log but don't fail - memory is optional
                import logging
                logging.warning(f"Failed to record to memory system: {e}")
        
        # Record to rollout recorder if available (for complete audit trail)
        if self.rollout_recorder and hasattr(self.rollout_recorder, 'record'):
            try:
                await self.rollout_recorder.record(message)
            except Exception as e:
                import logging
                logging.warning(f"Failed to record to rollout recorder: {e}")
    
    def build_state(self) -> "State":
        """Build a State object for the planner from current session data."""
        # Import here at runtime to avoid circular imports
        from ..agent.types import State
        return State(
            session_id=self.session_id,
            task_id=None,  # This needs to be set properly based on context
            agent_id="",
            created_at=self.created_at,
            iteration=self.turn_count,
            tools=[],  # These will be set by the agent
            tool_call_results=[],
            context=self.context.copy(),
            metadata={"task_list": self.task_list.copy(), **self.context}
        )
    
    def update_from_state(self, state: "State") -> None:
        """Update session data from a State object after planner execution."""
        self.turn_count = state.iteration
        if "task_list" in state.metadata:
            self.task_list = state.metadata["task_list"]
        self.context.update(state.context)
    
    def get_metrics_summary(self) -> dict[str, _t.Any]:
        """Get a summary of session metrics."""
        current_time = datetime.now().timestamp()
        elapsed = current_time - self.metrics["start_time"]
        
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "message_count": len(self.messages),
            "elapsed_seconds": elapsed,
            "total_tool_calls": self.metrics.get("total_tool_calls", 0),
            "input_tokens": self.metrics.get("input_tokens", 0),
            "output_tokens": self.metrics.get("output_tokens", 0),
            "cost": self.metrics.get("cost", 0.0),
            "last_activity": self.metrics.get("last_activity", self.created_at.timestamp())
        }


class SessionManager:
    """Manages active sessions with optional cleanup and persistence."""
    
    def __init__(self, memory_system_factory: _t.Callable[[], MemoryService] | None = None):
        self._sessions: dict[str, Session] = {}
        self._memory_system_factory = memory_system_factory or (lambda: MemMemoryService())
        self._cleanup_lock = asyncio.Lock()
    
    async def get_or_create_session(
        self, 
        session_id: str,
        memory_system: MemoryService | None = None
    ) -> Session:
        """Get existing session or create a new one."""
        if session_id not in self._sessions:
            # Create new session
            if memory_system is None:
                memory_system = self._memory_system_factory()
            
            session = Session(
                session_id=session_id,
                memory_system=memory_system
            )
            self._sessions[session_id] = session
        
        return self._sessions[session_id]
    
    def get_session(self, session_id: str) -> Session | None:
        """Get existing session by ID."""
        return self._sessions.get(session_id)
    
    async def close_session(self, session_id: str) -> None:
        """Close and cleanup a session."""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            
            # Cleanup memory system if it has cleanup method
            if session.memory_system and hasattr(session.memory_system, 'cleanup'):
                try:
                    await session.memory_system.cleanup()
                except Exception as e:
                    import logging
                    logging.warning(f"Error cleaning up memory system for session {session_id}: {e}")
            
            # Cleanup rollout recorder if it has cleanup method
            if session.rollout_recorder and hasattr(session.rollout_recorder, 'cleanup'):
                try:
                    await session.rollout_recorder.cleanup()
                except Exception as e:
                    import logging
                    logging.warning(f"Error cleaning up rollout recorder for session {session_id}: {e}")
            
            del self._sessions[session_id]
    
    async def cleanup_inactive_sessions(self, max_age_seconds: float = 3600) -> int:
        """Cleanup sessions that haven't been active for specified time."""
        async with self._cleanup_lock:
            current_time = datetime.now().timestamp()
            inactive_sessions = []
            
            for session_id, session in self._sessions.items():
                last_activity = session.metrics.get("last_activity", session.created_at.timestamp())
                if current_time - last_activity > max_age_seconds:
                    inactive_sessions.append(session_id)
            
            # Close inactive sessions
            for session_id in inactive_sessions:
                await self.close_session(session_id)
            
            return len(inactive_sessions)
    
    def get_active_sessions(self) -> dict[str, dict[str, _t.Any]]:
        """Get summary of all active sessions."""
        return {
            session_id: session.get_metrics_summary()
            for session_id, session in self._sessions.items()
        }
    
    async def cleanup(self) -> None:
        """Cleanup all sessions."""
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.close_session(session_id) 