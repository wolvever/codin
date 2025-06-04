"""Session service implementations."""

import typing as _t
from datetime import datetime

# Import TaskInfo and TaskStatus from agent types
from ..agent.types import TaskInfo, TaskStatus

__all__ = [
    "SessionService",
    "ReplayService",
    "TaskService",
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


class ReplayService:
    """Service for recording execution replay logs."""
    
    def __init__(self):
        self._replay_logs: dict[str, list[dict]] = {}
    
    async def record_step(self, session_id: str, step: _t.Any, result: _t.Any) -> None:
        """Record step execution for replay."""
        if session_id not in self._replay_logs:
            self._replay_logs[session_id] = []
        
        self._replay_logs[session_id].append({
            "timestamp": datetime.now().isoformat(),
            "step_id": getattr(step, 'step_id', 'unknown'),
            "step_type": getattr(step, 'step_type', 'unknown'),
            "step_data": self._serialize_step(step),
            "result": self._serialize_result(result)
        })
    
    async def get_replay_log(self, session_id: str) -> list[dict]:
        """Get replay log for session."""
        return self._replay_logs.get(session_id, [])
    
    def _serialize_step(self, step: _t.Any) -> dict:
        """Serialize step for logging."""
        # Basic serialization - could be enhanced
        return {
            "type": type(step).__name__,
            "data": str(step)
        }
    
    def _serialize_result(self, result: _t.Any) -> dict:
        """Serialize result for logging."""
        # Basic serialization - could be enhanced
        return {
            "type": type(result).__name__,
            "data": str(result)
        }


class TaskService:
    """Service for managing task lifecycle."""
    
    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._counter = 0
    
    async def create_task(
        self, 
        query: str, 
        parent_id: str | None = None,
        metadata: dict | None = None
    ) -> TaskInfo:
        """Create a new task."""
        self._counter += 1
        task_id = f"task_{self._counter}"
        
        task = TaskInfo(
            id=task_id,
            parent_id=parent_id,
            query=query,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        self._tasks[task_id] = task
        return task
    
    async def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task by ID."""
        return self._tasks.get(task_id)
    
    async def update_task_status(
        self, 
        task_id: str, 
        status: TaskStatus,
        metadata: dict | None = None
    ) -> bool:
        """Update task status."""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = status
            
            if status == TaskStatus.RUNNING and task.started_at is None:
                task.started_at = datetime.now()
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                task.completed_at = datetime.now()
            
            if metadata:
                task.metadata.update(metadata)
            
            return True
        return False
    
    async def start_task(self, task_id: str) -> bool:
        """Start task execution."""
        return await self.update_task_status(task_id, TaskStatus.RUNNING)
    
    async def pause_task(self, task_id: str) -> bool:
        """Pause task execution."""
        return await self.update_task_status(task_id, TaskStatus.PAUSED)
    
    async def resume_task(self, task_id: str) -> bool:
        """Resume paused task."""
        return await self.update_task_status(task_id, TaskStatus.RUNNING)
    
    async def complete_task(self, task_id: str, metadata: dict | None = None) -> bool:
        """Mark task as completed."""
        return await self.update_task_status(task_id, TaskStatus.COMPLETED, metadata)
    
    async def cancel_task(self, task_id: str, metadata: dict | None = None) -> bool:
        """Cancel task execution."""
        return await self.update_task_status(task_id, TaskStatus.CANCELLED, metadata)
    
    async def fail_task(self, task_id: str, error: str, metadata: dict | None = None) -> bool:
        """Mark task as failed."""
        fail_metadata = {"error": error}
        if metadata:
            fail_metadata.update(metadata)
        return await self.update_task_status(task_id, TaskStatus.FAILED, fail_metadata) 