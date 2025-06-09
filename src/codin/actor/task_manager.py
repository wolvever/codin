"""Manages information and state of tasks processed by actors.

This module provides `TaskInfo` to store details about a task and
`TaskRegistry` to manage a collection of tasks, allowing for state
updates and tracking.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Optional, Dict

from pydantic import BaseModel, Field

from .envelope_types import TaskState, EnvelopeKind


class TaskInfo(BaseModel):
    """Represents the state and metadata of a task being processed by an actor.

    Attributes:
        task_id: Unique identifier for the task.
        runner_id: Identifier for the dispatcher's execution run handling this task.
        actor_id: ID of the actor instance assigned to or processing the task.
                  Can be empty initially until an actor is assigned.
        current_state: The current `TaskState` of the task.
        envelope_kind: The `EnvelopeKind` that initiated this task.
        reply_to: Optional reply address from the original envelope headers.
        submitted_at: Timestamp when the task was first submitted.
        last_updated_at: Timestamp when the task's state or info was last updated.
        error_info: Optional string containing error details if the task failed.
    """
    task_id: str
    runner_id: str
    actor_id: str = "" # May be updated after actor assignment
    current_state: TaskState = TaskState.SUBMITTED
    envelope_kind: EnvelopeKind
    reply_to: Optional[str] = None
    submitted_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    last_updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    error_info: Optional[str] = None

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True # Enables auto-update of last_updated_at via Pydantic if setters were used.
                                   # For manual update as in TaskRegistry, this isn't strictly needed for that effect.


class TaskRegistry:
    """A thread-safe (async-safe) registry for managing `TaskInfo` objects.

    Provides methods to add, retrieve, update, and remove task information.
    Uses an `asyncio.Lock` to ensure atomic operations on the internal task store.
    """

    def __init__(self):
        """Initializes the TaskRegistry with an empty task store and an asyncio Lock."""
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()

    async def add_task(self, task_info: TaskInfo) -> None:
        """Adds or updates a task in the registry.

        If a task with the same `task_id` already exists, it will be overwritten.
        The operation is made atomic using an asyncio lock.

        Args:
            task_info: The `TaskInfo` object to add or update.
        """
        async with self._lock:
            task_info.last_updated_at = datetime.datetime.now() # Ensure update time on any add/overwrite
            self._tasks[task_info.task_id] = task_info

    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Retrieves a task from the registry by its ID.

        Args:
            task_id: The ID of the task to retrieve.

        Returns:
            The `TaskInfo` object if found, otherwise `None`.
        """
        async with self._lock:
            return self._tasks.get(task_id)

    async def update_task_state(
        self, task_id: str, new_state: TaskState, error_info: Optional[str] = None
    ) -> bool:
        """Updates the state of an existing task.

        If the task is found, its `current_state` and `last_updated_at` are updated.
        If `error_info` is provided, it's also updated on the task.

        Args:
            task_id: The ID of the task to update.
            new_state: The new `TaskState` for the task.
            error_info: Optional error message if the task is moving to a FAILED state.

        Returns:
            `True` if the task was found and updated, `False` otherwise.
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.current_state = new_state
                task.last_updated_at = datetime.datetime.now()
                if error_info is not None: # Allow clearing error_info if new state isn't FAILED
                    task.error_info = error_info
                elif new_state != TaskState.FAILED: # Clear error if not failing
                    task.error_info = None
                return True
            return False

    async def remove_task(self, task_id: str) -> bool:
        """Removes a task from the registry.

        Args:
            task_id: The ID of the task to remove.

        Returns:
            `True` if the task was found and removed, `False` otherwise.
        """
        async with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
            return False

    async def list_all_tasks(self) -> list[TaskInfo]:
        """Lists all tasks currently in the registry.

        Returns:
            A list of `TaskInfo` objects.
        """
        async with self._lock:
            return list(self._tasks.values())


__all__ = [
    "TaskInfo",
    "TaskRegistry",
]
