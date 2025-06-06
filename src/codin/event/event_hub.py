"""Event hub abstraction for codin agents.

This module defines the abstract EventHub interface for publishing and subscribing
to task events, enabling decoupled communication between system components.
"""

import typing as _t

from abc import ABC, abstractmethod

from codin.lifecycle import LifecycleMixin


class EventHub(ABC, LifecycleMixin):
    """Abstract base class for event hub implementations."""

    @abstractmethod
    async def publish(self, task_id: str, data: dict) -> None:
        """Publish data to a task's event stream.

        Args:
            task_id: The ID of the task to publish to
            data: The data to publish
        """

    @abstractmethod
    async def subscribe(self, task_id: str) -> _t.AsyncIterator[dict]:
        """Subscribe to a task's event stream.

        Args:
            task_id: The ID of the task to subscribe to

        Returns:
            An async iterator yielding published data
        """

    @abstractmethod
    async def recycle(self, task_id: str) -> None:
        """Recycle a task's queue.

        This method allows recycling a queue, which can be
        used to trigger cleanup of resources associated with that task.

        Args:
            task_id: The ID of the task whose queue should be marked for cleanup
        """
