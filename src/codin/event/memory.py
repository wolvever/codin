"""In-memory event hub implementation."""

import asyncio
import typing as _t
from .event_hub import EventHub

class InMemoryEventHub(EventHub):
    """In-memory implementation of EventHub using asyncio queues.
    
    This implementation provides high-performance event streaming using Python's
    built-in asyncio queues. Events are stored in memory and will be lost when
    the process terminates.
    """
    
    def __init__(self):
        """Initialize the in-memory event hub.
        
        Creates a dictionary to store task queues and a lock for thread safety.
        """
        self._queues: dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
        self._is_up = True
        
    async def up(self) -> None:
        """No-op since this is an in-memory implementation."""
        self._is_up = True
        
    async def down(self) -> None:
        """Clean up all queues and set _is_up to False."""
        async with self._lock:
            self._queues.clear()
        self._is_up = False
        
    async def _ensure_queue(self, task_id: str) -> asyncio.Queue:
        """Ensure a queue exists for the given task ID.
        
        Args:
            task_id: The ID of the task to ensure queue for
            
        Returns:
            The queue for the task
        """
        async with self._lock:
            if task_id not in self._queues:
                self._queues[task_id] = asyncio.Queue()
            return self._queues[task_id]
            
    async def publish(self, task_id: str, data: dict) -> None:
        """Publish data to a task's event stream.
        
        Args:
            task_id: The ID of the task to publish to
            data: The data to publish
        """
        queue = await self._ensure_queue(task_id)
        await queue.put(data)
        
    async def subscribe(self, task_id: str) -> _t.AsyncIterator[dict]:
        """Subscribe to a task's event stream.
        
        Args:
            task_id: The ID of the task to subscribe to
            
        Returns:
            An async iterator yielding published data
        """
        queue = await self._ensure_queue(task_id)
        while True:
            data = await queue.get()
            yield data

    async def recycle(self, task_id: str) -> None:
        """Recycle a task's queue.
        
        This method allows recycling a queue, which can be
        used to trigger cleanup of resources associated with that task.

        Args:
            task_id: The ID of the task whose queue should be marked for cleanup
        """
        async with self._lock:
            if task_id in self._queues:
                del self._queues[task_id] 