"""Redis-backed event hub implementation."""

import json
import typing as _t
import redis.asyncio as aioredis
from .event_hub import EventHub

class RedisEventHub(EventHub):
    """Redis-backed implementation of EventHub.
    
    This implementation uses Redis streams for high-performance event distribution.
    Events are stored in Redis and will persist even if the process terminates.
    """
    
    def __init__(self, stream_ns="a2a_stream", redis_url="redis://localhost:6379"):
        """Initialize the Redis event hub.
        
        Args:
            stream_ns: Namespace prefix for Redis stream keys
            redis_url: URL to connect to Redis server
        """
        self.redis = None
        self.ns = stream_ns
        self.redis_url = redis_url
        self._is_up = False

    async def up(self) -> None:
        """Set up Redis connection.
        
        Establishes connection to Redis server using the configured URL.
        """
        self.redis = aioredis.from_url(self.redis_url)
        self._is_up = True

    async def down(self) -> None:
        """Clean up Redis connection.

        Closes the Redis connection and sets the _is_up flag to False.
        """
        if self.redis:
            await self.redis.close()
        self._is_up = False

    async def publish(self, task_id: str, data: dict) -> None:
        """Publish data to a task's event stream.
        
        Args:
            task_id: The ID of the task to publish to
            data: The data to publish
        """
        if not self._is_up:
            raise RuntimeError("RedisEventHub not started. Call up() first.")
            
        entry = json.dumps(data)
        await self.redis.xadd(f"{self.ns}:{task_id}", {"data": entry})

    async def subscribe(self, task_id: str) -> _t.AsyncIterator[dict]:
        """Subscribe to a task's event stream.
        
        Args:
            task_id: The ID of the task to subscribe to
            
        Returns:
            An async iterator yielding published data
        """
        if not self._is_up:
            raise RuntimeError("RedisEventHub not started. Call up() first.")
            
        key = f"{self.ns}:{task_id}"
        last_id = "0-0"
        while True:
            resp = await self.redis.xread({key: last_id}, block=0)
            for _, messages in resp:
                for msg_id, fields in messages:
                    payload = json.loads(fields[b"data"])
                    last_id = msg_id.decode()
                    yield payload

    async def recycle(self, task_id: str) -> None:
        """Recycle a task's queue.
        
        This method allows recycling a queue, which can be
        used to trigger cleanup of resources associated with that task.

        Args:
            task_id: The ID of the task whose queue should be marked for cleanup
        """
        if not self._is_up:
            raise RuntimeError("RedisEventHub not started. Call up() first.")
            
        await self.redis.xdel(f"{self.ns}:{task_id}", "*") 