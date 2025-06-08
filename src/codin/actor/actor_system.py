from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from ..lifecycle import LifecycleMixin
from .dispatcher import (
    Dispatcher,  # for typing
)
from .supervisor import ActorSupervisor

__all__ = ["ActorSystem"]


class ActorSystem(LifecycleMixin):
    """Simple concurrent actor system handling queued requests."""

    def __init__(
        self,
        dispatcher: Dispatcher,
        actor_manager: ActorSupervisor,
        *,
        workers: int = 4,
    ) -> None:
        super().__init__()
        self.dispatcher = dispatcher
        self.actor_manager = actor_manager
        self.workers = workers
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

    async def submit(self, request: dict[str, Any]) -> asyncio.Future:
        """Enqueue an A2A request and return a future with the runner id."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        await self._queue.put({"request": request, "future": fut})
        return fut

    async def _worker(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            request = item["request"]
            fut: asyncio.Future = item["future"]
            try:
                runner_id = await self.dispatcher.submit(request)
            except Exception as exc:
                fut.set_exception(exc)
            else:
                fut.set_result(runner_id)

    async def _up(self) -> None:
        for _ in range(self.workers):
            self._tasks.append(asyncio.create_task(self._worker()))

    async def _down(self) -> None:
        for _ in self._tasks:
            await self._queue.put(None)
        for task in self._tasks:
            with contextlib.suppress(Exception):
                await task
        self._tasks.clear()
        # flush queue
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
