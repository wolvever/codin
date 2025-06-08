"""Actor system with simple work stealing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ..lifecycle import LifecycleMixin
from .dispatcher import Dispatcher
from .supervisor import ActorSupervisor

__all__ = ["WorkStealingActorSystem"]


class WorkStealingActorSystem(LifecycleMixin):
    """Concurrent actor system using per-worker queues with stealing."""

    def __init__(
        self,
        dispatcher: Dispatcher,
        actor_manager: ActorSupervisor,
        *,
        workers: int = 4,
        choose_worker: Callable[[dict[str, Any]], int] | None = None,
    ) -> None:
        super().__init__()
        self.dispatcher = dispatcher
        self.actor_manager = actor_manager
        self.workers = workers
        self.choose_worker = choose_worker
        self._queues: list[asyncio.Queue] = [asyncio.Queue() for _ in range(workers)]
        self._tasks: list[asyncio.Task] = []
        self._rr_counter = 0

    async def submit(self, request: dict[str, Any]) -> asyncio.Future:
        """Enqueue an A2A request and return a future with the runner id."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        idx = (
            self.choose_worker(request)
            if self.choose_worker is not None
            else self._rr_counter % self.workers
        )
        self._rr_counter = (self._rr_counter + 1) % self.workers
        await self._queues[idx].put({"request": request, "future": fut})
        return fut

    async def _get_item(self, my_idx: int) -> dict[str, Any] | None:
        my_queue = self._queues[my_idx]
        try:
            return my_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        for idx, q in enumerate(self._queues):
            if idx == my_idx:
                continue
            try:
                return q.get_nowait()
            except asyncio.QueueEmpty:
                continue
        return await my_queue.get()

    async def _worker(self, idx: int) -> None:
        while True:
            item = await self._get_item(idx)
            if item is None:
                break
            request = item["request"]
            fut: asyncio.Future = item["future"]
            try:
                runner_id = await self.dispatcher.submit(request)
            except Exception as exc:  # pragma: no cover - best effort
                fut.set_exception(exc)
            else:
                fut.set_result(runner_id)

    async def _up(self) -> None:
        for i in range(self.workers):
            self._tasks.append(asyncio.create_task(self._worker(i)))

    async def _down(self) -> None:
        for q in self._queues:
            await q.put(None)
        for task in self._tasks:
            try:
                await task
            except Exception:  # pragma: no cover - best effort
                pass
        self._tasks.clear()
        for q in self._queues:
            while not q.empty():
                q.get_nowait()
                q.task_done()
