from __future__ import annotations

import asyncio
from typing import List

from .runner import AgentRunner

__all__ = ["ConcurrentRunner"]


class ConcurrentRunner:
    """Manage a group of :class:`AgentRunner` instances."""

    def __init__(self) -> None:
        self._runners: List[AgentRunner] = []

    @property
    def runners(self) -> List[AgentRunner]:
        """Return the managed runners."""
        return self._runners

    def add_runner(self, runner: AgentRunner) -> None:
        """Add a runner to the group."""
        self._runners.append(runner)

    async def start_all(self) -> None:
        """Start all runners concurrently."""
        await asyncio.gather(*(r.start() for r in self._runners))

    async def stop_all(self) -> None:
        """Stop all runners concurrently."""
        await asyncio.gather(*(r.stop() for r in self._runners))
