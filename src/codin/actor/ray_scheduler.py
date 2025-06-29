from __future__ import annotations

import builtins

__all__ = ["RayActorManager"]


class RayActorManager:
    """Minimal stand-in for the real Ray actor manager used in tests."""

    def __init__(self) -> None:
        self._actors: dict[str, dict] = {}

    async def acquire(self, name: str, version: str) -> dict:
        actor_id = f"{name}:{version}"
        self._actors[actor_id] = {"id": actor_id}
        return self._actors[actor_id]

    async def release(self, actor_id: str) -> None:
        self._actors.pop(actor_id, None)

    async def list(self) -> builtins.list[dict]:
        return list(self._actors.values())

    async def info(self, actor_id: str) -> dict | None:
        return self._actors.get(actor_id)
