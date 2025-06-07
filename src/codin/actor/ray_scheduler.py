"""Ray-based actor scheduler for distributed agent execution."""

from __future__ import annotations

import typing as _t
from datetime import datetime

from pydantic import Field

from .scheduler import ActorInfo, ActorScheduler

if _t.TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from ..agent.base import Agent
    from ..agent.types import AgentRunInput


try:  # pragma: no cover - optional dependency
    import ray
except Exception:  # pragma: no cover
    ray = None


__all__ = ["RayActorManager"]


class _RayAgentWrapper:
    """Ray remote wrapper executing an Agent."""

    def __init__(self, agent: "Agent") -> None:  # pragma: no cover - executed on ray worker
        self._agent = agent

    def run(self, input_data: dict) -> list[dict]:  # pragma: no cover - executed on ray worker
        import asyncio
        from ..agent.types import AgentRunInput

        data = AgentRunInput(**input_data)
        outputs: list[dict] = []

        async def _collect() -> None:
            async for out in self._agent.run(data):
                outputs.append(out.dict())

        asyncio.run(_collect())
        return outputs

    def cleanup(self) -> None:  # pragma: no cover - executed on ray worker
        import asyncio

        if hasattr(self._agent, "cleanup"):
            asyncio.run(self._agent.cleanup())


if ray:  # pragma: no cover - avoid ray usage when not available
    RayAgentActor = ray.remote(_RayAgentWrapper)
else:  # pragma: no cover
    RayAgentActor = None


class RayActorManager(ActorScheduler):
    """Manage agents as Ray actors."""

    def __init__(self, agent_factory: _t.Callable[[str, str], _t.Awaitable["Agent"]] | None = None) -> None:
        if ray is None:
            raise ImportError("ray is required for RayActorManager")
        self._actors: dict[str, ActorInfo] = {}
        self._factory = agent_factory

    async def get_or_create(self, actor_type: str, key: str) -> "ray.actor.ActorHandle":
        actor_id = f"{actor_type}:{key}"
        if actor_id in self._actors:
            info = self._actors[actor_id]
            info.last_accessed = datetime.now()
            return info.agent

        if self._factory:
            agent = await self._factory(actor_type, key)
        else:
            from ..agent.code_planner import CodePlanner
            from ..agent.base_agent import BaseAgent

            planner = CodePlanner()
            agent = BaseAgent(agent_id=actor_id, name=f"{actor_type}-{key}", planner=planner)

        handle = RayAgentActor.remote(agent)
        info = ActorInfo(actor_id=actor_id, actor_type=actor_type, agent=handle)
        self._actors[actor_id] = info
        return handle

    async def deactivate(self, agent_id: str) -> None:
        info = self._actors.pop(agent_id, None)
        if info is None:
            return
        try:
            await info.agent.cleanup.remote()
        finally:  # pragma: no cover - best effort
            ray.kill(info.agent)

    async def list_actors(self) -> list[ActorInfo]:
        return list(self._actors.values())

    async def get_actor_info(self, agent_id: str) -> ActorInfo | None:
        return self._actors.get(agent_id)
