"""Ray-based actor scheduler for distributed agent execution."""

from __future__ import annotations

import typing as _t
import asyncio
from datetime import datetime

from .supervisor import ActorInfo, ActorSupervisor

if _t.TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from ..agent.base import Agent


try:  # pragma: no cover - optional dependency
    import ray
except Exception:  # pragma: no cover
    ray = None


__all__ = ["RayActorManager"]

# When running under Ray we store ``ActorHandle`` objects in ``ActorInfo``. The
# default annotation for the ``agent`` field expects an ``Agent`` instance which
# causes validation errors. Relax the annotation so Pydantic accepts the handle
# without attempting to validate it.
if ray is not None:
    ActorInfo.__annotations__["agent"] = _t.Any
    if "agent" in ActorInfo.__pydantic_fields__:
        ActorInfo.__pydantic_fields__["agent"].annotation = _t.Any
    ActorInfo.model_rebuild(force=True)


class _RayAgentWrapper:
    """Ray remote wrapper executing an Agent."""

    def __init__(self, agent: Agent) -> None:  # pragma: no cover - executed on ray worker
        self._agent = agent

    def run(self, input_data: dict) -> list[dict]:  # pragma: no cover - executed on ray worker
        import asyncio
        from ..agent.types import AgentRunInput, State, Message
        from ..tool.base import Tool
        from ..artifact.base import ArtifactService

        # Ensure Pydantic models are fully defined on the worker
        State.model_rebuild(
            force=True,
            _types_namespace={"Tool": Tool, "ArtifactService": ArtifactService, "Message": Message},
        )

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
            coro = self._agent.cleanup()
            if asyncio.iscoroutine(coro):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(coro)
                else:
                    loop.run_until_complete(coro)


if ray:  # pragma: no cover - avoid ray usage when not available
    RayAgentActor = ray.remote(_RayAgentWrapper)
else:  # pragma: no cover
    RayAgentActor = None


class RayActorManager(ActorSupervisor):
    """Manage agents as Ray actors."""

    def __init__(self, agent_factory: _t.Callable[..., _t.Awaitable[Agent]] | None = None) -> None:
        if ray is None:
            raise ImportError("ray is required for RayActorManager")
        self._actors: dict[str, ActorInfo] = {}
        self._factory = agent_factory

    async def acquire(self, actor_type: str, key: str, *args: _t.Any, **kwargs: _t.Any) -> ray.actor.ActorHandle:
        actor_id = f"{actor_type}:{key}"
        if actor_id in self._actors:
            info = self._actors[actor_id]
            info.last_accessed = datetime.now()
            return info.agent

        if self._factory:
            agent = await self._factory(actor_type, key, *args, **kwargs)
        else:
            from ..agent.base_agent import BaseAgent
            from ..agent.base_planner import BasePlanner

            planner = BasePlanner()
            agent = BaseAgent(agent_id=actor_id, name=f"{actor_type}-{key}", planner=planner)

        handle = RayAgentActor.remote(agent)
        info = ActorInfo(actor_id=actor_id, actor_type=actor_type, agent=handle)
        self._actors[actor_id] = info
        return handle

    async def release(self, agent_id: str) -> None:
        info = self._actors.pop(agent_id, None)
        if info is None:
            return
        try:
            # ``cleanup`` is a synchronous actor method so run it in a thread to
            # avoid blocking the event loop.
            await asyncio.to_thread(lambda: ray.get(info.agent.cleanup.remote()))
        finally:  # pragma: no cover - best effort
            ray.kill(info.agent)

    async def list(self) -> list[ActorInfo]:
        return list(self._actors.values())

    async def info(self, agent_id: str) -> ActorInfo | None:
        return self._actors.get(agent_id)
