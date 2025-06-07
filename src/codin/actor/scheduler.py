"""Actor scheduler for agent lifecycle management.

This module provides actor scheduling and management services for controlling
agent lifecycles, including creation, activation, deactivation, and cleanup
of agent instances in the codin framework.
"""

import typing as _t

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field


if _t.TYPE_CHECKING:
    from ..agent.base import Agent

__all__ = ['ActorInfo', 'ActorScheduler', 'LocalActorManager']


class ActorInfo(BaseModel):
    """Information about a managed actor/agent."""

    actor_id: str
    actor_type: str
    agent: 'Agent'
    metadata: dict[str, _t.Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class ActorScheduler(ABC):
    """Abstract actor manager protocol from design document."""

    @abstractmethod
    async def get_or_create(self, actor_type: str, key: str) -> 'Agent':
        """Get or create an actor/agent instance."""

    @abstractmethod
    async def deactivate(self, agent_id: str) -> None:
        """Deactivate an actor/agent instance."""

    @abstractmethod
    async def list_actors(self) -> list[ActorInfo]:
        """List all active actors."""

    @abstractmethod
    async def get_actor_info(self, agent_id: str) -> ActorInfo | None:
        """Get information about a specific actor."""


class LocalActorManager(ActorScheduler):
    """Local implementation of actor manager using in-memory storage."""

    def __init__(self, agent_factory: _t.Callable[[str, str], _t.Awaitable['Agent']] | None = None):
        """Initialize with optional agent factory function.

        Args:
            agent_factory: Async function that takes (actor_type, key) and returns Agent instance
        """
        self._actors: dict[str, ActorInfo] = {}
        self._agent_factory = agent_factory

    async def get_or_create(self, actor_type: str, key: str) -> 'Agent':
        """Get or create an actor/agent instance."""
        actor_id = f'{actor_type}:{key}'

        if actor_id in self._actors:
            # Update last accessed time
            self._actors[actor_id].last_accessed = datetime.now()
            return self._actors[actor_id].agent

        # Create new agent
        if self._agent_factory:
            agent = await self._agent_factory(actor_type, key)
        else:
            # Default agent creation - import here to avoid circular import
            from ..agent.base_agent import BaseAgent
            from ..agent.base_planner import BasePlanner

            planner = BasePlanner()
            agent = BaseAgent(name=f'{actor_type}-{key}', agent_id=actor_id, planner=planner)

        # Store actor info
        actor_info = ActorInfo(actor_id=actor_id, actor_type=actor_type, agent=agent)
        self._actors[actor_id] = actor_info

        return agent

    async def deactivate(self, agent_id: str) -> None:
        """Deactivate an actor/agent instance."""
        if agent_id in self._actors:
            # Perform cleanup if agent has cleanup method
            agent = self._actors[agent_id].agent
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()

            # Remove from active actors
            del self._actors[agent_id]

    async def list_actors(self) -> list[ActorInfo]:
        """List all active actors."""
        return list(self._actors.values())

    async def get_actor_info(self, agent_id: str) -> ActorInfo | None:
        """Get information about a specific actor."""
        return self._actors.get(agent_id)

    async def cleanup_idle_actors(self, max_idle_time: float = 3600.0) -> int:
        """Clean up actors that have been idle for too long.

        Args:
            max_idle_time: Maximum idle time in seconds before cleanup

        Returns:
            Number of actors cleaned up
        """
        current_time = datetime.now()
        idle_actors = []

        for actor_id, actor_info in self._actors.items():
            idle_seconds = (current_time - actor_info.last_accessed).total_seconds()
            if idle_seconds > max_idle_time:
                idle_actors.append(actor_id)

        # Clean up idle actors
        for actor_id in idle_actors:
            await self.deactivate(actor_id)

        return len(idle_actors)
