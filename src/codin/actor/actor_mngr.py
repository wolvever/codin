"""Actor Manager for agent lifecycle management."""

import typing as _t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

if _t.TYPE_CHECKING:
    from ..agent.base import Agent

__all__ = ["ActorManager", "LocalActorManager", "ActorInfo"]


@dataclass
class ActorInfo:
    """Information about a managed actor/agent."""
    actor_id: str
    actor_type: str
    key: str
    agent: "Agent"
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: dict[str, _t.Any] = field(default_factory=dict)


class ActorManager(ABC):
    """Abstract actor manager protocol from design document."""
    
    @abstractmethod
    async def get_or_create(self, actor_type: str, key: str) -> "Agent":
        """Get or create an actor/agent instance."""
        pass
    
    @abstractmethod
    async def deactivate(self, agent_id: str) -> None:
        """Deactivate an actor/agent instance."""
        pass
    
    @abstractmethod
    async def list_actors(self) -> list[ActorInfo]:
        """List all active actors."""
        pass
    
    @abstractmethod
    async def get_actor_info(self, agent_id: str) -> ActorInfo | None:
        """Get information about a specific actor."""
        pass


class LocalActorManager(ActorManager):
    """Local implementation of actor manager using in-memory storage."""
    
    def __init__(self, agent_factory: _t.Callable[[str, str], _t.Awaitable["Agent"]] | None = None):
        """Initialize with optional agent factory function.
        
        Args:
            agent_factory: Async function that takes (actor_type, key) and returns Agent instance
        """
        self._actors: dict[str, ActorInfo] = {}
        self._agent_factory = agent_factory
        
    async def get_or_create(self, actor_type: str, key: str) -> "Agent":
        """Get or create an actor/agent instance."""
        actor_id = f"{actor_type}:{key}"
        
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
            from ..agent.code_planner import CodePlanner
            
            planner = CodePlanner()
            agent = BaseAgent(
                name=f"{actor_type}-{key}",
                agent_id=actor_id,
                planner=planner
            )
        
        # Store actor info
        actor_info = ActorInfo(
            actor_id=actor_id,
            actor_type=actor_type,
            key=key,
            agent=agent
        )
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
