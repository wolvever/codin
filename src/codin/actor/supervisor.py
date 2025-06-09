"""Actor supervisor for actor lifecycle management with capability awareness.

This module provides actor supervision and management services for controlling
lifecycles of `CallableActor` instances. Each actor's capabilities are
tracked via the `Capability` model, stored within `ActorInfo`.
"""

import asyncio # Added for _DefaultSupervisorActor's pause loop
import logging
import typing as _t
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator, Optional

from pydantic import BaseModel, Field

from .types import CallableActor, ActorRunInput, ActorRunOutput
from .envelope_types import Capability, EnvelopeKind, TaskState # Added TaskState for potential use in _DefaultSupervisorActor

logger = logging.getLogger(__name__)

__all__ = [
    'ActorInfo',
    'ActorSupervisor',
    'LocalActorManager',
]


class ActorInfo(BaseModel):
    """Information about a managed actor instance, including its capabilities."""
    actor_id: str
    actor_type: str
    agent: CallableActor
    capability: Capability
    metadata: dict[str, _t.Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class ActorSupervisor(ABC):
    """Abstract base class for an actor supervisor."""
    @abstractmethod
    async def acquire(self, actor_type: str, key: str, *args: _t.Any, **kwargs: _t.Any) -> CallableActor:
        """Acquires an actor instance."""
        ...
    @abstractmethod
    async def release(self, actor_id: str) -> None:
        """Releases an active actor instance."""
        ...
    @abstractmethod
    async def list(self) -> list[ActorInfo]:
        """Lists all actors currently managed by the supervisor."""
        ...
    @abstractmethod
    async def info(self, actor_id: str) -> Optional[ActorInfo]:
        """Retrieves information about a specific actor."""
        ...
    @abstractmethod
    async def get_actor_instance(self, actor_id: str) -> Optional[CallableActor]:
        """Retrieves an active actor instance by its ID."""
        ...


class _DefaultSupervisorActor(CallableActor):
    """A minimal default actor provided by the `LocalActorManager`."""
    capability: Capability = Capability(
        accepts={EnvelopeKind.PLAIN_TASK, EnvelopeKind.CONTROL},
        payload_schema=None
    )

    def __init__(self, actor_id: str, name: str):
        self.actor_id = actor_id
        self.name = name
        self._cancel_requested: bool = False
        self._paused: bool = False # Added for pause/resume
        logger.debug(f"_DefaultSupervisorActor '{self.name}' (ID: {self.actor_id}) initialized.")

    async def run(self, input_data: ActorRunInput) -> AsyncIterator[ActorRunOutput]:
        logger.debug(f"_DefaultSupervisorActor '{self.name}' (ID: {self.actor_id}) run started for request: {input_data.request_id}, kind: {input_data.envelope_kind}")

        # Check for initial cancellation
        if self._cancel_requested:
            logger.info(f"_DefaultSupervisorActor {self.actor_id} run cancelled before processing request {input_data.request_id}.")
            yield {"status": TaskState.CANCELED.value, "detail": "Processing cancelled by request before start."}
            return

        # Handle pause loop
        while self._paused and not self._cancel_requested:
            logger.debug(f"_DefaultSupervisorActor {self.actor_id} is paused. Checking again in 1s for request {input_data.request_id}.")
            await asyncio.sleep(1)

        if self._cancel_requested: # Re-check after potential pause
            logger.info(f"_DefaultSupervisorActor {self.actor_id} run cancelled after pause for request {input_data.request_id}.")
            yield {"status": TaskState.CANCELED.value, "detail": "Processing cancelled by request after pause."}
            return

        # Actual processing based on kind
        if input_data.envelope_kind == EnvelopeKind.PLAIN_TASK:
            logger.info(f"_DefaultSupervisorActor {self.actor_id} processing PLAIN_TASK for request {input_data.request_id}.")
            yield {
                "status": f"{self.name} (ID: {self.actor_id}) processed PLAIN_TASK for request {input_data.request_id}.",
                "actor_name": self.name, "actor_id": self.actor_id, "request_id": input_data.request_id,
                "context_id": input_data.context_id, "payload_type": type(input_data.payload).__name__,
            }
        elif input_data.envelope_kind == EnvelopeKind.CONTROL:
            logger.info(f"_DefaultSupervisorActor {self.actor_id} acknowledging CONTROL for request {input_data.request_id}.")
            yield {
                "status": f"{self.name} (ID: {self.actor_id}) acknowledged CONTROL for request {input_data.request_id}.",
                "actor_name": self.name, "actor_id": self.actor_id, "request_id": input_data.request_id,
                "control_payload": input_data.payload,
            }
        else:
            logger.warning(f"_DefaultSupervisorActor {self.actor_id} received unhandled kind {input_data.envelope_kind} for request {input_data.request_id}")
            yield { "error": f"{self.name} (ID: {self.actor_id}) cannot handle kind {input_data.envelope_kind}", "request_id": input_data.request_id, }
        logger.debug(f"_DefaultSupervisorActor '{self.name}' (ID: {self.actor_id}) run finished for request: {input_data.request_id}.")


    async def request_cancel(self) -> None:
        logger.info(f"_DefaultSupervisorActor {self.actor_id} ({self.name}) received cancel request.")
        self._cancel_requested = True

    async def request_pause(self) -> None:
        logger.info(f"_DefaultSupervisorActor {self.actor_id} ({self.name}) received pause request.")
        self._paused = True

    async def request_resume(self, followup_data: Optional[dict]) -> None:
        logger.info(f"_DefaultSupervisorActor {self.actor_id} ({self.name}) received resume request. Followup: {bool(followup_data)}. Resuming...")
        self._paused = False
        # followup_data is not used by this simple actor.

    async def cleanup(self) -> None:
        logger.debug(f"_DefaultSupervisorActor '{self.name}' (ID: {self.actor_id}) cleaning up.")
        pass


class LocalActorManager(ActorSupervisor):
    """Local, in-memory ActorSupervisor with capability awareness."""

    def __init__(self, actor_factory: _t.Callable[..., _t.Awaitable[tuple[CallableActor, Capability]]] | None = None):
        self._actors: dict[str, ActorInfo] = {}
        self._actor_factory = actor_factory

    async def acquire(self, actor_type: str, key: str, *args: _t.Any, **kwargs: _t.Any) -> CallableActor:
        actor_id = f'{actor_type}:{key}'
        if actor_id in self._actors:
            self._actors[actor_id].last_accessed = datetime.now()
            return self._actors[actor_id].agent
        actor_instance: CallableActor
        actor_capability: Capability
        if self._actor_factory:
            actor_instance, actor_capability = await self._actor_factory(actor_type, key, *args, **kwargs)
        else:
            actor_instance, actor_capability = self._create_default_actor_with_capability(actor_type, key, actor_id)
        actor_info = ActorInfo(
            actor_id=actor_id, actor_type=actor_type,
            agent=actor_instance, capability=actor_capability
        )
        self._actors[actor_id] = actor_info
        return actor_instance

    def _create_default_actor_with_capability(
        self, actor_type: str, key: str, actor_id: str
    ) -> tuple[CallableActor, Capability]:
        default_actor_name = f'{actor_type}-{key}'
        default_actor = _DefaultSupervisorActor(actor_id=actor_id, name=default_actor_name)
        return default_actor, _DefaultSupervisorActor.capability

    async def release(self, actor_id: str) -> None:
        if actor_id in self._actors:
            actor_info = self._actors[actor_id]
            actor_instance = actor_info.agent
            try:
                await actor_instance.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup of actor {actor_id}: {e}", exc_info=True)
            del self._actors[actor_id]

    async def list(self) -> list[ActorInfo]:
        return list(self._actors.values())

    async def info(self, actor_id: str) -> Optional[ActorInfo]:
        return self._actors.get(actor_id)

    async def get_actor_instance(self, actor_id: str) -> Optional[CallableActor]:
        actor_info = self._actors.get(actor_id)
        return actor_info.agent if actor_info else None

    async def cleanup_idle_actors(self, max_idle_time: float = 3600.0) -> int:
        current_time = datetime.now()
        idle_actor_ids: list[str] = []
        for actor_id, actor_info in self._actors.items():
            idle_seconds = (current_time - actor_info.last_accessed).total_seconds()
            if idle_seconds > max_idle_time:
                idle_actor_ids.append(actor_id)
        for actor_id_to_release in idle_actor_ids:
            await self.release(actor_id_to_release)
        return len(idle_actor_ids)
