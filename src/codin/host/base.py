from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import CodinConfig, load_config
from ..lifecycle import LifecycleMixin

if TYPE_CHECKING:
    from ..actor.dispatcher import Dispatcher
    from ..actor.supervisor import ActorSupervisor



logger = logging.getLogger(__name__)


class BaseHost(LifecycleMixin):
    """Base host that manages dispatcher and actor manager."""

    def __init__(self, config_file: str | Path | None = None) -> None:
        super().__init__()
        self.config_file = Path(config_file) if config_file else None
        self.config: CodinConfig | None = None
        self.dispatcher: Dispatcher | None = None
        self.actor_manager: ActorSupervisor | None = None
        self._resources: list[LifecycleMixin] = []

    async def _create_actor_manager(self) -> ActorSupervisor:
        """Create the actor manager instance."""
        return __import__('codin.actor.supervisor', fromlist=['LocalActorManager']).LocalActorManager()

    async def _create_dispatcher(self, manager: ActorSupervisor) -> Dispatcher:
        """Create the dispatcher instance."""
        return __import__('codin.actor.dispatcher', fromlist=['LocalDispatcher']).LocalDispatcher(manager)

    async def _init_components(self) -> None:
        self.actor_manager = await self._create_actor_manager()
        self.dispatcher = await self._create_dispatcher(self.actor_manager)
        for obj in (self.actor_manager, self.dispatcher):
            if isinstance(obj, LifecycleMixin):
                self._resources.append(obj)

    async def _up(self) -> None:
        self.config = load_config(self.config_file)
        await self._init_components()
        for res in self._resources:
            await res.up()
        logger.info("Host ready")

    async def _down(self) -> None:
        for res in reversed(self._resources):
            try:
                await res.down()
            except Exception as exc:  # pragma: no cover - best effort
                logger.error("Error shutting down %s: %s", res, exc)
        self._resources.clear()
        self.dispatcher = None
        self.actor_manager = None
        logger.info("Host stopped")
