from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import BaseHost


try:  # pragma: no cover - optional dependency
    import ray
except Exception:  # pragma: no cover - used when ray missing
    ray = None

logger = logging.getLogger(__name__)


class RayHost(BaseHost):
    """Host that manages agents using Ray."""

    def __init__(self, config_file: str | Path | None = None, *, ray_init_kwargs: dict | None = None):
        super().__init__(config_file=config_file)
        self._ray_init_kwargs = ray_init_kwargs or {"local_mode": True, "ignore_reinit_error": True}

    async def _create_actor_manager(self) -> 'ActorSupervisor':
        if ray is None:
            raise ImportError("ray is required for RayHost")
        if not ray.is_initialized():  # pragma: no cover - init if needed
            ray.init(**self._ray_init_kwargs)
        return __import__('codin.actor.ray_scheduler', fromlist=['RayActorManager']).RayActorManager()

    async def _create_dispatcher(self, manager: 'ActorSupervisor') -> 'Dispatcher':
        # Dispatcher implementation is local for now
        return __import__('codin.actor.dispatcher', fromlist=['LocalDispatcher']).LocalDispatcher(manager)

    async def _down(self) -> None:
        await super()._down()
        if ray and ray.is_initialized():  # pragma: no cover
            ray.shutdown()
