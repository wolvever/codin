from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import BaseHost


logger = logging.getLogger(__name__)


class LocalHost(BaseHost):
    """Local in-process host implementation."""

    def __init__(self, config_file: str | Path | None = None):
        super().__init__(config_file=config_file)

    async def _create_actor_manager(self) -> 'ActorSupervisor':
        return __import__('codin.actor.supervisor', fromlist=['LocalActorManager']).LocalActorManager()

    async def _create_dispatcher(self, manager: 'ActorSupervisor') -> 'Dispatcher':
        return __import__('codin.actor.dispatcher', fromlist=['LocalDispatcher']).LocalDispatcher(manager)
