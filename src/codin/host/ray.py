from __future__ import annotations

from pathlib import Path
import logging

from .local import LocalHost

logger = logging.getLogger(__name__)


class RayHost(LocalHost):
    """Minimal Ray-based host using ``ray`` for process management."""

    def __init__(self, config_file: str | Path | None = None):
        super().__init__(config_file=config_file)

    async def _up(self) -> None:  # noqa: D401 - simple override
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        await super()._up()

    async def _down(self) -> None:  # noqa: D401 - simple override
        await super()._down()
        import ray

        if ray.is_initialized():
            ray.shutdown()
