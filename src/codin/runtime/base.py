from __future__ import annotations

import asyncio
import enum
import typing as _t
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

__all__ = [
    "WorkloadType",
    "Workload",
    "RuntimeResult",
    "Runtime",
]


class WorkloadType(str, enum.Enum):
    """Enumeration of supported workload kinds."""

    FUNCTION = "function"
    CLASS = "class"
    CLI = "cli"
    CONTAINER = "container"
    ENDPOINT = "endpoint"


class Workload(BaseModel):
    """Description of an executable workload.

    Depending on *kind*, one of the following fields must be provided:
      • FUNCTION / CLASS: ``callable``
      • CLI: ``command``
      • CONTAINER: ``image``
      • ENDPOINT: ``url``
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: WorkloadType
    callable: _t.Callable | None = None
    command: str | None = None
    image: str | None = None
    url: str | None = None
    args: list[str] | None = None
    kwargs: dict[str, _t.Any] | None = None
    timeout: float | None = None
    working_dir: str | None = None


class RuntimeResult(BaseModel):
    """Return payload from a Runtime execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    output: str | bytes | _t.Any = ""
    error: str | None = None

    # Streaming iterator (optional)
    stream: _t.AsyncIterator[str] | None = None


class Runtime(ABC):
    """Abstract base class for an execution backend."""

    name: str = "base"

    async def _timeout_wrapper(self, cor: _t.Coroutine, timeout: float | None) -> _t.Any:
        if timeout is None:
            return await cor
        try:
            return await asyncio.wait_for(cor, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Workload execution timed out") from exc

    async def run(self, workload: Workload, /, *, stream: bool = False) -> RuntimeResult:
        """Public entrypoint with timeout handling."""

        try:
            result = await self._timeout_wrapper(self._run(workload, stream=stream), workload.timeout)
            return result
        except Exception as exc:  # noqa: BLE001
            return RuntimeResult(success=False, error=str(exc))

    # ---------------------------------------------------------------------
    # Mandatory implementation
    # ---------------------------------------------------------------------
    @abstractmethod
    async def _run(self, workload: Workload, /, *, stream: bool = False) -> RuntimeResult:  # noqa: D401
        """Actual execution provided by concrete runtime.""" 