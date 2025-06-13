from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from ..memory.base import Memory
from ..tool.base import Tool
from .types import RunConfig


@dataclass
class Session:
    """Execution context with tools, memory and config."""

    memory: Memory
    tools: List[Tool] = field(default_factory=list)
    config: RunConfig = field(default_factory=RunConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    async def add_message(self, message: "Message") -> None:
        await self.memory.add_message(message)

    async def get_history(self, limit: int = 50) -> List["Message"]:
        return await self.memory.get_history(limit=limit)
