"""Mailbox protocol for inter-agent communication."""

from __future__ import annotations

import asyncio
import typing as _t
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..agent.types import Message


__all__ = ["Mailbox", "LocalMailbox", "RayMailbox"]


class Mailbox(ABC):
    """Abstract bidirectional mailbox."""

    @abstractmethod
    async def put_inbox(
        self, msgs: Message | list[Message], timeout: float | None = None
    ) -> None:
        """Put message(s) into inbox."""

    @abstractmethod
    async def put_outbox(
        self, msgs: Message | list[Message], timeout: float | None = None
    ) -> None:
        """Put message(s) into outbox."""

    @abstractmethod
    async def get_inbox(
        self, max_messages: int = 1, timeout: float | None = None
    ) -> list[Message]:
        """Get up to ``max_messages`` from inbox."""

    @abstractmethod
    async def get_outbox(
        self, max_messages: int = 1, timeout: float | None = None
    ) -> list[Message]:
        """Get up to ``max_messages`` from outbox."""

    @abstractmethod
    async def subscribe_inbox(self) -> _t.AsyncIterator[Message]:
        """Iterate over inbox messages."""

    @abstractmethod
    async def subscribe_outbox(self) -> _t.AsyncIterator[Message]:
        """Iterate over outbox messages."""


from .local_mailbox import LocalMailbox
from .ray_mailbox import RayMailbox

