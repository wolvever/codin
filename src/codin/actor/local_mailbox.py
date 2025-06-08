from __future__ import annotations

from typing import TYPE_CHECKING

from .queue_mailbox import QueueMailbox

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from ..agent.types import Message

__all__ = ["LocalMailbox"]


class LocalMailbox(QueueMailbox):
    """Local asyncio-based mailbox built on :class:`QueueMailbox`."""

    def __init__(self, maxsize: int = 100) -> None:
        super().__init__(maxsize=maxsize)
