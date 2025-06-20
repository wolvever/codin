"""Queue-based mailbox implementation for actor communication."""

from __future__ import annotations

import asyncio
import typing as _t
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from ..agent.types import Message
    from .mailbox import Mailbox
else:
    # Import at runtime to avoid circular import
    from .mailbox import Mailbox

__all__ = ["QueueMailbox"]


class QueueMailbox(Mailbox):
    """Queue-based mailbox implementation using asyncio queues.
    
    This class defines the interface for mailboxes, which are used by agents
    to send and receive messages asynchronously, both internally (inbox) and
    externally (outbox).
    """

    def __init__(self, maxsize: int = 100) -> None:
        """Initialize the mailbox with asyncio queues.
        
        Args:
            maxsize: Maximum size for the queues (0 for unlimited).
        """
        self._inbox: asyncio.Queue[Message] = asyncio.Queue(maxsize=maxsize)
        self._outbox: asyncio.Queue[Message] = asyncio.Queue(maxsize=maxsize)

    async def put_inbox(
        self, msgs: Message | list[Message], timeout: float | None = None
    ) -> None:
        """Put message(s) into the agent's internal inbox."""
        messages = msgs if isinstance(msgs, list) else [msgs]
        
        for msg in messages:
            if timeout is not None:
                await asyncio.wait_for(self._inbox.put(msg), timeout=timeout)
            else:
                await self._inbox.put(msg)

    async def put_outbox(
        self, msgs: Message | list[Message], timeout: float | None = None
    ) -> None:
        """Put message(s) into the agent's external outbox."""
        messages = msgs if isinstance(msgs, list) else [msgs]
        
        for msg in messages:
            if timeout is not None:
                await asyncio.wait_for(self._outbox.put(msg), timeout=timeout)
            else:
                await self._outbox.put(msg)

    async def get_inbox(
        self, max_messages: int = 1, timeout: float | None = None
    ) -> list[Message]:
        """Get message(s) from the agent's internal inbox."""
        messages: list[Message] = []
        
        for _ in range(max_messages):
            try:
                if timeout is not None:
                    msg = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
                else:
                    msg = await self._inbox.get()
                messages.append(msg)
            except asyncio.TimeoutError:
                break
            except asyncio.QueueEmpty:
                break
                
        return messages

    async def get_outbox(
        self, max_messages: int = 1, timeout: float | None = None
    ) -> list[Message]:
        """Get message(s) from the agent's external outbox."""
        messages: list[Message] = []
        
        for _ in range(max_messages):
            try:
                if timeout is not None:
                    msg = await asyncio.wait_for(self._outbox.get(), timeout=timeout)
                else:
                    msg = await self._outbox.get()
                messages.append(msg)
            except asyncio.TimeoutError:
                break
            except asyncio.QueueEmpty:
                break
                
        return messages

    async def subscribe_inbox(self) -> _t.AsyncIterator[Message]:
        """Subscribe to messages arriving in the agent's internal inbox."""
        while True:
            try:
                msg = await self._inbox.get()
                yield msg
            except asyncio.CancelledError:
                break

    async def subscribe_outbox(self) -> _t.AsyncIterator[Message]:
        """Subscribe to messages being put into the agent's external outbox."""
        while True:
            try:
                msg = await self._outbox.get()
                yield msg
            except asyncio.CancelledError:
                break