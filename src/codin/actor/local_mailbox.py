from __future__ import annotations

import asyncio
import typing as _t

from .mailbox import Mailbox
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.types import Message

__all__ = ["LocalMailbox"]


class LocalMailbox(Mailbox):
    """Local asyncio based mailbox."""

    def __init__(self, maxsize: int = 100):
        self._inbox: asyncio.Queue[Message] = asyncio.Queue(maxsize=maxsize)
        self._outbox: asyncio.Queue[Message] = asyncio.Queue(maxsize=maxsize)

    async def _put(
        self, q: asyncio.Queue[Message], msgs: Message | list[Message], timeout: float | None
    ) -> None:
        if not isinstance(msgs, list):
            msgs = [msgs]
        for msg in msgs:
            if timeout is None:
                while not q.empty():
                    await asyncio.sleep(0)
                await q.put(msg)
            else:
                await asyncio.wait_for(q.put(msg), timeout=timeout)

    async def put_inbox(self, msgs: Message | list[Message], timeout: float | None = None) -> None:
        await self._put(self._inbox, msgs, timeout)

    async def put_outbox(self, msgs: Message | list[Message], timeout: float | None = None) -> None:
        await self._put(self._outbox, msgs, timeout)

    async def _get(
        self, q: asyncio.Queue[Message], max_messages: int, timeout: float | None
    ) -> list[Message]:
        msgs: list[Message] = []
        for _ in range(max_messages):
            msg = await asyncio.wait_for(q.get(), timeout=timeout)
            msgs.append(msg)
        return msgs

    async def get_inbox(
        self, max_messages: int = 1, timeout: float | None = None
    ) -> list[Message]:
        return await self._get(self._inbox, max_messages, timeout)

    async def get_outbox(
        self, max_messages: int = 1, timeout: float | None = None
    ) -> list[Message]:
        return await self._get(self._outbox, max_messages, timeout)

    async def subscribe_inbox(self) -> _t.AsyncIterator[Message]:
        while True:
            msg = await self._inbox.get()
            yield msg

    async def subscribe_outbox(self) -> _t.AsyncIterator[Message]:
        while True:
            msg = await self._outbox.get()
            yield msg
