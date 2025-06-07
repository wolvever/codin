"""Mailbox implementations for inter-agent communication."""

from __future__ import annotations

import asyncio
import typing as _t
from abc import ABC, abstractmethod

from ..agent.types import Message

try:  # pragma: no cover - optional dependency
    import ray
except Exception:  # pragma: no cover
    ray = None


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


if ray:  # pragma: no cover - avoid ray when not installed

    @ray.remote
    class _MailboxActor:
        def __init__(self, maxsize: int = 100):
            self._inbox: asyncio.Queue[Message] = asyncio.Queue(maxsize=maxsize)
            self._outbox: asyncio.Queue[Message] = asyncio.Queue(maxsize=maxsize)

        async def put_inbox(self, msgs: list[Message]):
            for m in msgs:
                await self._inbox.put(m)

        async def put_outbox(self, msgs: list[Message]):
            for m in msgs:
                await self._outbox.put(m)

        async def get_inbox(self, n: int, timeout: float | None):
            res = []
            for _ in range(n):
                res.append(await asyncio.wait_for(self._inbox.get(), timeout=timeout))
            return res

        async def get_outbox(self, n: int, timeout: float | None):
            res = []
            for _ in range(n):
                res.append(await asyncio.wait_for(self._outbox.get(), timeout=timeout))
            return res

    class RayMailbox(Mailbox):
        """Mailbox backed by a Ray actor."""

        def __init__(self, agent_id: str, maxsize: int = 100):
            if ray is None:
                raise ImportError("ray is required for RayMailbox")
            self._actor = _MailboxActor.options(name=None).remote(maxsize)
            self.agent_id = agent_id

        async def put_inbox(self, msgs: Message | list[Message], timeout: float | None = None) -> None:
            if not isinstance(msgs, list):
                msgs = [msgs]
            await self._actor.put_inbox.remote(msgs)

        async def put_outbox(self, msgs: Message | list[Message], timeout: float | None = None) -> None:
            if not isinstance(msgs, list):
                msgs = [msgs]
            await self._actor.put_outbox.remote(msgs)

        async def get_inbox(
            self, max_messages: int = 1, timeout: float | None = None
        ) -> list[Message]:
            return await self._actor.get_inbox.remote(max_messages, timeout)

        async def get_outbox(
            self, max_messages: int = 1, timeout: float | None = None
        ) -> list[Message]:
            return await self._actor.get_outbox.remote(max_messages, timeout)

        async def subscribe_inbox(self) -> _t.AsyncIterator[Message]:
            while True:
                msgs = await self.get_inbox()
                for m in msgs:
                    yield m

        async def subscribe_outbox(self) -> _t.AsyncIterator[Message]:
            while True:
                msgs = await self.get_outbox()
                for m in msgs:
                    yield m

else:  # pragma: no cover - provide stub

    class RayMailbox(Mailbox):
        def __init__(self, *a, **k):
            raise ImportError("ray is required for RayMailbox")

        async def put_inbox(self, msgs: Message | list[Message], timeout: float | None = None) -> None:
            raise NotImplementedError()

        async def put_outbox(self, msgs: Message | list[Message], timeout: float | None = None) -> None:
            raise NotImplementedError()

        async def get_inbox(
            self, max_messages: int = 1, timeout: float | None = None
        ) -> list[Message]:
            raise NotImplementedError()

        async def get_outbox(
            self, max_messages: int = 1, timeout: float | None = None
        ) -> list[Message]:
            raise NotImplementedError()

        async def subscribe_inbox(self) -> _t.AsyncIterator[Message]:
            raise NotImplementedError()

        async def subscribe_outbox(self) -> _t.AsyncIterator[Message]:
            raise NotImplementedError()

