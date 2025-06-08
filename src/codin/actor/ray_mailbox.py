from __future__ import annotations

import typing as _t

from ..agent.types import Message
from .mailbox import Mailbox
from .queue_mailbox import QueueMailbox

try:  # pragma: no cover - optional dependency
    import ray
except Exception:  # pragma: no cover
    ray = None

__all__ = ["RayMailbox"]


if ray:  # pragma: no cover - avoid ray when not installed

    @ray.remote
    class _MailboxActor(QueueMailbox):
        def __init__(self, maxsize: int = 100):
            super().__init__(maxsize)

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
