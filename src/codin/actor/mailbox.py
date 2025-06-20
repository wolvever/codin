"""Mailbox protocol for inter-agent communication."""

from __future__ import annotations

import typing as _t
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.types import Message

__all__ = ["Mailbox"]


class Mailbox(ABC):
    """Abstract bidirectional mailbox for agent communication.

    This class defines the interface for mailboxes, which are used by agents
    to send and receive messages asynchronously, both internally (inbox) and
    externally (outbox).
    """

    @abstractmethod
    async def put_inbox(
        self, msgs: Message | list[Message], timeout: float | None = None
    ) -> None:
        """Put message(s) into the agent's internal inbox.

        Args:
            msgs: A single message or a list of messages to be added.
            timeout: Optional timeout in seconds for the operation.
        """

    @abstractmethod
    async def put_outbox(
        self, msgs: Message | list[Message], timeout: float | None = None
    ) -> None:
        """Put message(s) into the agent's external outbox.

        Args:
            msgs: A single message or a list of messages to be added.
            timeout: Optional timeout in seconds for the operation.
        """

    @abstractmethod
    async def get_inbox(
        self, max_messages: int = 1, timeout: float | None = None
    ) -> list[Message]:
        """Get message(s) from the agent's internal inbox.

        Args:
            max_messages: The maximum number of messages to retrieve.
            timeout: Optional timeout in seconds for the operation.

        Returns:
            A list of messages retrieved from the inbox.
        """

    @abstractmethod
    async def get_outbox(
        self, max_messages: int = 1, timeout: float | None = None
    ) -> list[Message]:
        """Get message(s) from the agent's external outbox.

        Args:
            max_messages: The maximum number of messages to retrieve.
            timeout: Optional timeout in seconds for the operation.

        Returns:
            A list of messages retrieved from the outbox.
        """

    @abstractmethod
    async def subscribe_inbox(self) -> _t.AsyncIterator[Message]:
        """Subscribe to messages arriving in the agent's internal inbox.

        Returns:
            An asynchronous iterator yielding messages as they arrive.
        """

    @abstractmethod
    async def subscribe_outbox(self) -> _t.AsyncIterator[Message]:
        """Subscribe to messages being put into the agent's external outbox.

        Returns:
            An asynchronous iterator yielding messages as they are put.
        """
