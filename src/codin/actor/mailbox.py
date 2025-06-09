"""Mailbox protocol for inter-agent communication."""

from __future__ import annotations

import typing as _t
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.types import Message
    # Forward reference for LocalMailbox, and RayMailbox if it's conditionally imported
    from .local_mailbox import LocalMailbox # Ensure LocalMailbox is recognized for type hints
    # RayMailbox is conditionally imported below, but we might need its type.
    # It will be properly imported for type checking in the try/except block if Ray is not present.

from .local_mailbox import LocalMailbox

__all__ = ["Mailbox", "LocalMailbox"]

try:
    # Try to import Ray and RayMailbox for runtime availability.
    import ray  # type: ignore
    from .ray_mailbox import RayMailbox
    __all__.append("RayMailbox")
except ImportError:
    # Ray is not installed or RayMailbox is not available at runtime.
    # We still need the type hint for RayMailbox if type checking (e.g., mypy) is active,
    # to allow for code that type-hints RayMailbox usage without causing a runtime error
    # when Ray is absent.
    if TYPE_CHECKING:
        from .ray_mailbox import RayMailbox # Make type available for static analysis
    pass


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
