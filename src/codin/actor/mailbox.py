"""Mailbox for inter-agent communication."""

import asyncio
import typing as _t
from datetime import datetime

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict

from a2a.types import Message

__all__ = ["Mailbox", "AsyncMailbox", "LocalAsyncMailbox", "MailboxMessage"]


class MailboxMessage(BaseModel):
    """Message wrapper for mailbox delivery."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    message: Message
    sender_id: str
    recipient_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, _t.Any] = Field(default_factory=dict)


class Mailbox(ABC):
    """Abstract bidirectional mailbox protocol from design document."""
    
    @abstractmethod
    async def put_inbox(self, msg: Message) -> None:
        """Put a message into the inbox (control/user feedback)."""
        pass
    
    @abstractmethod
    async def put_outbox(self, msg: Message) -> None:
        """Put a message into the outbox (events/deltas).""" 
        pass
    
    @abstractmethod
    async def get_inbox(self, timeout: float | None = None) -> Message:
        """Get a message from the inbox with optional timeout."""
        pass
    
    @abstractmethod
    async def get_outbox(self, timeout: float | None = None) -> Message:
        """Get a message from the outbox with optional timeout."""
        pass
    
    @abstractmethod
    async def subscribe_inbox(self) -> _t.AsyncIterator[Message]:
        """Subscribe to inbox messages."""
        pass
    
    @abstractmethod
    async def subscribe_outbox(self) -> _t.AsyncIterator[Message]:
        """Subscribe to outbox messages."""
        pass


class AsyncMailbox:
    """Legacy interface for backward compatibility."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._inbox: list[MailboxMessage] = []
        self._outbox: list[MailboxMessage] = []
        self._subscribers: list[_t.Callable[[MailboxMessage], _t.Awaitable[None]]] = []
    
    async def send_message(
        self, 
        message: Message, 
        recipient_id: str, 
        metadata: dict[str, _t.Any] | None = None
    ) -> None:
        """Send a message to another agent."""
        mailbox_msg = MailboxMessage(
            message=message,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            metadata=metadata or {}
        )
        self._outbox.append(mailbox_msg)
        
        # Notify subscribers
        for subscriber in self._subscribers:
            try:
                await subscriber(mailbox_msg)
            except Exception as e:
                # Log error but don't fail the send operation
                pass
    
    async def receive_message(self, mailbox_msg: MailboxMessage) -> None:
        """Receive a message from another agent."""
        if mailbox_msg.recipient_id == self.agent_id:
            self._inbox.append(mailbox_msg)
    
    def get_inbox_messages(self, limit: int | None = None) -> list[MailboxMessage]:
        """Get messages from inbox."""
        if limit:
            return self._inbox[-limit:]
        return list(self._inbox)
    
    def get_outbox_messages(self, limit: int | None = None) -> list[MailboxMessage]:
        """Get messages from outbox."""
        if limit:
            return self._outbox[-limit:]
        return list(self._outbox)
    
    def clear_inbox(self) -> None:
        """Clear all inbox messages."""
        self._inbox.clear()
    
    def clear_outbox(self) -> None:
        """Clear all outbox messages."""
        self._outbox.clear()
    
    def subscribe(self, callback: _t.Callable[[MailboxMessage], _t.Awaitable[None]]) -> None:
        """Subscribe to outgoing messages."""
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: _t.Callable[[MailboxMessage], _t.Awaitable[None]]) -> None:
        """Unsubscribe from outgoing messages."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)


class LocalAsyncMailbox(Mailbox):
    """Local asyncio implementation of bidirectional mailbox."""
    
    def __init__(self, agent_id: str, maxsize: int = 100):
        self.agent_id = agent_id
        self._inbox: asyncio.Queue[Message] = asyncio.Queue(maxsize=maxsize)
        self._outbox: asyncio.Queue[Message] = asyncio.Queue(maxsize=maxsize)
        
    async def put_inbox(self, msg: Message) -> None:
        """Put a message into the inbox (control/user feedback)."""
        await self._inbox.put(msg)
    
    async def put_outbox(self, msg: Message) -> None:
        """Put a message into the outbox (events/deltas)."""
        await self._outbox.put(msg)
    
    async def get_inbox(self, timeout: float | None = None) -> Message:
        """Get a message from the inbox with optional timeout."""
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError("Inbox get operation timed out")
    
    async def get_outbox(self, timeout: float | None = None) -> Message:
        """Get a message from the outbox with optional timeout."""
        try:
            return await asyncio.wait_for(self._outbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError("Outbox get operation timed out")
    
    async def subscribe_inbox(self) -> _t.AsyncIterator[Message]:
        """Subscribe to inbox messages."""
        while True:
            try:
                message = await self._inbox.get()
                yield message
            except asyncio.CancelledError:
                break
    
    async def subscribe_outbox(self) -> _t.AsyncIterator[Message]:
        """Subscribe to outbox messages."""
        while True:
            try:
                message = await self._outbox.get()
                yield message
            except asyncio.CancelledError:
                break 