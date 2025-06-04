"""Mailbox for inter-agent communication."""

import asyncio
import typing as _t
from datetime import datetime
from dataclasses import dataclass, field

from a2a.types import Message

__all__ = ["Mailbox", "MailboxMessage"]


@dataclass
class MailboxMessage:
    """Message wrapper for mailbox delivery."""
    message: Message
    sender_id: str
    recipient_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, _t.Any] = field(default_factory=dict)


class Mailbox:
    """Mailbox for inter-agent communication."""
    
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