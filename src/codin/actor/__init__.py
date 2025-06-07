"""Actor system for inter-agent communication.

This module provides the actor model infrastructure for codin agents,
including mailboxes for message passing, schedulers for task coordination,
and dispatchers for routing requests between agents.
"""

from .dispatcher import DispatchRequest, DispatchResult, Dispatcher, LocalDispatcher
from .mailbox import AsyncMailbox, LocalAsyncMailbox, Mailbox, MailboxMessage
from .scheduler import ActorInfo, ActorScheduler, LocalActorManager
from .ray_scheduler import RayActorManager


__all__ = [
    # Mailbox types
    'Mailbox',
    'AsyncMailbox',
    'LocalAsyncMailbox',
    'MailboxMessage',
    # Actor manager types
    'ActorScheduler',
    'LocalActorManager',
    'RayActorManager',
    'ActorInfo',
    # Dispatcher types
    'Dispatcher',
    'LocalDispatcher',
    'DispatchRequest',
    'DispatchResult',
]
