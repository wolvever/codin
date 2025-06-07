"""Actor system for inter-agent communication.

This module provides the actor model infrastructure for codin agents,
including mailboxes for message passing, schedulers for task coordination,
and dispatchers for routing requests between agents.
"""

from .dispatcher import DispatchRequest, DispatchResult, Dispatcher, LocalDispatcher
from .mailbox import Mailbox, LocalMailbox, RayMailbox
from .supervisor import ActorInfo, ActorSupervisor, LocalActorManager
from .ray_scheduler import RayActorManager


__all__ = [
    # Mailbox types
    'Mailbox',
    'LocalMailbox',
    'RayMailbox',
    # Actor manager types
    'ActorSupervisor',
    'LocalActorManager',
    'RayActorManager',
    'ActorInfo',
    # Dispatcher types
    'Dispatcher',
    'LocalDispatcher',
    'DispatchRequest',
    'DispatchResult',
]
