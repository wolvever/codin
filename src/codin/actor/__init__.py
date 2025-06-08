"""Actor system for inter-agent communication.

This module provides the actor model infrastructure for codin agents,
including mailboxes for message passing, schedulers for task coordination,
and dispatchers for routing requests between agents.
"""

from .actor_system import ActorSystem
from .dispatcher import Dispatcher, DispatchRequest, DispatchResult, LocalDispatcher
from .mailbox import LocalMailbox, Mailbox, RayMailbox
from .ray_scheduler import RayActorManager
from .supervisor import ActorInfo, ActorSupervisor, LocalActorManager

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
    'ActorSystem',
]
