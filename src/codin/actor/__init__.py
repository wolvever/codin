"""Actor system for inter-agent communication and task processing.

This module provides the actor model infrastructure for CoDIN,
including mailboxes for message passing, supervisors for lifecycle management,
dispatchers for routing requests, concrete actor implementations, task management,
and standardized types for actor communication (envelopes, inputs, outputs,
protocols, task states, and control actions).
"""

from .actor_system import ActorSystem
from .work_stealing import WorkStealingActorSystem
from .dispatcher import Dispatcher, DispatchRequest, DispatchResult, LocalDispatcher # DispatchRequest might be deprecated
from .mailbox import LocalMailbox, Mailbox # RayMailbox is conditionally imported in mailbox.py
from .ray_scheduler import RayActorManager
from .supervisor import ActorInfo, ActorSupervisor, LocalActorManager
from .types import ActorRunInput, ActorRunOutput, CallableActor
from .envelope_types import (
    Envelope,
    EnvelopeKind,
    EnvelopeHeaders,
    AuthDetails,
    ControlPayload,
    Capability,
    TaskState,
    ControlAction,
)
from .actors import AgentActor, PlainActor, AGENT_CAP
from .task_manager import TaskInfo, TaskRegistry # Added TaskManager imports


# Conditionally add RayMailbox to __all__ if it's available.
_ray_mailbox_available = False
try:
    from .mailbox import RayMailbox as _RayMailbox_alias
    _ray_mailbox_available = True
except ImportError:
    pass

__all__ = [
    # Mailbox types
    'Mailbox',
    'LocalMailbox',
    # Actor Supervisor types
    'ActorSupervisor',
    'LocalActorManager',
    'ActorInfo',
    # Ray-specific actor manager
    'RayActorManager',
    # Dispatcher types
    'Dispatcher',
    'LocalDispatcher',
    'DispatchRequest',
    'DispatchResult',
    # Actor Core Protocol and I/O types
    'CallableActor',
    'ActorRunInput',
    'ActorRunOutput',
    # Envelope, Task State, and Control Action types
    'Envelope',
    'EnvelopeKind',
    'EnvelopeHeaders',
    'AuthDetails',
    'ControlPayload',
    'ControlAction',
    'Capability',
    'TaskState',
    # Task Management types
    'TaskInfo',
    'TaskRegistry',
    # Concrete Actor Implementations & Capabilities
    'AgentActor',
    'PlainActor',
    'AGENT_CAP',
    # Actor System types
    'ActorSystem',
    'WorkStealingActorSystem',
]

if _ray_mailbox_available:
    __all__.append('RayMailbox')
