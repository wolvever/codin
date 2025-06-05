"""Actor module for inter-agent communication."""

from .mailbox import Mailbox, AsyncMailbox, LocalAsyncMailbox, MailboxMessage
from .actor_mngr import ActorManager, LocalActorManager, ActorInfo
from .dispatcher import Dispatcher, LocalDispatcher, DispatchRequest, DispatchResult

__all__ = [
    # Mailbox types
    "Mailbox", 
    "AsyncMailbox", 
    "LocalAsyncMailbox", 
    "MailboxMessage",
    
    # Actor manager types
    "ActorManager",
    "LocalActorManager", 
    "ActorInfo",
    
    # Dispatcher types
    "Dispatcher",
    "LocalDispatcher",
    "DispatchRequest", 
    "DispatchResult",
] 