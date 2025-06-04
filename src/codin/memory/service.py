"""Memory service implementation that bridges MemorySystem to MemoryService interface."""

import abc
import typing as _t
from a2a.types import Message

from .base import MemorySystem, InMemoryStore

__all__ = [
    "ChatHistory",
    "MemoryService", 
    "MemorySystemChatHistory",
    "MemorySystemService",
]


class ChatHistory(abc.ABC):
    """Readonly chat history interface."""
    
    @abc.abstractmethod
    def get_recent_messages(self, count: int = 10) -> list[Message]:
        """Get recent messages for context."""
        ...
    
    @abc.abstractmethod
    def search_messages(self, query: str) -> list[Message]:
        """Search through message history."""
        ...
    
    @abc.abstractmethod
    def get_all_messages(self) -> list[Message]:
        """Get all messages in history."""
        ...


class MemoryService(abc.ABC):
    """Service for managing conversation memory and chat history."""
    
    @abc.abstractmethod
    async def get_chat_history(self, session_id: str) -> ChatHistory:
        """Get chat history for session."""
        ...
    
    @abc.abstractmethod 
    async def add_message(self, session_id: str, message: Message) -> None:
        """Add message to chat history."""
        ...


class MemorySystemChatHistory(ChatHistory):
    """ChatHistory implementation that wraps a MemorySystem and session_id."""
    
    def __init__(self, memory_system: MemorySystem, session_id: str):
        self.memory_system = memory_system
        self.session_id = session_id
        self._cached_messages: list[Message] = []
        self._last_fetch_count = 0
    
    async def _ensure_messages_loaded(self, count: int = 50) -> None:
        """Ensure we have loaded enough messages."""
        if len(self._cached_messages) < count or count > self._last_fetch_count:
            self._cached_messages = await self.memory_system.get_history(
                self.session_id, 
                limit=max(count, 100)  # Load extra for caching
            )
            self._last_fetch_count = max(count, 100)
    
    def get_recent_messages(self, count: int = 10) -> list[Message]:
        """Get recent messages for context."""
        # Since this is a sync method but we need async data, we'll use the cached messages
        # In practice, the agent should call the async memory service methods directly
        return self._cached_messages[-count:] if count > 0 else self._cached_messages[:]
    
    def search_messages(self, query: str) -> list[Message]:
        """Search through message history."""
        # Simple text search on cached messages
        results = []
        query_lower = query.lower()
        
        for message in self._cached_messages:
            for part in message.parts:
                if hasattr(part, 'text') and query_lower in part.text.lower():
                    results.append(message)
                    break
        
        return results
    
    def get_all_messages(self) -> list[Message]:
        """Get all messages in history."""
        return self._cached_messages[:]


class MemorySystemService(MemoryService):
    """MemoryService implementation that uses the existing MemorySystem."""
    
    def __init__(self, memory_system: MemorySystem | None = None):
        self.memory_system = memory_system or InMemoryStore()
        self._chat_histories: dict[str, MemorySystemChatHistory] = {}
    
    async def get_chat_history(self, session_id: str) -> ChatHistory:
        """Get chat history for session."""
        if session_id not in self._chat_histories:
            chat_history = MemorySystemChatHistory(self.memory_system, session_id)
            # Pre-load some messages for the sync methods
            await chat_history._ensure_messages_loaded(50)
            self._chat_histories[session_id] = chat_history
        return self._chat_histories[session_id]
    
    async def add_message(self, session_id: str, message: Message) -> None:
        """Add message to chat history."""
        await self.memory_system.add_message(message)
        
        # Update cached messages if we have this session loaded
        if session_id in self._chat_histories:
            chat_history = self._chat_histories[session_id]
            chat_history._cached_messages.append(message)
    
    async def get_history_with_search(
        self, 
        session_id: str, 
        limit: int = 50,
        query: str | None = None
    ) -> list[Message]:
        """Get conversation history with optional search query."""
        return await self.memory_system.get_history(session_id, limit, query)
    
    async def compress_old_messages(
        self,
        session_id: str,
        keep_recent: int = 20,
        chunk_size: int = 10,
        llm_summarizer: _t.Callable[[str], _t.Awaitable[dict[str, _t.Any]]] | None = None
    ) -> int:
        """Compress old messages using the memory system."""
        if hasattr(self.memory_system, 'compress_old_messages'):
            return await self.memory_system.compress_old_messages(
                session_id, keep_recent, chunk_size, llm_summarizer
            )
        return 0 