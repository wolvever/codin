"""Memory system interfaces and implementations.

This module provides the core memory abstractions for storing and retrieving
conversation history, memory chunks, and contextual information. It supports
A2A Message format and provides both abstract interfaces and in-memory implementations.
"""

import abc
import typing as _t
import uuid

from datetime import datetime
from enum import Enum

from a2a.types import Message, Role, TextPart
from pydantic import BaseModel, Field


__all__ = [
    'ChunkType',
    'MemMemoryService',
    'Memory',
    'MemoryChunk',
    'MemoryService',  # Alias for backward compatibility
]


class ChunkType(str, Enum):
    """Types of memory chunks for different content categories."""

    MEMORY_ENTITY = 'memory_entity'
    MEMORY_ID_MAPPING = 'memory_id_mapping'
    MEMORY_SUMMARY = 'memory_summary'


class MemoryChunk(BaseModel):
    """Enhanced memory chunk with structured content and search optimization."""

    doc_id: str
    chunk_id: str
    session_id: str
    chunk_type: ChunkType
    title: str
    start_message_id: str | None = None
    end_message_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    message_count: int = 0
    metadata: dict[str, _t.Any] = Field(default_factory=dict)
    _content_dict: dict[str, _t.Any] | None = None
    content: str

    def __init__(self, **data):
        """Initialize a memory chunk with content processing.

        Args:
            **data: Keyword arguments including content and other chunk fields
        """
        content = data.pop('content', None)
        super().__init__(**data)

        # Store content as string for search, preserve original for access
        if isinstance(content, dict):
            self._content_dict = content
            self.content = self._dict_to_searchable_string(content)
        else:
            self._content_dict = None
            self.content = content

    def _dict_to_searchable_string(self, content_dict: dict[str, _t.Any]) -> str:
        """Convert dictionary content to searchable string format."""
        searchable_parts = []

        def flatten_dict(d: dict, prefix: str = '') -> None:
            for key, value in d.items():
                full_key = f'{prefix}.{key}' if prefix else key
                if isinstance(value, dict):
                    flatten_dict(value, full_key)
                elif isinstance(value, (list, tuple)):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            flatten_dict(item, f'{full_key}[{i}]')
                        else:
                            searchable_parts.append(f'{full_key}[{i}]: {item!s}')
                else:
                    searchable_parts.append(f'{full_key}: {value!s}')

        flatten_dict(content_dict)
        return '\n'.join(searchable_parts)

    def get_content_dict(self) -> dict[str, _t.Any] | None:
        """Get the original dictionary content if available."""
        return self._content_dict

    def get_content_string(self) -> str:
        """Get the searchable string content."""
        return self.content

    def to_message(self) -> Message:
        """Convert memory chunk to A2A Message format."""
        # Create content based on chunk type
        if self.chunk_type == ChunkType.MEMORY_SUMMARY:
            message_text = f'[MEMORY SUMMARY - {self.message_count} messages]\n'
            message_text += f'Title: {self.title}\n\n{self.content}'
        elif self.chunk_type == ChunkType.MEMORY_ENTITY:
            message_text = '[MEMORY ENTITIES]\n'
            message_text += f'Title: {self.title}\n\n{self.content}'
        elif self.chunk_type == ChunkType.MEMORY_ID_MAPPING:
            message_text = '[MEMORY ID MAPPINGS]\n'
            message_text += f'Title: {self.title}\n\n{self.content}'
        else:
            message_text = '[MEMORY CHUNK]\n'
            message_text += f'Title: {self.title}\n\n{self.content}'

        return Message(
            messageId=f'memory-chunk-{self.chunk_id}',
            role=Role.user,  # Memory chunks appear as user context
            parts=[TextPart(text=message_text)],
            contextId=self.session_id,
            kind='message',  # Use standard message kind
            metadata={
                'doc_id': self.doc_id,
                'chunk_id': self.chunk_id,
                'chunk_type': self.chunk_type.value,
                'title': self.title,
                'message_count': self.message_count,
                'created_at': self.created_at.isoformat(),
                'is_memory_chunk': True,
                **self.metadata,  # Include any additional metadata
            },
        )


class Memory(abc.ABC):
    """Abstract chat/task memory backend with A2A Message support."""

    @abc.abstractmethod
    async def add_message(self, message: Message) -> None:
        """Add a message to memory."""
        ...

    @abc.abstractmethod
    async def get_history(self, limit: int = 50, query: str | None = None) -> list[Message]:
        """Get conversation history with optional search query.

        Args:
            limit: Maximum number of recent messages to return
            query: Optional search query to include relevant memory chunks

        Returns:
            List of Messages including recent messages and relevant memory chunks
        """
        ...

    @abc.abstractmethod
    async def set_chunk_builder(
        self, chunk_builder: _t.Callable[[list[Message]], _t.Awaitable[list[MemoryChunk]]]
    ) -> None:
        """Set a callable that creates memory chunks from a list of messages.

        Args:
            chunk_builder: Async function that takes a list of Messages and returns MemoryChunks
        """
        ...

    @abc.abstractmethod
    async def build_chunk(self, start_index: int | None = None, end_index: int | None = None) -> int:
        """Compress messages into memory chunks with optional range.

        Args:
            start_index: Optional start index of messages to compress
            end_index: Optional end index of messages to compress

        Returns:
            Number of chunk groups created
        """
        ...

    @abc.abstractmethod
    async def search_chunk(self, session_id: str, query: str, limit: int = 5) -> list[MemoryChunk]:
        """Search memory chunks by query.

        Args:
            session_id: Session identifier
            query: Search query
            limit: Maximum number of chunks to return

        Returns:
            List of relevant memory chunks
        """
        ...


class MemMemoryService(Memory):
    """In-memory implementation of Memory with A2A Message support."""

    def __init__(self):
        """Initialize an in-memory memory service."""
        self._messages: dict[str, list[Message]] = {}
        self._chunks: dict[str, list[MemoryChunk]] = {}
        self._chunk_creator: _t.Callable[[list[Message]], _t.Awaitable[list[MemoryChunk]]] | None = None
        self._current_session_id: str | None = None

    def set_session_id(self, session_id: str) -> None:
        """Set the current session ID for this memory instance."""
        self._current_session_id = session_id

    async def add_message(self, message: Message) -> None:
        """Add a message to memory."""
        session_id = message.contextId or self._current_session_id or 'default'
        self._messages.setdefault(session_id, []).append(message)

    async def get_history(self, limit: int = 50, query: str | None = None) -> list[Message]:
        """Get conversation history with optional search query."""
        session_id = self._current_session_id or 'default'
        messages = self._messages.get(session_id, [])
        recent_messages = messages[-limit:]

        if query:
            # Search for relevant memory chunks
            relevant_chunks = await self.search_chunk(session_id, query, limit=3)
            chunk_messages = [chunk.to_message() for chunk in relevant_chunks]
            # Insert chunks at the beginning to provide context
            return chunk_messages + recent_messages

        return recent_messages

    async def set_chunk_builder(
        self, chunk_creator: _t.Callable[[list[Message]], _t.Awaitable[list[MemoryChunk]]]
    ) -> None:
        """Set a callable that creates memory chunks from a list of messages."""
        self._chunk_creator = chunk_creator

    async def build_chunk(self, start_index: int | None = None, end_index: int | None = None) -> int:
        """Compress messages into memory chunks with optional range."""
        session_id = self._current_session_id or 'default'
        messages = self._messages.get(session_id, [])

        if not messages or not self._chunk_creator:
            return 0

        # Determine the range of messages to compress
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(messages)

        # Ensure indices are valid
        start_index = max(0, start_index)
        end_index = min(len(messages), end_index)

        if start_index >= end_index:
            return 0

        # Get messages to compress
        messages_to_compress = messages[start_index:end_index]

        if not messages_to_compress:
            return 0

        try:
            # Create chunks using the configured creator
            chunks = await self._chunk_creator(messages_to_compress)

            # Store the chunks
            for chunk in chunks:
                self._chunks.setdefault(session_id, []).append(chunk)

            # Remove the compressed messages from the original list
            self._messages[session_id] = messages[:start_index] + messages[end_index:]

            return len(chunks) if chunks else 0

        except Exception:
            # If chunk creation fails, don't modify the messages
            return 0

    async def search_chunk(self, session_id: str, query: str, limit: int = 5) -> list[MemoryChunk]:
        """Search memory chunks by query with title weighting."""
        chunks = self._chunks.get(session_id, [])
        if not chunks:
            return []

        # Enhanced text-based search with title weighting
        query_lower = query.lower()
        scored_chunks = []

        for chunk in chunks:
            score = 0

            # Search in title (highest weight)
            if query_lower in chunk.title.lower():
                score += 5

            # Search in content (medium weight)
            if query_lower in chunk.content.lower():
                score += 2

            # For dictionary content, also search in original structure
            if chunk.get_content_dict():
                content_dict = chunk.get_content_dict()
                for key, value in content_dict.items():
                    if query_lower in key.lower() or query_lower in str(value).lower():
                        score += 1

            # Bonus for exact matches in title
            if query_lower == chunk.title.lower():
                score += 3

            # Bonus for chunk type relevance
            if query_lower in chunk.chunk_type.value.lower():
                score += 1

            if score > 0:
                scored_chunks.append((score, chunk))

        # Sort by score (descending) and return top results
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:limit]]

    def _simple_summarize(self, text: str) -> str:
        """Simple text summarization fallback."""
        lines = text.split('\n')
        if len(lines) <= 3:
            return text

        # Take first and last few lines
        summary_lines = lines[:2] + ['...'] + lines[-2:]
        return '\n'.join(summary_lines)

    async def compress_old_messages(
        self,
        session_id: str,
        keep_recent: int = 20,
        chunk_size: int = 10,
        llm_summarizer: _t.Callable[[str], _t.Awaitable[dict[str, _t.Any]]] | None = None,
    ) -> int:
        """Compress old messages into memory chunks.

        Args:
            session_id: Session to compress
            keep_recent: Number of recent messages to keep uncompressed
            chunk_size: Number of messages per chunk
            llm_summarizer: Optional LLM summarizer function

        Returns:
            Number of chunk groups created (each group may contain multiple chunks)
        """
        messages = self._messages.get(session_id, [])
        if len(messages) <= keep_recent:
            return 0

        # Messages to compress (all except the most recent)
        to_compress = messages[:-keep_recent]
        chunk_groups_created = 0

        # Create chunks using the legacy method for compatibility
        for i in range(0, len(to_compress), chunk_size):
            chunk_messages = to_compress[i : i + chunk_size]
            if chunk_messages:
                chunks = await self._create_memory_chunk_legacy(session_id, chunk_messages, llm_summarizer)
                chunk_groups_created += 1

        # Remove compressed messages, keep only recent ones
        self._messages[session_id] = messages[-keep_recent:]

        return chunk_groups_created

    async def _create_memory_chunk_legacy(
        self,
        session_id: str,
        messages: list[Message],
        llm_summarizer: _t.Callable[[str], _t.Awaitable[dict[str, _t.Any]]] | None = None,
    ) -> list[MemoryChunk]:
        """Legacy method for creating memory chunks - kept for backward compatibility."""
        if not messages:
            raise ValueError('Cannot create memory chunk from empty message list')

        doc_id = str(uuid.uuid4())
        start_message_id = messages[0].messageId
        end_message_id = messages[-1].messageId

        # Extract text content from messages
        text_content = []
        for msg in messages:
            role = 'User' if msg.role == Role.user else 'Assistant'
            for part in msg.parts:
                if hasattr(part, 'text'):
                    text_content.append(f'{role}: {part.text}')
                elif hasattr(part, 'root') and hasattr(part.root, 'text'):
                    text_content.append(f'{role}: {part.root.text}')

        conversation_text = '\n'.join(text_content)

        if llm_summarizer:
            # Use LLM for intelligent summarization
            try:
                summary_result = await llm_summarizer(conversation_text)
                summary = summary_result.get('summary', 'Conversation summary')
                entities = summary_result.get('entities', {})
                id_mappings = summary_result.get('id_mappings', {})
            except Exception:
                # Fallback to simple summarization
                summary = self._simple_summarize(conversation_text)
                entities = {}
                id_mappings = {}
        else:
            # Simple summarization
            summary = self._simple_summarize(conversation_text)
            entities = {}
            id_mappings = {}

        chunks = []

        # Create summary chunk
        summary_chunk = MemoryChunk(
            doc_id=doc_id,
            chunk_id=f'{doc_id}-summary',
            session_id=session_id,
            chunk_type=ChunkType.MEMORY_SUMMARY,
            content=summary,
            title=f'Conversation Summary ({len(messages)} messages)',
            start_message_id=start_message_id,
            end_message_id=end_message_id,
            created_at=datetime.now(),
            message_count=len(messages),
        )
        chunks.append(summary_chunk)

        # Create entities chunk if we have entities
        if entities:
            entities_chunk = MemoryChunk(
                doc_id=doc_id,
                chunk_id=f'{doc_id}-entities',
                session_id=session_id,
                chunk_type=ChunkType.MEMORY_ENTITY,
                content=entities,
                title='Extracted Entities',
                start_message_id=start_message_id,
                end_message_id=end_message_id,
                created_at=datetime.now(),
                message_count=len(messages),
            )
            chunks.append(entities_chunk)

        # Create ID mappings chunk if we have mappings
        if id_mappings:
            mappings_chunk = MemoryChunk(
                doc_id=doc_id,
                chunk_id=f'{doc_id}-mappings',
                session_id=session_id,
                chunk_type=ChunkType.MEMORY_ID_MAPPING,
                content=id_mappings,
                title='ID Mappings',
                start_message_id=start_message_id,
                end_message_id=end_message_id,
                created_at=datetime.now(),
                message_count=len(messages),
            )
            chunks.append(mappings_chunk)

        # Store all chunks
        for chunk in chunks:
            self._chunks.setdefault(session_id, []).append(chunk)

        return chunks


# Backward compatibility alias
MemoryService = Memory
