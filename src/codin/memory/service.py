"""Unified memory service using endpoint configuration."""

from __future__ import annotations

import json
import typing as _t

from codin.agent.types import Message
from codin.endpoint import EndpointConfig, MemoryConfig
from codin.endpoint.resolver import EndpointManager

from .base import Memory as MemoryBase
from .base import MemoryChunk
from .local import MemMemoryService


class Memory(MemoryBase):
    """Unified memory service that supports both local and remote endpoints."""

    def __init__(self, config: MemoryConfig | EndpointConfig | None = None):
        if isinstance(config, MemoryConfig):
            self.config = config.endpoint
        elif isinstance(config, EndpointConfig):
            self.config = config
        else:
            # Auto-detect from environment
            memory_config = MemoryConfig.from_env()
            self.config = memory_config.endpoint
        self.manager = EndpointManager(self.config)
        self._chunk_creator: _t.Callable[[list[Message]], _t.Awaitable[list[MemoryChunk]]] | None = None
        self._current_session_id: str | None = None

        # For local endpoints, create a MemMemoryService instance
        self._local_service: MemMemoryService | None = None
        if self.config.is_local:
            # Use the local path as index path for LanceDB
            index_path = f"{self.config.local_path}/index" if self.config.local_path else None
            self._local_service = MemMemoryService(index_path=index_path)

    def set_session_id(self, session_id: str) -> None:
        """Set the current session ID."""
        self._current_session_id = session_id
        if self._local_service:
            self._local_service.set_session_id(session_id)

    async def add_message(self, message: Message) -> None:
        """Add a message to memory."""
        if self.config.is_local and self._local_service:
            await self._local_service.add_message(message)
        else:
            # Store message in remote endpoint
            session_id = message.contextId or self._current_session_id or "default"
            message_data = message.model_dump_json().encode("utf-8")
            await self.manager.write(f"sessions/{session_id}/messages/{message.messageId}.json", message_data)

    async def get_history(
        self, limit: int = 50, query: str | None = None, session_id: str | None = None
    ) -> list[Message]:
        """Get message history."""
        if self.config.is_local and self._local_service:
            return await self._local_service.get_history(limit, query, session_id)

        # Load from remote endpoint
        final_session_id = session_id or self._current_session_id or "default"
        messages = []

        try:
            # List message files in session directory
            message_files = await self.manager.list(f"sessions/{final_session_id}/messages")
            # Sort by filename (assuming timestamp-based naming)
            message_files.sort()

            # Load recent messages
            for message_file in message_files[-limit:]:
                try:
                    message_data = await self.manager.read(f"sessions/{final_session_id}/messages/{message_file}")
                    message_dict = json.loads(message_data.decode("utf-8"))
                    messages.append(Message(**message_dict))
                except Exception:
                    continue

            # If query is provided, search chunks and prepend
            if query:
                relevant_chunks = await self.search_chunk(final_session_id, query, limit=3)
                chunk_messages = [c.to_message() for c in relevant_chunks]
                return chunk_messages + messages

            return messages
        except Exception:
            return []

    async def set_chunk_builder(
        self, chunk_builder: _t.Callable[[list[Message]], _t.Awaitable[list[MemoryChunk]]]
    ) -> None:
        """Set the chunk builder function."""
        self._chunk_creator = chunk_builder
        if self._local_service:
            await self._local_service.set_chunk_builder(chunk_builder)

    async def build_chunk(self, start_index: int | None = None, end_index: int | None = None) -> int:
        """Build memory chunks from messages."""
        if self.config.is_local and self._local_service:
            return await self._local_service.build_chunk(start_index, end_index)

        # For remote endpoints, we need to implement chunk building
        if not self._chunk_creator:
            return 0

        session_id = self._current_session_id or "default"

        try:
            # Get all messages for the session
            all_messages = await self.get_history(limit=10000, session_id=session_id)

            if not all_messages:
                return 0

            # Apply index bounds
            if start_index is None:
                start_index = 0
            if end_index is None:
                end_index = len(all_messages)

            start_index = max(0, start_index)
            end_index = min(len(all_messages), end_index)

            if start_index >= end_index:
                return 0

            messages_to_compress = all_messages[start_index:end_index]

            if not messages_to_compress:
                return 0

            # Create chunks
            chunks = await self._chunk_creator(messages_to_compress)

            # Store chunks
            for chunk in chunks:
                chunk_data = chunk.model_dump_json().encode("utf-8")
                await self.manager.write(f"sessions/{session_id}/chunks/{chunk.chunk_id}.json", chunk_data)

            # Remove compressed messages (this is simplified - in production
            # you'd want more sophisticated message management)
            remaining_messages = all_messages[:start_index] + all_messages[end_index:]

            # Clear and re-store remaining messages
            try:
                # This is a simplified approach - production would be more efficient
                for i, message in enumerate(remaining_messages):
                    message_data = message.model_dump_json().encode("utf-8")
                    await self.manager.write(
                        f"sessions/{session_id}/messages/{i:06d}_{message.messageId}.json", message_data
                    )
            except Exception:
                pass  # If we can't clean up messages, that's okay

            return len(chunks) if chunks else 0

        except Exception:
            return 0

    async def search_chunk(self, session_id: str, query: str, limit: int = 5) -> list[MemoryChunk]:
        """Search memory chunks."""
        if self.config.is_local and self._local_service:
            return await self._local_service.search_chunk(session_id, query, limit)

        # Load chunks from remote endpoint and search
        chunks = []

        try:
            # List chunk files
            chunk_files = await self.manager.list(f"sessions/{session_id}/chunks")

            # Load all chunks
            for chunk_file in chunk_files:
                try:
                    chunk_data = await self.manager.read(f"sessions/{session_id}/chunks/{chunk_file}")
                    chunk_dict = json.loads(chunk_data.decode("utf-8"))
                    chunks.append(MemoryChunk(**chunk_dict))
                except Exception:
                    continue

            # Simple text-based search
            if not chunks:
                return []

            query_lower = query.lower()
            scored: list[tuple[int, MemoryChunk]] = []

            for chunk in chunks:
                score = 0

                if query_lower in chunk.title.lower():
                    score += 5

                if chunk.content and query_lower in chunk.content.lower():
                    score += 2

                if chunk.get_content_dict():
                    for key, value in chunk.get_content_dict().items():
                        if query_lower in key.lower() or query_lower in str(value).lower():
                            score += 1

                if query_lower == chunk.title.lower():
                    score += 3

                if query_lower in chunk.chunk_type.value.lower():
                    score += 1

                if score > 0:
                    scored.append((score, chunk))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:limit]]

        except Exception:
            return []

    async def close(self):
        """Close the memory service and cleanup resources."""
        await self.manager.close()
