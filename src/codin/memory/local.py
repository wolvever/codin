"""In-memory memory implementation with optional vector search."""

from __future__ import annotations

import typing as _t
import uuid
from datetime import datetime

from codin.agent.types import Message, Role

from .base import ChunkType, Memory, MemoryChunk

try:  # pragma: no cover - optional dependency
    import lancedb  # type: ignore
    import pyarrow as pa  # type: ignore
except Exception:  # pragma: no cover - handle missing dependency
    lancedb = None
    pa = None


def _hash_embed(text: str, dim: int = 64) -> list[float]:
    import hashlib

    data = hashlib.sha256(text.encode("utf-8")).digest()
    values = [b / 255 for b in data]
    while len(values) < dim:
        values.extend(values)
    return values[:dim]


class _LanceIndex:
    def __init__(self, path: str) -> None:
        if lancedb is None:
            raise RuntimeError("lancedb is not installed")
        self.db = lancedb.connect(path)
        if "chunks" in self.db.table_names():
            self.table = self.db.open_table("chunks")
        else:
            schema = pa.schema(
                [
                    pa.field("chunk_id", pa.string()),
                    pa.field("session_id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32())),
                    pa.field("text", pa.string()),
                ]
            )
            self.table = self.db.create_table("chunks", schema=schema)

    def add(self, chunk: MemoryChunk) -> None:
        vector = _hash_embed(chunk.get_content_string())
        self.table.add(
            pa.Table.from_pylist(
                [
                    {
                        "chunk_id": chunk.chunk_id,
                        "session_id": chunk.session_id,
                        "vector": vector,
                        "text": chunk.get_content_string(),
                    }
                ]
            )
        )

    def search(self, session_id: str, query: str, limit: int) -> list[str]:
        vector = _hash_embed(query)
        df = self.table.search(vector).where(f"session_id == '{session_id}'").limit(limit).to_pandas()
        return df.get("chunk_id", []).tolist()


class MemMemoryService(Memory):
    """In-memory implementation of :class:`Memory`."""

    def __init__(self, index_path: str | None = None) -> None:
        self._messages: dict[str, list[Message]] = {}
        self._chunks: dict[str, list[MemoryChunk]] = {}
        self._chunk_creator: _t.Callable[[list[Message]], _t.Awaitable[list[MemoryChunk]]] | None = None
        self._current_session_id: str | None = None
        self._chunk_map: dict[str, MemoryChunk] = {}
        self._index: _LanceIndex | None = None
        if index_path:
            try:
                self._index = _LanceIndex(index_path)
            except Exception:
                self._index = None

    def set_session_id(self, session_id: str) -> None:
        self._current_session_id = session_id

    async def add_message(self, message: Message) -> None:
        session_id = message.contextId or self._current_session_id or "default"
        self._messages.setdefault(session_id, []).append(message)

    async def get_history(
        self, limit: int = 50, query: str | None = None, session_id: str | None = None
    ) -> list[Message]:
        """Return conversation history for *session_id*.

        ``session_id`` is optional for backward compatibility. When omitted,
        the currently active session is used.
        """

        session_id = session_id or self._current_session_id or "default"
        messages = self._messages.get(session_id, [])
        recent_messages = messages[-limit:]
        if query:
            relevant = await self.search_chunk(session_id, query, limit=3)
            chunk_messages = [c.to_message() for c in relevant]
            return chunk_messages + recent_messages
        return recent_messages

    async def set_chunk_builder(
        self, chunk_creator: _t.Callable[[list[Message]], _t.Awaitable[list[MemoryChunk]]]
    ) -> None:
        self._chunk_creator = chunk_creator

    async def build_chunk(self, start_index: int | None = None, end_index: int | None = None) -> int:
        session_id = self._current_session_id or "default"
        messages = self._messages.get(session_id, [])
        if not messages or not self._chunk_creator:
            return 0
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(messages)
        start_index = max(0, start_index)
        end_index = min(len(messages), end_index)
        if start_index >= end_index:
            return 0
        messages_to_compress = messages[start_index:end_index]
        if not messages_to_compress:
            return 0
        try:
            chunks = await self._chunk_creator(messages_to_compress)
            for chunk in chunks:
                self._chunks.setdefault(session_id, []).append(chunk)
                self._chunk_map[chunk.chunk_id] = chunk
                if self._index:
                    self._index.add(chunk)
            self._messages[session_id] = messages[:start_index] + messages[end_index:]
            return len(chunks) if chunks else 0
        except Exception:
            return 0


    async def search_chunk(self, session_id: str, query: str, limit: int = 5) -> list[MemoryChunk]:
        if self._index:
            try:
                ids = self._index.search(session_id, query, limit)
                results = [self._chunk_map[i] for i in ids if i in self._chunk_map]
                if results:
                    return results
            except Exception:
                pass
        chunks = self._chunks.get(session_id, [])
        if not chunks:
            return []
        query_lower = query.lower()
        scored: list[tuple[int, MemoryChunk]] = []
        for chunk in chunks:
            score = 0
            if query_lower in chunk.title.lower():
                score += 5
            if query_lower in chunk.content.lower():
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


