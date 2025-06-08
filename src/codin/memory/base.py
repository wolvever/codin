"""Core memory abstractions."""

from __future__ import annotations

import abc
import typing as _t
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from codin.agent.types import Message, Role, TextPart

__all__ = [
    "ChunkType",
    "MemoryChunk",
    "Memory",
    "MemoryService",
]


class ChunkType(str, Enum):
    """Types of memory chunks for different content categories."""

    MEMORY_ENTITY = "memory_entity"
    MEMORY_ID_MAPPING = "memory_id_mapping"
    MEMORY_SUMMARY = "memory_summary"


class MemoryChunk(BaseModel):
    """Enhanced memory chunk with structured content."""

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
    content: str | None = None

    def __init__(self, **data: _t.Any) -> None:  # type: ignore[override]
        content = data.pop("content", None)
        super().__init__(**data)

        if isinstance(content, dict):
            self._content_dict = content
            self.content = self._dict_to_searchable_string(content)
        else:
            self._content_dict = None
            self.content = content

    def _dict_to_searchable_string(self, content_dict: dict[str, _t.Any]) -> str:
        parts: list[str] = []

        def flatten(d: dict, prefix: str = "") -> None:
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flatten(value, full_key)
                elif isinstance(value, list | tuple):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            flatten(item, f"{full_key}[{i}]")
                        else:
                            parts.append(f"{full_key}[{i}]: {item!s}")
                else:
                    parts.append(f"{full_key}: {value!s}")

        flatten(content_dict)
        return "\n".join(parts)

    def get_content_dict(self) -> dict[str, _t.Any] | None:
        return self._content_dict

    def get_content_string(self) -> str:
        return self.content

    def to_message(self) -> Message:
        if self.chunk_type == ChunkType.MEMORY_SUMMARY:
            text = f"[MEMORY SUMMARY - {self.message_count} messages]\n"
            text += f"Title: {self.title}\n\n{self.content}"
        elif self.chunk_type == ChunkType.MEMORY_ENTITY:
            text = "[MEMORY ENTITIES]\n" + f"Title: {self.title}\n\n{self.content}"
        elif self.chunk_type == ChunkType.MEMORY_ID_MAPPING:
            text = "[MEMORY ID MAPPINGS]\n" + f"Title: {self.title}\n\n{self.content}"
        else:
            text = "[MEMORY CHUNK]\n" + f"Title: {self.title}\n\n{self.content}"

        return Message(
            messageId=f"memory-chunk-{self.chunk_id}",
            role=Role.user,
            parts=[TextPart(text=text)],
            contextId=self.session_id,
            kind="message",
            metadata={
                "doc_id": self.doc_id,
                "chunk_id": self.chunk_id,
                "chunk_type": self.chunk_type.value,
                "title": self.title,
                "message_count": self.message_count,
                "created_at": self.created_at.isoformat(),
                "is_memory_chunk": True,
                **self.metadata,
            },
        )


class Memory(abc.ABC):
    """Abstract chat/task memory backend with A2A Message support."""

    @abc.abstractmethod
    async def add_message(self, message: Message) -> None: ...

    @abc.abstractmethod
    async def get_history(self, limit: int = 50, query: str | None = None) -> list[Message]: ...

    @abc.abstractmethod
    async def set_chunk_builder(
        self, chunk_builder: _t.Callable[[list[Message]], _t.Awaitable[list[MemoryChunk]]]
    ) -> None: ...

    @abc.abstractmethod
    async def build_chunk(self, start_index: int | None = None, end_index: int | None = None) -> int: ...

    @abc.abstractmethod
    async def search_chunk(self, session_id: str, query: str, limit: int = 5) -> list[MemoryChunk]: ...


MemoryService = Memory

# Backwards-compatibility import
try:  # pragma: no cover - optional
    from .local import MemMemoryService  # noqa: F401

    __all__.append("MemMemoryService")
except Exception:  # pragma: no cover - ignore if local not available
    pass
