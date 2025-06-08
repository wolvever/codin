"""HTTP-based memory client."""

from __future__ import annotations

import typing as _t

from codin.agent.types import Message

from ..client import Client, ClientConfig
from .base import Memory, MemoryChunk


class MemoryClient(Memory):
    """Client for a remote memory service."""

    def __init__(self, base_url: str, config: ClientConfig | None = None) -> None:
        self._client = Client(config or ClientConfig(base_url=base_url))
        self._chunk_creator: _t.Callable[[list[Message]], _t.Awaitable[list[MemoryChunk]]] | None = None
        self._current_session_id: str | None = None

    def set_session_id(self, session_id: str) -> None:
        self._current_session_id = session_id

    async def add_message(self, message: Message) -> None:
        await self._client.prepare()
        session_id = message.contextId or self._current_session_id or "default"
        await self._client.post(f"/sessions/{session_id}/messages", json=message.model_dump())

    async def get_history(self, limit: int = 50, query: str | None = None) -> list[Message]:
        await self._client.prepare()
        session_id = self._current_session_id or "default"
        params = {"limit": limit}
        if query:
            params["query"] = query
        resp = await self._client.get(f"/sessions/{session_id}/history", params=params)
        resp.raise_for_status()
        data = resp.json()
        return [Message(**m) for m in data.get("messages", [])]

    async def set_chunk_builder(
        self, chunk_builder: _t.Callable[[list[Message]], _t.Awaitable[list[MemoryChunk]]]
    ) -> None:
        self._chunk_creator = chunk_builder

    async def build_chunk(self, start_index: int | None = None, end_index: int | None = None) -> int:
        await self._client.prepare()
        session_id = self._current_session_id or "default"
        payload = {}
        if start_index is not None:
            payload["start_index"] = start_index
        if end_index is not None:
            payload["end_index"] = end_index
        resp = await self._client.post(f"/sessions/{session_id}/chunks", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("count", 0)

    async def search_chunk(self, session_id: str, query: str, limit: int = 5) -> list[MemoryChunk]:
        await self._client.prepare()
        params = {"query": query, "limit": limit}
        resp = await self._client.get(f"/sessions/{session_id}/chunks/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        return [MemoryChunk(**chunk) for chunk in data.get("chunks", [])]
