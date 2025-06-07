"""Prompt-based chunk builder utilities."""

from __future__ import annotations

import json
import typing as _t
import uuid
from datetime import datetime

from a2a.types import Message, Role

from ..prompt import prompt_run
from .base import ChunkType, MemoryChunk


async def prompt_chunk_builder(messages: list[Message]) -> list[MemoryChunk]:
    """Create memory chunks using :func:`prompt_run` to summarize messages."""
    if not messages:
        raise ValueError("Cannot create memory chunk from empty message list")

    lines: list[str] = []
    for m in messages:
        role = "User" if m.role == Role.user else "Assistant"
        for part in m.parts:
            if hasattr(part, "root") and hasattr(part.root, "text"):
                lines.append(f"{role}: {part.root.text}")
            elif hasattr(part, "text"):
                lines.append(f"{role}: {part.text}")
    conversation_text = "\n".join(lines)

    response = await prompt_run("conversation_summary", variables={"conversation_text": conversation_text})

    content = ""
    if hasattr(response, "message") and response.message:
        for p in response.message.parts:
            if hasattr(p, "root") and hasattr(p.root, "text"):
                content += p.root.text
            elif hasattr(p, "text"):
                content += p.text
    elif hasattr(response, "content"):
        content = str(response.content)
    else:
        content = str(response)

    try:
        data = json.loads(content)
    except Exception:
        data = {"summary": content, "entities": {}, "id_mappings": {}}

    doc_id = str(uuid.uuid4())
    start_id = messages[0].messageId
    end_id = messages[-1].messageId
    session_id = messages[0].contextId or "default"

    chunks: list[MemoryChunk] = []
    summary_chunk = MemoryChunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}-summary",
        session_id=session_id,
        chunk_type=ChunkType.MEMORY_SUMMARY,
        content=data.get("summary", ""),
        title=f"Conversation Summary ({len(messages)} messages)",
        start_message_id=start_id,
        end_message_id=end_id,
        created_at=datetime.now(),
        message_count=len(messages),
    )
    chunks.append(summary_chunk)

    entities = data.get("entities", {})
    if entities:
        chunks.append(
            MemoryChunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}-entities",
                session_id=session_id,
                chunk_type=ChunkType.MEMORY_ENTITY,
                content=entities,
                title="Extracted Entities",
                start_message_id=start_id,
                end_message_id=end_id,
                created_at=datetime.now(),
                message_count=len(messages),
            )
        )

    mappings = data.get("id_mappings", {})
    if mappings:
        chunks.append(
            MemoryChunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}-mappings",
                session_id=session_id,
                chunk_type=ChunkType.MEMORY_ID_MAPPING,
                content=mappings,
                title="ID Mappings",
                start_message_id=start_id,
                end_message_id=end_id,
                created_at=datetime.now(),
                message_count=len(messages),
            )
        )
    return chunks
