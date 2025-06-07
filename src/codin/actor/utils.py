"""Utility helpers for the actor subsystem."""

from __future__ import annotations

import uuid
import typing as _t

from ..agent.types import Message, Role, TextPart

__all__ = ["make_message"]


def make_message(data: dict, context_id: str) -> Message:
    """Convert raw dict data into a :class:`Message`."""
    parts: list[TextPart] = []
    if "parts" in data:
        for part in data["parts"]:
            if part.get("kind") == "text":
                parts.append(TextPart(text=part.get("text", "")))
    elif "text" in data:
        parts.append(TextPart(text=data["text"]))
    else:
        parts.append(TextPart(text=""))

    return Message(
        messageId=data.get("messageId", str(uuid.uuid4())),
        role=Role(data.get("role", "user")),
        parts=parts,
        contextId=context_id,
        kind="message",
        metadata=data.get("metadata", {}),
    )
