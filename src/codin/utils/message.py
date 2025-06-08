"""Utility helpers for working with A2A messages."""

from __future__ import annotations

from a2a.types import Message

__all__ = [
    "extract_text_from_message",
    "format_history_for_prompt",
    "format_tool_results_for_conversation",
]


def extract_text_from_message(message: Message) -> str:
    """Extract concatenated text from all parts of a message."""
    text_parts: list[str] = []
    for part in message.parts:
        if hasattr(part, "root") and hasattr(part.root, "text"):
            text_parts.append(part.root.text)
        elif hasattr(part, "text"):
            text_parts.append(part.text)
    return "\n".join(text_parts)


def format_history_for_prompt(history_messages: list[dict]) -> str:
    """Format conversation history for LLM prompt consumption."""
    if not history_messages:
        return ""
    formatted = []
    for msg in history_messages:
        role = msg["role"].title()
        formatted.append(f"{role}: {msg['content']}")
    return "\n\n".join(formatted)


def format_tool_results_for_conversation(tool_results: list) -> str:
    """Format tool execution results for conversation display."""
    if not tool_results:
        return ""
    formatted = []
    for result in tool_results:
        if hasattr(result, "success"):
            status = "✅ Success" if result.success else "❌ Failed"
            formatted.append(f"**Tool Call {result.call_id}** {status}")
            if result.output:
                formatted.append(f"Output: {result.output}")
            if result.error:
                formatted.append(f"Error: {result.error}")
        else:
            formatted.append(f"Result: {result!s}")
        formatted.append("")
    return "\n".join(formatted)
