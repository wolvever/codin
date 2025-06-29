"""Utility helpers for working with A2A messages."""

from __future__ import annotations

__all__ = [
    "extract_text_from_message",
    "format_history_for_prompt",
    "format_tool_results_for_conversation",
]


def extract_text_from_message(message: object) -> str:
    """Extract plain text content from a message object."""
    if message is None:
        return ""

    if isinstance(message, str):
        return message

    # Support simple dict representations used in tests
    if isinstance(message, dict):
        if "content" in message:
            return message["content"]
        if "parts" in message:
            return "\n".join(p.get("text", "") for p in message["parts"])

    # Pydantic Message models have ``parts`` with TextPart objects
    parts = getattr(message, "parts", None)
    if parts:
        texts = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
        if texts:
            return "\n".join(texts)

    # Fallback to ``content`` attribute
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    return ""


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
