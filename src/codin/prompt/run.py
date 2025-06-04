"""Elegant prompt execution API - simple and concise."""

from __future__ import annotations

import typing as _t
import uuid

# Use a2a SDK types directly
from a2a.types import Message, TextPart, Role

from .engine import PromptEngine
from .registry import set_endpoint
from .base import ToolDefinition, PromptResponse

__all__ = ["prompt_run", "render_only", "set_endpoint"]

# Global engine instance for convenience
_engine: PromptEngine | None = None


def _get_engine() -> PromptEngine:
    """Get or create the global prompt engine."""
    global _engine
    if _engine is None:
        _engine = PromptEngine()
    return _engine


def _convert_tools(tools: list[ToolDefinition | dict] | None) -> list[ToolDefinition] | None:
    """Convert tools to ToolDefinition objects."""
    if not tools:
        return None
    
    converted = []
    for tool in tools:
        if isinstance(tool, dict):
            converted.append(ToolDefinition(
                name=tool.get('name', ''),
                description=tool.get('description', ''),
                parameters=tool.get('parameters', {}),
                metadata=tool.get('metadata')
            ))
        else:
            converted.append(tool)
    return converted


def _convert_history(history: list[Message | dict] | None) -> list[Message] | None:
    """Convert history to Message objects."""
    if not history:
        return None
    
    converted = []
    for msg in history:
        if isinstance(msg, dict):
            role = Role.user if msg.get('role') == 'user' else Role.agent
            content = msg.get('content', '')
            
            a2a_msg = Message(
                message_id=msg.get('message_id', str(uuid.uuid4())),
                role=role,
                parts=[TextPart(text=content)],
                context_id=msg.get('context_id'),
                task_id=msg.get('task_id'),
                metadata=msg.get('metadata')
            )
            converted.append(a2a_msg)
        else:
            converted.append(msg)
    return converted


async def prompt_run(
    name: str,
    /,
    version: str | None = None,
    variables: dict[str, _t.Any] | None = None,
    tools: list[ToolDefinition | dict] | None = None,
    conditions: dict[str, _t.Any] | None = None,
    stream: bool = False,
    **kwargs
) -> PromptResponse:
    """Execute a prompt template with elegant simplicity.
    
    Args:
        name: Template name
        version: Template version (optional)
        variables: Template variables
        tools: Available tools
        conditions: Template selection conditions
        stream: Stream response
        **kwargs: Additional variables (including history if needed)
        
    Returns:
        A2AResponse with result
        
    Examples:
        # Simple usage
        response = await prompt_run("summarize", text="Long text...")
        
        # With tools and conditions
        response = await prompt_run(
            "code_assistant",
            tools=[execute_tool, read_file_tool],
            conditions={"model_family": "claude"},
            user_input="Help me debug this code",
            history=conversation_history
        )
    """
    engine = _get_engine()
    
    # Extract history from kwargs if provided
    history = kwargs.pop('history', None)
    
    return await engine.run(
        name,
        version=version,
        variables=variables,
        tools=_convert_tools(tools),
        history=_convert_history(history),
        conditions=conditions,
        stream=stream,
        **kwargs
    )


async def render_only(
    name: str,
    /,
    version: str | None = None,
    variables: dict[str, _t.Any] | None = None,
    conditions: dict[str, _t.Any] | None = None,
    **kwargs
) -> str:
    """Render a template without executing LLM.
    
    Args:
        name: Template name
        version: Template version (optional)
        variables: Template variables
        conditions: Template selection conditions
        **kwargs: Additional variables
        
    Returns:
        Rendered prompt text
    """
    engine = _get_engine()
    
    rendered = await engine.render_only(
        name,
        version=version,
        variables=variables,
        conditions=conditions,
        **kwargs
    )
    return rendered.text 