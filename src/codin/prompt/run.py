"""Prompt execution utilities for codin agents.

This module provides high-level utilities for executing prompts
with automatic template discovery and LLM integration.

Elegant prompt execution API - simple and concise.
"""

from __future__ import annotations

import typing as _t
import uuid

# Use agent protocol types directly
from codin.agent.types import Message, Role, TextPart

from .base import PromptResponse, ToolDefinition
from .engine import PromptEngine
from .registry import set_endpoint
from typing import Any # Added for type hinting

__all__ = ["prompt_run", "set_endpoint", "prompt_render"]

# Global engine instance for convenience
# _engine is used by _get_engine(), which is used by prompt_run.
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
            converted.append(
                ToolDefinition(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    parameters=tool.get("parameters", {}),
                    metadata=tool.get("metadata"),
                )
            )
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
            role = Role.user if msg.get("role") == "user" else Role.agent
            content = msg.get("content", "")

            a2a_msg = Message(
                message_id=msg.get("message_id", str(uuid.uuid4())),
                role=role,
                parts=[TextPart(text=content)],
                context_id=msg.get("context_id"),
                task_id=msg.get("task_id"),
                metadata=msg.get("metadata"),
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
    **kwargs,
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

    # Prepare variables for engine.run, which expects tools and history within its 'variables' dict
    engine_run_vars = (variables or {}).copy()

    processed_tools = _convert_tools(tools)
    if processed_tools is not None: # _convert_tools returns None if tools is None
        engine_run_vars["tools"] = processed_tools

    # Extract history from kwargs if provided, then add to engine_run_vars
    history_data = kwargs.pop("history", None)
    if history_data is not None:
        processed_history = _convert_history(history_data)
        if processed_history is not None: # _convert_history returns None if history_data is None
             engine_run_vars["history"] = processed_history

    # Note: context_id and task_id are not explicitly handled here in prompt_run wrapper.
    # If they need to be passed to engine.run, they should be part of the 'variables'
    # dict passed to prompt_run, or part of **kwargs. engine.run will then extract them
    # from its 'variables' parameter (which is 'engine_run_vars' here).

    return await engine.run(
        name,
        version=version,
        variables=engine_run_vars, # Pass the combined variables
        conditions=conditions,
        stream=stream,
        **kwargs, # Pass remaining kwargs
    )

# The render_only function has been deleted.

async def prompt_render(
    name: str,
    /,
    version: str | None = None,
    variables: dict[str, Any] | None = None,
    conditions: dict[str, Any] | None = None,
    **kwargs,
) -> str:
    """Render a template text without executing LLM, using a fresh engine.

    This function is specifically for rendering the text of a prompt
    and guarantees that no LLM is initialized or used in the process.
    It instantiates a fresh PromptEngine for rendering.

    Args:
        name: The name of the prompt template to render.
        version: Optional. The specific version of the template to use.
            If None, the latest version is typically used.
        variables: Optional. A dictionary of variables to be interpolated
            into the prompt template.
        conditions: Optional. A dictionary of conditions used to select
            the appropriate variant of the prompt template.
        **kwargs: Additional keyword arguments passed to the underlying
            `engine.render` call.

    Returns:
        str: The rendered text content of the prompt template.

    Example:
        >>> variables = {"user_input": "Can you explain prompt engineering?"}
        >>> conditions = {"model_family": "claude"}
        >>> prompt_text = await prompt_render(
        ...     "explain_concept",
        ...     variables=variables,
        ...     conditions=conditions
        ... )
        >>> print(prompt_text)
    """
    # Instantiate a new PromptEngine without any LLM
    engine = PromptEngine()

    # PromptEngine.render_only was renamed to render and now returns str
    rendered_text = await engine.render(
        name,
        version=version,
        variables=variables,
        conditions=conditions,
        **kwargs,
    )
    return rendered_text
