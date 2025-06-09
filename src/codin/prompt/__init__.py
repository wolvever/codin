"""Prompt management system for codin agents.

This module provides a comprehensive prompt management system including
template storage, rendering, and execution with LLM integration.

This module provides a simple API for prompt templates with:
- Local/remote storage (fs:// or http:// endpoints)
- Template versions and dict-based conditions
- A2A protocol compliance
- Simple API: prompt_run(name, [version], [variables], [tools], stream=True/False)

Quick Start:
    # Set template storage
    set_endpoint("fs://./prompt_templates")

    # Simple usage
    response = await prompt_run("summarize", text="Long text...")

    # With tools and conditions
    response = await prompt_run(
        "code_assistant",
        tools=[my_tool],
        conditions={"model_family": "claude"},
        user_input="Help me code"
    )
"""

from __future__ import annotations

# Base types for advanced usage
from .base import (
    A2ADataPart,
    A2AFilePart,
    A2AMessage,
    # A2A Protocol - now using a2a SDK
    A2ARole,
    A2ATextPart,
    ModelOptions,
    PromptResponse,
    # Templates
    PromptTemplate,
    PromptVariant,
    RenderedPrompt,
    # Tools
    ToolDefinition,
)

# Engine and registry for advanced usage
from .engine import PromptEngine
from .registry import PromptRegistry, get_registry

# Core API - Simple and elegant
from .run import prompt_run, prompt_render, set_endpoint

__all__ = [
    # Primary API - Use these for most cases
    'prompt_run',
    'prompt_render',
    'set_endpoint',
    # A2A Protocol Types
    'A2ARole',
    'A2ATextPart',
    'A2AFilePart',
    'A2ADataPart',
    'A2AMessage',
    'PromptResponse',
    # Tool Definition
    'ToolDefinition',
    # Template Types
    'PromptTemplate',
    'PromptVariant',
    'RenderedPrompt',
    'ModelOptions',
    # Advanced Usage
    'PromptEngine',
    'PromptRegistry',
    'get_registry',
]


# Convenience imports for backward compatibility
def get_template_registry(endpoint: str | None = None) -> PromptRegistry:
    """Get a template registry instance (convenience function)."""
    return get_registry(endpoint)


def create_prompt_engine(llm=None, endpoint: str | None = None) -> PromptEngine:
    """Create a prompt engine instance (convenience function)."""
    return PromptEngine(llm=llm, endpoint=endpoint)
