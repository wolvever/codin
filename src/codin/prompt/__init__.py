"""Elegant and concise prompt system.

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

# Core API - Simple and elegant
from .run import prompt_run, render_only, set_endpoint

# Base types for advanced usage
from .base import (
    # A2A Protocol - now using a2a SDK
    A2ARole, A2ATextPart, A2AFilePart, A2ADataPart, 
    A2AMessage, PromptResponse,
    # Tools
    ToolDefinition,
    # Templates
    PromptTemplate, PromptVariant, RenderedPrompt,
    ModelOptions
)

# Engine and registry for advanced usage
from .engine import PromptEngine
from .registry import PromptRegistry, get_registry

__all__ = [
    # Primary API - Use these for most cases
    "prompt_run",
    "render_only", 
    "set_endpoint",
    
    # A2A Protocol Types
    "A2ARole",
    "A2ATextPart", 
    "A2AFilePart",
    "A2ADataPart",
    "A2AMessage",
    "PromptResponse",
    
    # Tool Definition
    "ToolDefinition",
    
    # Template Types
    "PromptTemplate",
    "PromptVariant", 
    "RenderedPrompt",
    "ModelOptions",
    
    # Advanced Usage
    "PromptEngine",
    "PromptRegistry",
    "get_registry",
]


# Convenience imports for backward compatibility
def get_template_registry(endpoint: str | None = None) -> PromptRegistry:
    """Get a template registry instance (convenience function)."""
    return get_registry(endpoint)


def create_prompt_engine(llm=None, endpoint: str | None = None) -> PromptEngine:
    """Create a prompt engine instance (convenience function)."""
    return PromptEngine(llm=llm, endpoint=endpoint) 