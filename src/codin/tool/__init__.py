"""Tool system for codin agents.

This module provides the core tool infrastructure including tool definitions,
execution, registry, and MCP (Model Context Protocol) integration.
"""


from .base import Tool, ToolContext, Toolset, ToolSpec  # type: ignore F401 – re-export
# FetchTool import removed: from .core_tools import FetchTool
from .decorators import ToolDecorator, tool
from .executor import ToolExecutor
from .generic import GenericTool, create_tool_from_function

# Import MCP utilities so that callers can do `from codin.tool import MCPTool`.
from .mcp import (
    HttpServerParams,
    MCPSessionManager,
    MCPTool,
    MCPToolset,
    SseServerParams,
    StdioServerParams,
)
from .registry import ToolEndpoint, ToolRegistry, ToolRegistryConfig
from .sandbox_tools import ( # Changed from .sandbox to .sandbox_tools
    SandboxTool,
    SandboxToolset,
)

__all__ = [
    # Core tool system
    "Tool",
    "Toolset",
    "ToolContext",
    "ToolSpec",
    "ToolRegistry",
    "ToolRegistryConfig",
    "ToolEndpoint",
    "ToolExecutor",
    # Generic tool support
    "GenericTool",
    "create_tool_from_function",
    "tool",
    "ToolDecorator",
    # Core tools
    # "FetchTool", # FetchTool removed from __all__
    # Sandbox tools
    "SandboxTool",
    "SandboxToolset",
    # MCP tool support
    "MCPTool",
    "MCPToolset",
    "MCPSessionManager",
    "HttpServerParams",
    "StdioServerParams",
    "SseServerParams",
]
