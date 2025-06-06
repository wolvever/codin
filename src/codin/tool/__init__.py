"""Tool system for codin agents.

This module provides the core tool infrastructure including tool definitions,
execution, registry, and MCP (Model Context Protocol) integration.
"""

import typing as _t

from .base import Tool, ToolContext, ToolSpec, Toolset  # type: ignore F401 – re-export
from .core_tools import (
    # All core tools
    CodebaseSearchTool,
    CoreToolset,
    DeleteFileTool,
    EditFileTool,
    FetchTool,
    FileSearchTool,
    GrepSearchTool,
    ListDirTool,
    ReadFileTool,
    ReapplyTool,
    RunShellTool,
    SearchReplaceTool,
    WebSearchTool,
)
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
from .sandbox import (
    SandboxTool,
    SandboxToolset,
)


__all__ = [
    # Core tool system
    'Tool',
    'Toolset',
    'ToolContext',
    'ToolSpec',
    'ToolRegistry',
    'ToolRegistryConfig',
    'ToolEndpoint',
    'ToolExecutor',
    # Generic tool support
    'GenericTool',
    'create_tool_from_function',
    'tool',
    'ToolDecorator',
    # All core tools
    'CodebaseSearchTool',
    'ReadFileTool',
    'RunShellTool',
    'ListDirTool',
    'GrepSearchTool',
    'EditFileTool',
    'SearchReplaceTool',
    'FileSearchTool',
    'DeleteFileTool',
    'ReapplyTool',
    'WebSearchTool',
    'FetchTool',
    'CoreToolset',
    # Sandbox tools
    'SandboxTool',
    'SandboxToolset',
    # MCP tool support
    'MCPTool',
    'MCPToolset',
    'MCPSessionManager',
    'HttpServerParams',
    'StdioServerParams',
    'SseServerParams',
]
