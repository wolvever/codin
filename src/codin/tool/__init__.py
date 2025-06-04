from __future__ import annotations

"""Tool system subpackage - public re-exports."""

from .base import Tool, Toolset, ToolContext, ToolSpec  # type: ignore F401 – re-export
from .registry import ToolRegistry, ToolRegistryConfig, ToolEndpoint
from .executor import ToolExecutor
from .generic import GenericTool, create_tool_from_function
from .decorators import tool, ToolDecorator
from .sandbox import (
    SandboxTool,
    SandboxToolset,
)
from .core_tools import (
    # All core tools
    CodebaseSearchTool,
    ReadFileTool,
    RunShellTool,
    ListDirTool,
    GrepSearchTool,
    EditFileTool,
    SearchReplaceTool,
    FileSearchTool,
    DeleteFileTool,
    ReapplyTool,
    WebSearchTool,
    FetchTool,
    CoreToolset,
)

# Import MCP utilities so that callers can do `from codin.tool import MCPTool`.
from .mcp import (  # noqa: F401 (lazy import)
    MCPTool,
    MCPToolset,
    MCPSessionManager,
    HttpServerParams,
    StdioServerParams, 
    SseServerParams,
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
    
    # All core tools
    "CodebaseSearchTool",
    "ReadFileTool",
    "RunShellTool",
    "ListDirTool",
    "GrepSearchTool",
    "EditFileTool",
    "SearchReplaceTool",
    "FileSearchTool",
    "DeleteFileTool",
    "ReapplyTool",
    "WebSearchTool",
    "FetchTool",
    "CoreToolset",
    
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