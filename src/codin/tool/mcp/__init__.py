from __future__ import annotations

"""MCP tool support for codin.

This sub-package provides utilities and tool implementations that enable
calling Model Context Protocol (MCP) tools from *codin*'s tool execution
stack. It supports multiple connection protocols:

- HTTP: For standard RESTful API access to MCP servers
- stdio: For launching and communicating with local MCP tools
- SSE: For Server-Sent Events connections to MCP servers
"""

from .session_manager import (
    MCPSessionManager,
    HttpSessionManager,
    StdioSessionManager,
    SseSessionManager,
)
from .mcp_tool import MCPTool
from .toolset import MCPToolset
from .conversion_utils import convert_mcp_to_protocol_types
from .server_connection import (
    HttpServerParams,
    StdioServerParams,
    SseServerParams,
)
from .utils import retry_on_closed_resource

__all__ = [
    # Session management
    "MCPSessionManager",
    "HttpSessionManager", 
    "StdioSessionManager",
    "SseSessionManager",
    
    # Tools
    "MCPTool",
    "MCPToolset",
    
    # Utilities
    "convert_mcp_to_protocol_types",
    "retry_on_closed_resource",
    
    # Connection parameters
    "HttpServerParams",
    "StdioServerParams", 
    "SseServerParams",
] 