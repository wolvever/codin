"""MCP server connection parameters for codin agents.

This module defines connection parameter classes for different MCP
server connection types including HTTP, stdio, and SSE.
"""


from pydantic import BaseModel, Field


__all__ = [
    'HttpServerParams',
    'SseServerParams',
    'StdioServerParams',
]


class HttpServerParams(BaseModel):
    """Parameters for HTTP connections to an MCP server.

    Parameters
    ----------
    base_url:
        Base URL of the MCP server, e.g. ``"https://mcp.example.com"``.
    headers:
        Optional headers to include in HTTP requests.
    timeout:
        Connection and request timeout in seconds.
    """

    base_url: str
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 30.0


class StdioServerParams(BaseModel):
    """Parameters for stdio connections to an MCP server.

    Parameters
    ----------
    command:
        The command to execute (e.g., "npx", "python").

    Args:
        Command line arguments for the command.
    env:
        Environment variables to set for the command.
    """

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class SseServerParams(BaseModel):
    """Parameters for Server-Sent Events (SSE) connections to an MCP server.

    Parameters
    ----------
    url:
        URL of the SSE endpoint.
    headers:
        Optional headers to include in HTTP requests.
    timeout:
        Connection timeout in seconds.
    sse_read_timeout:
        Timeout for reading SSE events in seconds.
    """

    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 5.0
    sse_read_timeout: float = 300.0  # 5 minutes default
