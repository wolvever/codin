from __future__ import annotations

"""Connection parameters for MCP servers.

This module provides the parameter classes needed to establish connections
to MCP servers using various protocols (HTTP, stdio, SSE).
"""

import typing as _t
from dataclasses import dataclass, field

__all__ = [
    "HttpServerParams",
    "StdioServerParams",
    "SseServerParams",
]


@dataclass(slots=True)
class HttpServerParams:
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
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0


@dataclass(slots=True)
class StdioServerParams:
    """Parameters for stdio connections to an MCP server.

    Parameters
    ----------
    command:
        The command to execute (e.g., "npx", "python").
    args:
        Command line arguments for the command.
    env:
        Environment variables to set for the command.
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SseServerParams:
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
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 5.0
    sse_read_timeout: float = 300.0  # 5 minutes default 