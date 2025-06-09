"""Custom exceptions for MCP related errors."""

from __future__ import annotations
import typing as _t

class MCPError(Exception):
    """Base class for MCP related errors."""
    pass

class MCPConnectionError(MCPError):
    """For issues related to establishing or maintaining a connection."""
    pass

class MCPTimeoutError(MCPConnectionError):
    """For timeout specific errors."""
    pass

class MCPProtocolError(MCPError):
    """For errors in MCP communication.

    Examples:
        - Invalid JSON
        - Unexpected response structure
        - Pydantic validation failures on responses
    """
    def __init__(self, message: str, underlying_error: Exception | None = None):
        super().__init__(message)
        self.underlying_error = underlying_error

class MCPHttpError(MCPConnectionError):
    """For HTTP specific errors."""
    def __init__(self, message: str, *, status_code: int | None = None, response_text: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text

    def __str__(self) -> str:
        s = super().__str__()
        if self.status_code is not None:
            s += f" (Status Code: {self.status_code})"
        if self.response_text:
            s += f" Response: {self.response_text[:100]}..." # Truncate long responses
        return s

class MCPToolError(Exception):
    """Base class for errors specific to MCPTool operation."""
    pass

class MCPInputError(MCPToolError, ValueError):
    """For issues with tool input arguments when calling MCP tools or methods."""
    pass

__all__ = [
    "MCPError",
    "MCPConnectionError",
    "MCPTimeoutError",
    "MCPProtocolError",
    "MCPHttpError",
    "MCPToolError",
    "MCPInputError",
]
