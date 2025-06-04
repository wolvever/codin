from __future__ import annotations

"""Tool implementation for calling MCP tools via various protocols.

This module provides the MCPTool class, which implements the Tool interface
for Model Context Protocol (MCP) tools. It supports various protocols for
communicating with MCP servers, including HTTP, stdio, and SSE.
"""

import logging
import typing as _t

import pydantic as _pyd

from ..base import Tool, ToolContext, LifecycleState
from .session_manager import MCPSessionManager
from .utils import retry_on_closed_resource

__all__ = [
    "MCPTool",
]

_logger = logging.getLogger(__name__)


class _DefaultInputModel(_pyd.BaseModel):
    """Fallback schema that accepts *any* input."""

    class Config:
        extra = "allow"


class MCPTool(Tool):
    """A Tool implementation for remote MCP tools.
    
    This class wraps an MCP tool, allowing it to be used like any other
    tool in the codin system. It communicates with the MCP server using
    the provided session manager.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        session_manager: MCPSessionManager,
        input_schema: _t.Type[_pyd.BaseModel] | None = None,
        is_generative: bool = False,
    ) -> None:
        """Initialize an MCPTool.
        
        Parameters
        ----------
        name:
            Name of the tool on the MCP server
        description:
            Human-readable description of the tool
        session_manager:
            Session manager instance for communicating with the MCP server
        input_schema:
            Optional Pydantic model for validating tool inputs
        is_generative:
            Whether this tool produces streaming/generative output
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=input_schema or _DefaultInputModel,
            is_generative=is_generative,
        )
        self._session_manager = session_manager
        
    async def initialize(self) -> None:
        """Initialize the MCP tool by checking connection to the server."""
        try:
            # Test the connection by attempting to list tools
            # This will verify the session manager is working
            await self._session_manager.list_tools()
            self._state = LifecycleState.UP
        except Exception as e:
            _logger.error(f"Failed to initialize MCP tool {self.name}: {e}")
            self._state = LifecycleState.ERROR
            raise
    
    async def cleanup(self) -> None:
        """Cleanup the MCP tool."""
        self._state = LifecycleState.DISCONNECTED
        
    @retry_on_closed_resource("_reinitialize_session")
    async def run(
        self, args: dict[str, _t.Any], tool_context: ToolContext
    ) -> _t.Any:
        """Call the MCP tool via the session manager.
        
        Parameters
        ----------
        args:
            Arguments to pass to the tool
        tool_context:
            Context information (unused by MCP tools)
            
        Returns
        -------
        Any:
            The JSON response from the tool
        """
        _logger.debug("Calling MCP tool '%s'", self.name)
        try:
            result = await self._session_manager.call_tool(self.name, args)
            return result
        except Exception as e:
            _logger.error("Error calling MCP tool '%s': %s", self.name, e)
            raise
            
    async def _reinitialize_session(self) -> None:
        """Reestablish session if connection was lost."""
        # Since we're using abstract session managers, we don't need to do
        # anything special here. The retry decorator will naturally retry the
        # operation, which will trigger the session's automatic reconnection.
        pass 