"""MCP toolset implementation for codin agents.

This module provides a toolset that dynamically loads and manages
tools from MCP servers, enabling seamless integration with external
tool providers.
"""

import asyncio
import logging
import os
import sys
import typing as _t
import warnings

from ..base import LifecycleState, Tool, Toolset
from .mcp_tool import MCPTool
from .server_connection import (
    HttpServerParams,
    SseServerParams,
    StdioServerParams,
)
from .session_manager import MCPSessionManager


__all__ = [
    'MCPToolset',
]

_logger = logging.getLogger(__name__)


class MCPToolset(Toolset):
    """Collection of MCP tools from a single server.

    This class discovers and instantiates all tools available from an MCP
    server, exposing them as a codin Toolset.

    Example:
    -------
    ```python
    # HTTP connection
    http_toolset = MCPToolset(
        name='Web MCP Tools',
        description='Tools from a remote web server',
        connection_params=HttpServerParams(base_url='https://mcp.example.com'),
    )

    # stdio connection
    stdio_toolset = MCPToolset(
        name='Node MCP Tools',
        description='MCP tools from a Node.js process',
        connection_params=StdioServerParams(command='npx', args=['-y', '@modelcontextprotocol/server-filesystem']),
    )

    # Register with a tool registry
    registry = ToolRegistry()
    registry.register_toolset(http_toolset)
    ```
    """

    def __init__(
        self,
        name: str,
        description: str,
        connection_params: HttpServerParams | StdioServerParams | SseServerParams,
        *,
        tool_filter: _t.Callable[[dict[str, _t.Any]], bool] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the MCP toolset.

        Parameters
        ----------
        name:
            Name for this toolset
        description:
            Human-readable description
        connection_params:
            Parameters for connecting to the MCP server
        tool_filter:
            Optional filter function to select which tools to include
        **kwargs:
            Additional arguments passed to the session manager
        """
        super().__init__(name=name, description=description, tools=[])
        self._session_manager = MCPSessionManager.create(connection_params, **kwargs)
        self._tool_filter = tool_filter
        self._initialized = False

    def _should_include_tool(self, tool_info: dict[str, _t.Any]) -> bool:
        """Determine if a tool should be included based on the filter."""
        if self._tool_filter is None:
            return True
        return self._tool_filter(tool_info)

    async def initialize(self) -> None:
        """Initialize the toolset by fetching tools from the server.

        This method is called automatically the first time get_tools() is called.
        You can call it explicitly if you want to control when the tools are
        fetched.
        """
        if self._initialized:
            return

        try:
            tools_info = await self._session_manager.list_tools()
            _logger.info(f'Discovered {len(tools_info)} MCP tools')

            tools = []
            for tool_info in tools_info:
                # Handle both dictionary and object formats for compatibility with tests and real MCP
                if isinstance(tool_info, dict):
                    # Dictionary format (from tests or some MCP implementations)
                    name = tool_info.get('name', '')
                    description = tool_info.get('description', '')
                    input_schema = tool_info.get('inputSchema', {})
                else:
                    # Object format (from real MCP client library)
                    name = getattr(tool_info, 'name', '')
                    description = getattr(tool_info, 'description', '') or ''
                    input_schema = getattr(tool_info, 'inputSchema', {})

                # Convert to dict for filter compatibility
                tool_dict = {
                    'name': name,
                    'description': description,
                    'inputSchema': input_schema,
                }

                if self._should_include_tool(tool_dict):
                    # Create the tool
                    tool = MCPTool(
                        name=name,
                        description=description,
                        session_manager=self._session_manager,
                    )

                    # Initialize the tool
                    await tool.initialize()
                    tools.append(tool)

            self.tools = tools
            self._tool_map = {tool.name: tool for tool in self.tools}
            self._initialized = True
            self._state = LifecycleState.UP  # MCP toolsets are up when connected

        except Exception as e:
            _logger.error(f'Failed to initialize MCP toolset {self.name}: {e}')
            self._state = LifecycleState.ERROR
            raise

    async def _up(self) -> None:
        """Bring up the MCP toolset by initializing and connecting to the server."""
        if not self._initialized:
            await self.initialize()
        # Call parent _up to bring up all discovered tools
        await super()._up()

    async def _down(self) -> None:
        """Bring down the MCP toolset and close connections."""
        await self.cleanup()

    async def cleanup(self) -> None:
        """Cleanup the MCP toolset and close connections."""
        # Save original handlers
        original_stderr = sys.stderr
        original_showwarning = warnings.showwarning
        original_excepthook = sys.excepthook

        try:
            # Completely suppress all output and exceptions during cleanup
            devnull = open(os.devnull, 'w')
            sys.stderr = devnull
            warnings.showwarning = lambda *args, **kwargs: None
            sys.excepthook = lambda *args, **kwargs: None

            # Suppress all warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                # Cleanup all tools with complete error suppression and timeout
                cleanup_tasks = []
                for tool in self.tools:

                    async def cleanup_tool(t):
                        try:
                            await asyncio.wait_for(t.cleanup(), timeout=0.5)
                        except Exception:
                            pass

                    cleanup_tasks.append(cleanup_tool(tool))

                # Run all tool cleanups concurrently with overall timeout
                if cleanup_tasks:
                    try:
                        await asyncio.wait_for(asyncio.gather(*cleanup_tasks, return_exceptions=True), timeout=2.0)
                    except Exception:
                        pass

                # Close session manager with improved error handling
                if self._session_manager:
                    try:
                        # Use shield to prevent cancellation during cleanup
                        await asyncio.shield(asyncio.wait_for(self._session_manager.close(), timeout=1.0))
                    except Exception:
                        # If normal cleanup fails, abandon the session manager
                        pass
                    finally:
                        self._session_manager = None

            self._state = LifecycleState.DISCONNECTED

        except Exception:
            self._state = LifecycleState.ERROR
            # Don't re-raise cleanup errors
        finally:
            # Always restore handlers
            if sys.stderr != original_stderr:
                try:
                    sys.stderr.close()
                except:
                    pass
            sys.stderr = original_stderr
            warnings.showwarning = original_showwarning
            sys.excepthook = original_excepthook

    async def get_tools(self) -> list[Tool]:
        """Get all tools from the MCP server.

        Returns:
        -------
        list[Tool]:
            List of MCPTool objects
        """
        if not self._initialized:
            await self.initialize()
        return self.tools

    async def close(self) -> None:
        """Close the session manager."""
        await self.cleanup()
