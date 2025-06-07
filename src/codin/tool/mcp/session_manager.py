"""MCP session management for codin agents.

This module provides session managers for different MCP connection types,
including HTTP, stdio and Server-Sent Events (SSE) connections. The
:class:`MCPSessionManager` maintains a connection to an MCP server using one of
several protocols and exposes a consistent interface to the rest of the system.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import sys
import typing as _t
from contextlib import AsyncExitStack

import httpx

from .server_connection import (
    HttpServerParams,
    SseServerParams,
    StdioServerParams,
)

try:
    # Make MCP imports optional so codin can work without special imports
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client

    HAS_MCP_CLIENT = True
except ImportError:
    HAS_MCP_CLIENT = False

__all__ = [
    'HttpSessionManager',
    'MCPSessionManager',
    'SseSessionManager',
    'StdioSessionManager',
]

_logger = logging.getLogger(__name__)


class MCPSessionManager(abc.ABC):
    """Abstract base class for MCP session managers.

    This class defines the common interface that all session managers must
    implement, regardless of the underlying protocol used to communicate
    with the MCP server.
    """

    @abc.abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict[str, _t.Any]) -> _t.Any:
        """Call a tool on the MCP server.

        Parameters
        ----------
        tool_name:
            Name of the tool to call
        arguments:
            Dictionary of arguments to pass to the tool

        Returns:
        -------
        Any:
            The result of the tool call, typically a JSON object
        """

    @abc.abstractmethod
    async def list_tools(self) -> list[dict[str, _t.Any]]:
        """Get a list of available tools from the MCP server.

        Returns:
        -------
        list[dict]:
            List of tool descriptions from the server
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the session and release resources."""

    @classmethod
    def create(
        cls, connection_params: HttpServerParams | StdioServerParams | SseServerParams, **kwargs
    ) -> MCPSessionManager:
        """Factory method to create the appropriate session manager.

        Parameters
        ----------
        connection_params:
            Connection parameters for the desired protocol
        **kwargs:
            Additional arguments passed to the specific session manager

        Returns:
        -------
        MCPSessionManager:
            An instance of the appropriate session manager subclass

        Raises:
        ------
        ValueError:
            If the connection parameters type is not recognized
        """
        if isinstance(connection_params, HttpServerParams):
            return HttpSessionManager(connection_params, **kwargs)
        if isinstance(connection_params, StdioServerParams):
            if not HAS_MCP_CLIENT:
                raise ImportError("To use stdio connections, install the 'mcp' package.")
            return StdioSessionManager(connection_params, **kwargs)
        if isinstance(connection_params, SseServerParams):
            if not HAS_MCP_CLIENT:
                raise ImportError("To use SSE connections, install the 'mcp' package.")
            return SseSessionManager(connection_params, **kwargs)
        raise ValueError(f'Unsupported connection parameters type: {type(connection_params)}')


class HttpSessionManager(MCPSessionManager):
    """Session manager for HTTP-based MCP servers.

    This implementation uses httpx to communicate with MCP servers that
    expose a REST API interface.
    """

    def __init__(
        self,
        params: HttpServerParams,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize the HTTP session manager.

        Parameters
        ----------
        params:
            HTTP connection parameters
        transport:
            Optional httpx transport (mainly for testing)
        """
        self._params = params
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        """Return a configured HTTP client.

        The client is lazily instantiated and reused across calls.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._params.base_url.rstrip('/'),
                headers=self._params.headers,
                timeout=self._params.timeout,
                transport=self._transport,
            )
        return self._client

    async def call_tool(self, tool_name: str, arguments: dict[str, _t.Any]) -> _t.Any:
        """Call a tool via HTTP POST.

        Parameters
        ----------
        tool_name:
            Name of the tool to call
        arguments:
            Dictionary of arguments to pass to the tool

        Returns:
        -------
        Any:
            The parsed JSON response

        Raises:
        ------
        httpx.HTTPStatusError:
            If the server returns an error status code
        """
        client = await self.get_client()
        endpoint = f'/tools/{tool_name}'

        _logger.debug("Calling MCP tool '%s' via HTTP", tool_name)
        response = await client.post(endpoint, json=arguments)
        response.raise_for_status()

        return response.json()

    async def list_tools(self) -> list[dict[str, _t.Any]]:
        """Get a list of available tools via HTTP GET.

        Returns:
        -------
        list[dict]:
            List of tool descriptions
        """
        client = await self.get_client()
        response = await client.get('/tools')
        response.raise_for_status()

        data = response.json()
        return data.get('tools', [])

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class StdioSessionManager(MCPSessionManager):
    """Session manager for stdio-based MCP servers.

    This implementation launches a subprocess and communicates with it
    using the MCP stdio client.
    """

    def __init__(
        self,
        params: StdioServerParams,
        *,
        errlog: _t.TextIO = sys.stderr,
    ) -> None:
        """Initialize the stdio session manager.

        Parameters
        ----------
        params:
            stdio connection parameters
        errlog:
            Stream for stderr output from the subprocess
        """
        if not HAS_MCP_CLIENT:
            raise ImportError("To use stdio connections, install the 'mcp' package.")

        self._params = params
        self._errlog = errlog
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def _ensure_session(self) -> ClientSession:
        """Ensure a session exists, creating one if necessary."""
        if self._session is None:
            # Create the MCP StdioServerParameters
            server_params = StdioServerParameters(
                command=self._params.command,
                args=self._params.args,
            )

            # Create the client and session using the correct MCP SDK pattern
            stdio_context = stdio_client(server_params)
            read_stream, write_stream = await self._exit_stack.enter_async_context(stdio_context)

            session_context = ClientSession(read_stream, write_stream)
            self._session = await self._exit_stack.enter_async_context(session_context)
            await self._session.initialize()

        return self._session

    async def call_tool(self, tool_name: str, arguments: dict[str, _t.Any]) -> _t.Any:
        """Call a tool using the MCP session.

        Parameters
        ----------
        tool_name:
            Name of the tool to call
        arguments:
            Dictionary of arguments to pass to the tool

        Returns:
        -------
        Any:
            The result from the tool
        """
        session = await self._ensure_session()
        _logger.debug("Calling MCP tool '%s' via stdio", tool_name)
        return await session.call_tool(tool_name, arguments=arguments)

    async def list_tools(self) -> list[dict[str, _t.Any]]:
        """Get a list of available tools.

        Returns:
        -------
        list[dict]:
            List of tool descriptions
        """
        session = await self._ensure_session()
        result = await session.list_tools()
        return result.tools

    async def close(self) -> None:
        """Close the session and terminate the subprocess."""
        # Mark as closed immediately to prevent further use
        session_to_close = self._session
        exit_stack_to_close = self._exit_stack
        self._session = None
        self._exit_stack = AsyncExitStack()  # Create new one for safety

        # Use a more robust cleanup approach
        import sys
        import warnings

        # Completely suppress all warnings and errors during cleanup
        original_stderr = sys.stderr
        original_showwarning = warnings.showwarning

        try:
            # Redirect all output to devnull
            devnull = open(os.devnull, 'w')
            sys.stderr = devnull
            warnings.showwarning = lambda *args, **kwargs: None

            # Close session first with timeout
            if session_to_close:
                try:
                    if hasattr(session_to_close, 'close'):
                        await asyncio.wait_for(session_to_close.close(), timeout=0.5)
                except Exception:
                    pass  # Ignore all errors

            # Close exit stack with improved error handling
            if exit_stack_to_close:
                try:
                    # Create a shield to prevent cancellation during cleanup
                    async def protected_cleanup():
                        try:
                            await exit_stack_to_close.aclose()
                        except Exception:
                            # Force close any remaining resources
                            if hasattr(exit_stack_to_close, '_exit_callbacks'):
                                exit_stack_to_close._exit_callbacks.clear()

                    # Run with shield to prevent cancellation, but also suppress task warnings
                    cleanup_task = asyncio.create_task(
                        asyncio.shield(asyncio.wait_for(protected_cleanup(), timeout=1.0))
                    )

                    # Add a done callback to suppress any remaining exceptions
                    def suppress_task_exception(task):
                        try:
                            task.result()
                        except Exception:
                            pass  # Completely suppress all exceptions

                    cleanup_task.add_done_callback(suppress_task_exception)

                    try:
                        await cleanup_task
                    except Exception:
                        pass  # Ignore all cleanup errors

                except (TimeoutError, asyncio.CancelledError):
                    # If cleanup times out or is cancelled, force cleanup
                    try:
                        if hasattr(exit_stack_to_close, '_exit_callbacks'):
                            exit_stack_to_close._exit_callbacks.clear()
                    except Exception:
                        pass
                except Exception:
                    # For any other error, just abandon the cleanup
                    pass

        except Exception:
            pass  # Ignore all errors
        finally:
            # Always restore stderr
            if sys.stderr != original_stderr:
                try:
                    sys.stderr.close()
                except Exception:
                    pass
            sys.stderr = original_stderr
            warnings.showwarning = original_showwarning


class SseSessionManager(MCPSessionManager):
    """Session manager for SSE-based MCP servers.

    This implementation uses the MCP SSE client to communicate with
    MCP servers that expose a Server-Sent Events interface.
    """

    def __init__(self, params: SseServerParams) -> None:
        """Initialize the SSE session manager.

        Parameters
        ----------
        params:
            SSE connection parameters
        """
        if not HAS_MCP_CLIENT:
            raise ImportError("To use SSE connections, install the 'mcp' package.")

        self._params = params
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def _ensure_session(self) -> ClientSession:
        """Ensure a session exists, creating one if necessary."""
        if self._session is None:
            # Create the SSE client using the correct MCP SDK pattern
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                sse_client(
                    url=self._params.url,
                    headers=self._params.headers,
                    timeout=self._params.timeout,
                    sse_read_timeout=self._params.sse_read_timeout,
                )
            )
            self._session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await self._session.initialize()

        return self._session

    async def call_tool(self, tool_name: str, arguments: dict[str, _t.Any]) -> _t.Any:
        """Call a tool using the MCP session.

        Parameters
        ----------
        tool_name:
            Name of the tool to call
        arguments:
            Dictionary of arguments to pass to the tool

        Returns:
        -------
        Any:
            The result from the tool
        """
        session = await self._ensure_session()
        _logger.debug("Calling MCP tool '%s' via SSE", tool_name)
        return await session.call_tool(tool_name, arguments=arguments)

    async def list_tools(self) -> list[dict[str, _t.Any]]:
        """Get a list of available tools.

        Returns:
        -------
        list[dict]:
            List of tool descriptions
        """
        session = await self._ensure_session()
        result = await session.list_tools()
        return result.tools

    async def close(self) -> None:
        """Close the session."""
        try:
            # First try to close the session gracefully
            if self._session:
                try:
                    # Try to close the session first
                    if hasattr(self._session, 'close'):
                        await self._session.close()
                except Exception as e:
                    _logger.debug(f'Error closing session gracefully: {e}')
                finally:
                    self._session = None

            # Then close the exit stack which will clean up the SSE client
            if self._exit_stack:
                try:
                    await self._exit_stack.aclose()
                except RuntimeError as e:
                    # Handle the specific "Attempted to exit cancel scope in a different task" error
                    if 'Attempted to exit cancel scope in a different task' in str(e):
                        _logger.debug(f'Task mismatch during MCP cleanup (expected on shutdown): {e}')
                    else:
                        _logger.warning(f'RuntimeError during exit stack cleanup: {e}')
                except Exception as e:
                    _logger.debug(f'Error closing exit stack: {e}')
        except Exception as e:
            _logger.debug(f'Error during SSE session cleanup: {e}')
        finally:
            self._session = None
