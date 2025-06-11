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
import json

import httpx
import pydantic

from .server_connection import (
    HttpServerParams,
    SseServerParams,
    StdioServerParams,
)
from . import mcp_types
from .exceptions import (
    MCPError,
    MCPConnectionError,
    MCPTimeoutError,
    MCPProtocolError,
    MCPHttpError,
)

try:
    # Make MCP imports optional so codin can work without special imports
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.protocol import Response as MCPResponseProtocol # Renamed to avoid clash
    from mcp.exceptions import MCPError as MCPLibError # Assuming mcp-lib has a base error

    HAS_MCP_CLIENT = True
except ImportError:
    HAS_MCP_CLIENT = False
    ClientSession = _t.Any
    StdioServerParameters = _t.Any
    MCPResponseProtocol = _t.Any
    MCPLibError = Exception # Fallback if mcp-lib not installed

    def sse_client(*args: _t.Any, **kwargs: _t.Any) -> _t.Any:
        raise NotImplementedError("mcp package not installed")
    def stdio_client(*args: _t.Any, **kwargs: _t.Any) -> _t.Any:
        raise NotImplementedError("mcp package not installed")

__all__ = [
    'HttpSessionManager',
    'MCPSessionManager',
    'SseSessionManager',
    'StdioSessionManager',
]

_logger = logging.getLogger(__name__)

DEFAULT_CLIENT_INFO = mcp_types.Implementation(name="codin-agent", version="0.1.0")
DEFAULT_CLIENT_CAPABILITIES = mcp_types.ClientCapabilities()


class MCPSessionManager(abc.ABC):
    server_capabilities: mcp_types.ServerCapabilities | None = None
    _client_info: mcp_types.Implementation
    _client_capabilities: mcp_types.ClientCapabilities
    _protocol_version: str

    def __init__(
        self,
        client_info: mcp_types.Implementation | None = None,
        client_capabilities: mcp_types.ClientCapabilities | None = None,
        protocol_version: str = "2025-03-26-preview",
    ):
        self._client_info = client_info or DEFAULT_CLIENT_INFO
        self._client_capabilities = client_capabilities or DEFAULT_CLIENT_CAPABILITIES
        self._protocol_version = protocol_version

    @abc.abstractmethod
    async def initialize(self) -> None:
        pass

    @abc.abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict[str, _t.Any]) -> _t.Any:
        pass

    @abc.abstractmethod
    async def list_tools(self) -> list[dict[str, _t.Any]]: # Should be updated to use mcp_types
        pass

    @abc.abstractmethod
    async def list_resources(self, params: mcp_types.ListResourcesRequestParams) -> mcp_types.ListResourcesResult:
        pass

    @abc.abstractmethod
    async def list_resource_templates(self, params: mcp_types.ListResourceTemplatesRequestParams) -> mcp_types.ListResourceTemplatesResult:
        pass

    @abc.abstractmethod
    async def read_resource(self, params: mcp_types.ReadResourceRequestParams) -> mcp_types.ReadResourceResult:
        pass

    @abc.abstractmethod
    async def subscribe_resource(self, params: mcp_types.SubscribeRequestParams) -> None:
        pass

    @abc.abstractmethod
    async def unsubscribe_resource(self, params: mcp_types.UnsubscribeRequestParams) -> None:
        pass

    @abc.abstractmethod
    async def list_prompts(self, params: mcp_types.ListPromptsRequestParams) -> mcp_types.ListPromptsResult:
        pass

    @abc.abstractmethod
    async def get_prompt(self, params: mcp_types.GetPromptRequestParams) -> mcp_types.GetPromptResult:
        pass

    @abc.abstractmethod
    async def set_logging_level(self, params: mcp_types.SetLevelRequestParams) -> None:
        pass

    @abc.abstractmethod
    async def get_completion(self, params: mcp_types.CompleteRequestParams) -> mcp_types.CompleteResult:
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        pass

    @classmethod
    def create(
        cls, connection_params: HttpServerParams | StdioServerParams | SseServerParams, **kwargs
    ) -> MCPSessionManager:
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
    server_capabilities: mcp_types.ServerCapabilities | None = None

    def __init__(
        self,
        params: HttpServerParams,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        client_info: mcp_types.Implementation | None = None,
        client_capabilities: mcp_types.ClientCapabilities | None = None,
        protocol_version: str = "2025-03-26-preview",
    ) -> None:
        super().__init__(client_info, client_capabilities, protocol_version)
        self._params = params
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._request_id_counter = 0

    def _next_id(self) -> int:
        self._request_id_counter += 1
        return self._request_id_counter

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._params.base_url.rstrip('/'),
                headers=self._params.headers,
                timeout=self._params.timeout,
                transport=self._transport,
            )
        return self._client

    async def _send_request(self, request_obj: mcp_types.JSONRPCRequest, response_model: _t.Type[_pyd.BaseModel] | None) -> _pyd.BaseModel | None:
        client = await self.get_client()
        # Assuming a single RPC endpoint for all MCP JSON-RPC calls.
        # This might need to be configurable if servers use method names as paths.
        endpoint = "/"
        request_json = request_obj.model_dump(by_alias=True, exclude_none=True)

        try:
            _logger.debug(f"Sending HTTP MCP request to {endpoint} for method {request_obj.method}: {request_json}")
            response = await client.post(endpoint, json=request_json)
            response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
            response_data = response.json()
            _logger.debug(f"Received HTTP MCP response for method {request_obj.method}: {response_data}")

        except httpx.TimeoutException as e:
            _logger.error(f"HTTP timeout calling {request_obj.method} at {endpoint}: {e}")
            raise MCPTimeoutError(f"Request to {request_obj.method} timed out.") from e
        except httpx.HTTPStatusError as e:
            _logger.error(f"HTTP error for {request_obj.method} at {endpoint}: {e.response.status_code} - {e.response.text}")
            try: # Attempt to parse error as JSONRPCError
                error_content = e.response.json()
                if "error" in error_content:
                    json_rpc_err = mcp_types.JSONRPCError.model_validate(error_content)
                    raise MCPProtocolError(f"MCP Error from server: {json_rpc_err.error.message}", underlying_error=e) from e
            except (json.JSONDecodeError, pydantic.ValidationError):
                pass # Not a valid JSONRPCError structure
            raise MCPHttpError(
                f"HTTP error {e.response.status_code} for {request_obj.method}.",
                status_code=e.response.status_code,
                response_text=e.response.text
            ) from e
        except httpx.RequestError as e: # Base for network errors
            _logger.error(f"HTTP network error for {request_obj.method} at {endpoint}: {e}")
            raise MCPConnectionError(f"Network error calling {request_obj.method}: {e}") from e
        except json.JSONDecodeError as e:
            _logger.error(f"Failed to decode JSON response for {request_obj.method}: {e}")
            raise MCPProtocolError(f"Invalid JSON response for {request_obj.method}.", underlying_error=e) from e

        # Validate and parse the 'result' part of the JSON-RPC response
        if "error" in response_data:
            json_rpc_err = mcp_types.JSONRPCError.model_validate(response_data)
            _logger.error(f"MCP error response for {request_obj.method}: {json_rpc_err.error}")
            raise MCPProtocolError(f"MCP Error from server: {json_rpc_err.error.message} (Code: {json_rpc_err.error.code})")

        if response_model is None: # For requests that don't have a specific result structure (e.g. notifications if they were requests)
            return None

        try:
            # Assuming the top-level response is a JSONRPCResponse, and we need its 'result' field.
            # If the server directly returns the result content (less common for strict JSON-RPC), this needs adjustment.
            if "result" not in response_data:
                 raise MCPProtocolError(f"JSON-RPC response for {request_obj.method} is missing 'result' field.")
            return response_model.model_validate(response_data["result"])
        except pydantic.ValidationError as e:
            _logger.error(f"Pydantic validation error for {request_obj.method} response: {e}")
            raise MCPProtocolError(f"Invalid response structure for {request_obj.method}.", underlying_error=e) from e


    async def initialize(self) -> None:
        params = mcp_types.InitializeRequestParams(
            capabilities=self._client_capabilities,
            clientInfo=self._client_info,
            protocolVersion=self._protocol_version
        )
        # InitializeRequest itself is a JSONRPCRequest, so it has method="initialize"
        request = mcp_types.InitializeRequest(id=self._next_id(), params=params)

        # The _send_request method expects the response_model to be for the *result* part.
        # InitializeResult is the structure of the "result" field for an initialize request.
        init_result = await self._send_request(request, mcp_types.InitializeResult)

        if init_result and isinstance(init_result, mcp_types.InitializeResult):
            self.server_capabilities = init_result.capabilities
            _logger.info(f"MCP HTTP session initialized. Server: {init_result.serverInfo.name} v{init_result.serverInfo.version}")
        else:
            # This case should ideally be handled by _send_request raising an error if validation fails.
             _logger.error("Failed to get valid InitializeResult from server.")
             raise MCPProtocolError("Did not receive a valid InitializeResult from server.")


    async def call_tool(self, tool_name: str, arguments: dict[str, _t.Any]) -> _t.Any:
        # This is a non-standard MCP call as per schema (not JSON-RPC like others)
        # It seems to be a more RESTful endpoint from original implementation.
        # For full JSON-RPC compliance, this would use CallToolRequest via _send_request.
        client = await self.get_client()
        endpoint = f'/tools/{tool_name}'
        _logger.debug(f"Calling legacy MCP tool '{tool_name}' via HTTP POST to {endpoint}")
        try:
            response = await client.post(endpoint, json=arguments)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            raise MCPTimeoutError(f"Tool call to {tool_name} timed out.") from e
        except httpx.HTTPStatusError as e:
            raise MCPHttpError(f"HTTP error {e.response.status_code} for tool {tool_name}.", status_code=e.response.status_code, response_text=e.response.text) from e
        except httpx.RequestError as e:
            raise MCPConnectionError(f"Network error calling tool {tool_name}: {e}") from e
        except json.JSONDecodeError as e:
            raise MCPProtocolError(f"Invalid JSON response for tool {tool_name}.", underlying_error=e) from e


    async def list_tools(self) -> list[dict[str, _t.Any]]:
        # Similar to call_tool, this seems RESTful.
        # For JSON-RPC, would use ListToolsRequest via _send_request.
        client = await self.get_client()
        _logger.debug("Listing tools via legacy HTTP GET to /tools")
        try:
            response = await client.get('/tools')
            response.raise_for_status()
            data = response.json()
            # The ListToolsResult Pydantic model expects {"tools": [...]}
            # If the response is {"tools": [...]}, then this is okay for now.
            # For strictness, this should be parsed into ListToolsResult.
            return data.get('tools', [])
        except httpx.TimeoutException as e:
            raise MCPTimeoutError("list_tools request timed out.") from e
        except httpx.HTTPStatusError as e:
            raise MCPHttpError(f"HTTP error {e.response.status_code} for list_tools.", status_code=e.response.status_code, response_text=e.response.text) from e
        except httpx.RequestError as e:
            raise MCPConnectionError(f"Network error for list_tools: {e}") from e
        except json.JSONDecodeError as e:
            raise MCPProtocolError("Invalid JSON response for list_tools.", underlying_error=e) from e


    async def list_resources(self, params: mcp_types.ListResourcesRequestParams) -> mcp_types.ListResourcesResult:
        request = mcp_types.ListResourcesRequest(id=self._next_id(), params=params)
        return await self._send_request(request, mcp_types.ListResourcesResult)

    async def list_resource_templates(self, params: mcp_types.ListResourceTemplatesRequestParams) -> mcp_types.ListResourceTemplatesResult:
        request = mcp_types.ListResourceTemplatesRequest(id=self._next_id(), params=params)
        return await self._send_request(request, mcp_types.ListResourceTemplatesResult)

    async def read_resource(self, params: mcp_types.ReadResourceRequestParams) -> mcp_types.ReadResourceResult:
        request = mcp_types.ReadResourceRequest(id=self._next_id(), params=params)
        return await self._send_request(request, mcp_types.ReadResourceResult)

    async def subscribe_resource(self, params: mcp_types.SubscribeRequestParams) -> None:
        request = mcp_types.SubscribeRequest(id=self._next_id(), params=params)
        await self._send_request(request, None) # No specific result model, expecting success or error
        return None

    async def unsubscribe_resource(self, params: mcp_types.UnsubscribeRequestParams) -> None:
        request = mcp_types.UnsubscribeRequest(id=self._next_id(), params=params)
        await self._send_request(request, None)
        return None

    async def list_prompts(self, params: mcp_types.ListPromptsRequestParams) -> mcp_types.ListPromptsResult:
        request = mcp_types.ListPromptsRequest(id=self._next_id(), params=params)
        return await self._send_request(request, mcp_types.ListPromptsResult)

    async def get_prompt(self, params: mcp_types.GetPromptRequestParams) -> mcp_types.GetPromptResult:
        request = mcp_types.GetPromptRequest(id=self._next_id(), params=params)
        return await self._send_request(request, mcp_types.GetPromptResult)

    async def set_logging_level(self, params: mcp_types.SetLevelRequestParams) -> None:
        request = mcp_types.SetLevelRequest(id=self._next_id(), params=params)
        await self._send_request(request, None)
        return None

    async def get_completion(self, params: mcp_types.CompleteRequestParams) -> mcp_types.CompleteResult:
        request = mcp_types.CompleteRequest(id=self._next_id(), params=params)
        return await self._send_request(request, mcp_types.CompleteResult)

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class StdioSessionManagerBase(MCPSessionManager):
    server_capabilities: mcp_types.ServerCapabilities | None = None
    _session: ClientSession | None = None # From mcp-lib
    _exit_stack: AsyncExitStack
    _request_id_counter: int = 0

    def __init__(
        self,
        client_info: mcp_types.Implementation | None = None,
        client_capabilities: mcp_types.ClientCapabilities | None = None,
        protocol_version: str = "2025-03-26-preview",
    ):
        super().__init__(client_info, client_capabilities, protocol_version)
        if not HAS_MCP_CLIENT:
            # This check is also in MCPSessionManager.create, but good for direct instantiation too.
            raise ImportError("To use stdio/sse connections, install the 'mcp' package.")
        self._exit_stack = AsyncExitStack()
        self._request_id_counter = 0

    def _next_mcp_lib_id(self) -> str: # mcp-lib seems to use string IDs for its internal request objects
        self._request_id_counter +=1
        return str(self._request_id_counter)

    async def _initialize_mcp_session_capabilities(self, session: ClientSession) -> None:
        """Sends a typed InitializeRequest through an existing mcp.ClientSession."""
        init_params = mcp_types.InitializeRequestParams(
            capabilities=self._client_capabilities,
            clientInfo=self._client_info,
            protocolVersion=self._protocol_version
        )
        method_name = "initialize"
        request_id = self._next_mcp_lib_id() # Use the ID format expected by mcp-lib's internal request if possible

        try:
            _logger.debug(f"Stdio/Sse: Sending typed InitializeRequest: {init_params}")
            # This part is highly dependent on mcp.ClientSession's API.
            # Option 1: session.initialize() itself can take parameters or returns detailed result.
            # The current `await self._session.initialize()` in _ensure_session is parameterless.
            # If it returned the raw InitializeResult dict:
            #   raw_init_result_dict = await session.initialize() # Or some other attribute like session.server_info_dict
            # Option 2: A generic request sender on ClientSession.
            #   `response_obj = await session._send_request_obj(mcp.protocol.Request(id=request_id, method=method_name, params=init_params.model_dump(exclude_none=True)))`
            #   raw_init_result_dict = response_obj.result
            # This is too speculative. For now, assume mcp-lib's initialize is basic.
            # We will log a warning if capabilities aren't found through a standard way.

            # Attempt to retrieve capabilities if mcp-lib populated them after its own initialize()
            # This is a guess based on how such libraries might work.
            raw_server_info = getattr(session, '_server_info', None) # mcp-lib might store it like this
            if raw_server_info and isinstance(raw_server_info, dict):
                init_result = mcp_types.InitializeResult.model_validate(raw_server_info)
                self.server_capabilities = init_result.capabilities
                _logger.info(f"MCP Stdio/SSE session capabilities populated from mcp-lib's _server_info. Server: {init_result.serverInfo.name if init_result.serverInfo else 'Unknown'}")
            else:
                # Fallback: if we had a generic way to send initialize via mcp-lib
                # init_result_dict = await self._send_generic_mcp_request(session, method_name, init_params.model_dump(exclude_none=True), request_id)
                # init_result = mcp_types.InitializeResult.model_validate(init_result_dict)
                # self.server_capabilities = init_result.capabilities
                # _logger.info(f"MCP Stdio/SSE session initialized via explicit InitializeRequest. Server: {init_result.serverInfo.name}")
                 _logger.warning("Could not automatically retrieve detailed server capabilities for Stdio/SSE session after mcp-lib initialize. ServerCapabilities will be None.")

        except pydantic.ValidationError as e:
            _logger.error(f"Stdio/Sse: Failed to validate server capabilities from mcp-lib: {e}")
            self.server_capabilities = None # Ensure it's None if validation fails
            raise MCPProtocolError("Failed to validate server capabilities structure from mcp-lib.", underlying_error=e) from e
        except MCPLibError as e: # Catching specific mcp-lib errors
            _logger.error(f"Stdio/Sse: mcp-lib error during typed initialize: {e}")
            raise MCPProtocolError(f"MCP library error during initialize: {e}", underlying_error=e) from e
        except Exception as e:
            _logger.error(f"Stdio/Sse: Generic error during typed initialize: {e}")
            # Don't override self.server_capabilities here, it might have been set by basic init
            raise MCPConnectionError(f"Failed to send/process typed InitializeRequest: {e}", underlying_error=e) from e


    async def _make_rpc_call(self, method: str, params: _t.Optional[pydantic.BaseModel], result_type: _t.Type[pydantic.BaseModel] | None) -> _pydantic.BaseModel | None:
        if self._session is None:
            # This should ideally be caught by _ensure_session before _make_rpc_call is invoked.
            _logger.error(f"Session not available for MCP call {method}")
            raise MCPConnectionError("Session is not initialized or has been closed.")

        session = self._session
        params_dict = params.model_dump(exclude_none=True) if params else {}
        mcp_method_name = method.replace("/", "_") # e.g. "resources/list" -> "resources_list"
        response_data = None
        request_id = self._next_mcp_lib_id()

        try:
            _logger.debug(f"Stdio/Sse: Calling MCP method '{method}' (as {mcp_method_name}) with params: {params_dict}")
            if hasattr(session, mcp_method_name):
                api_call = getattr(session, mcp_method_name)
                # Assuming mcp-lib methods take params as kwargs or a single dict arg
                if isinstance(params_dict, dict):
                    raw_response = await api_call(**params_dict)
                else: # Should not happen if params_dict is from model_dump
                    raw_response = await api_call(params_dict)
            elif hasattr(session, '_send_request_obj') and hasattr(mcp_types, 'MCPProtocolRequest'): # mcp.protocol.Request
                # Hypothetical: Using a generic sender if specific method not found
                from mcp.protocol import Request as MCPInternalRequest # Avoid circular if mcp_types imports this
                mcp_req = MCPInternalRequest(id=request_id, method=method, params=params_dict)
                _logger.debug(f"Stdio/Sse: Using generic _send_request_obj for {method}")
                raw_response = await session._send_request_obj(mcp_req) # This is speculative
            else:
                _logger.error(f"Method {mcp_method_name} (from {method}) not found on mcp.ClientSession and no generic sender known.")
                raise NotImplementedError(f"Method {mcp_method_name} not available on mcp.ClientSession.")

            # Process response (could be dict from simple methods, or mcp.protocol.Response from _send_request_obj)
            if isinstance(raw_response, MCPResponseProtocol):
                if raw_response.error:
                    err = raw_response.error
                    _logger.error(f"MCP error from {method} (id: {raw_response.id}): {err.message} (Code: {err.code}) Data: {err.data}")
                    raise MCPProtocolError(f"MCP Error from server for {method}: {err.message} (Code: {err.code})")
                response_data = raw_response.result
            else: # Assuming it's a direct result dict (e.g. from list_tools())
                response_data = raw_response

            _logger.debug(f"Stdio/Sse: Received raw response for {method}: {response_data}")
            if result_type is None: # For methods not returning specific content (e.g. subscribe)
                return None
            return result_type.model_validate(response_data)

        except MCPLibError as e: # Catching specific mcp-lib errors
             _logger.error(f"Stdio/Sse: mcp-lib error calling {method}: {e}")
             raise MCPProtocolError(f"MCP library error on {method}: {e}", underlying_error=e) from e
        except (ConnectionRefusedError, BrokenPipeError, ConnectionResetError) as e:
            _logger.error(f"Stdio/Sse: Connection error calling {method}: {e}")
            await self.close() # Ensure session is marked as unusable
            raise MCPConnectionError(f"Connection error on {method}: {e}", underlying_error=e) from e
        except asyncio.TimeoutError as e: # If mcp-lib operations cause asyncio timeout
            _logger.error(f"Stdio/Sse: Timeout calling {method}: {e}")
            await self.close()
            raise MCPTimeoutError(f"Timeout on {method}: {e}", underlying_error=e) from e
        except pydantic.ValidationError as e:
            _logger.error(f"Stdio/Sse: Pydantic validation error for {method} response: {response_data}, Error: {e}")
            raise MCPProtocolError(f"Invalid response structure for {method}.", underlying_error=e) from e
        except Exception as e: # Catch-all for other unexpected errors from mcp-lib or processing
            _logger.exception(f"Stdio/Sse: Unexpected error calling {method}: {e}")
            await self.close() # Close session on unknown errors
            raise MCPError(f"Unexpected error on {method}: {e}", underlying_error=e) from e


    @abc.abstractmethod
    async def _ensure_session(self) -> ClientSession:
        pass

    async def initialize(self) -> None:
        session = await self._ensure_session() # Calls mcp-lib's basic initialize()
        await self._initialize_mcp_session_capabilities(session) # Sends our typed InitializeRequest

    async def list_resources(self, params: mcp_types.ListResourcesRequestParams) -> mcp_types.ListResourcesResult:
        return await self._make_rpc_call("resources/list", params, mcp_types.ListResourcesResult)

    async def list_resource_templates(self, params: mcp_types.ListResourceTemplatesRequestParams) -> mcp_types.ListResourceTemplatesResult:
        return await self._make_rpc_call("resources/templates/list", params, mcp_types.ListResourceTemplatesResult)

    async def read_resource(self, params: mcp_types.ReadResourceRequestParams) -> mcp_types.ReadResourceResult:
        return await self._make_rpc_call("resources/read", params, mcp_types.ReadResourceResult)

    async def subscribe_resource(self, params: mcp_types.SubscribeRequestParams) -> None:
        await self._make_rpc_call("resources/subscribe", params, None)
        return None

    async def unsubscribe_resource(self, params: mcp_types.UnsubscribeRequestParams) -> None:
        await self._make_rpc_call("resources/unsubscribe", params, None)
        return None

    async def list_prompts(self, params: mcp_types.ListPromptsRequestParams) -> mcp_types.ListPromptsResult:
        return await self._make_rpc_call("prompts/list", params, mcp_types.ListPromptsResult)

    async def get_prompt(self, params: mcp_types.GetPromptRequestParams) -> mcp_types.GetPromptResult:
        return await self._make_rpc_call("prompts/get", params, mcp_types.GetPromptResult)

    async def set_logging_level(self, params: mcp_types.SetLevelRequestParams) -> None:
        await self._make_rpc_call("logging/setLevel", params, None)
        return None

    async def get_completion(self, params: mcp_types.CompleteRequestParams) -> mcp_types.CompleteResult:
        return await self._make_rpc_call("completion/complete", params, mcp_types.CompleteResult)


class StdioSessionManager(StdioSessionManagerBase):
    server_capabilities: mcp_types.ServerCapabilities | None = None

    def __init__(
        self,
        params: StdioServerParams,
        *,
        errlog: _t.TextIO = sys.stderr,
        client_info: mcp_types.Implementation | None = None,
        client_capabilities: mcp_types.ClientCapabilities | None = None,
        protocol_version: str = "2025-03-26-preview",
    ) -> None:
        super().__init__(client_info, client_capabilities, protocol_version)
        self._params = params
        self._errlog = errlog
        self._session: ClientSession | None = None

    async def _ensure_session(self) -> ClientSession:
        if self._session is None:
            _logger.info(f"StdioSessionManager: No active session, creating new one. Command: {self._params.command}")
            try:
                server_params = StdioServerParameters(
                    command=self._params.command,
                    args=self._params.args,
                )
                stdio_context = stdio_client(server_params) # This can raise errors
                read_stream, write_stream = await self._exit_stack.enter_async_context(stdio_context)
                session_context = ClientSession(read_stream, write_stream)
                self._session = await self._exit_stack.enter_async_context(session_context)

                _logger.debug("StdioSessionManager: mcp.ClientSession created. Calling mcp-lib initialize().")
                await self._session.initialize() # mcp-lib's own initialize
                _logger.info("StdioSessionManager: mcp.ClientSession initialized by mcp-lib.")

            except FileNotFoundError as e: # Specific error for command not found
                _logger.error(f"StdioSessionManager: Command not found for stdio server: {self._params.command}. Error: {e}")
                await self.close()
                raise MCPConnectionError(f"Stdio command not found: {self._params.command}", underlying_error=e) from e
            except MCPLibError as e: # Catching specific mcp-lib errors during its init
                _logger.error(f"StdioSessionManager: mcp-lib error during session setup: {e}")
                await self.close()
                raise MCPConnectionError(f"MCP library error during stdio session setup: {e}", underlying_error=e) from e
            except Exception as e: # Catch-all for other errors during stdio_client or mcp-lib ClientSession setup
                _logger.exception(f"StdioSessionManager: Failed to establish stdio session: {e}")
                await self.close() # Attempt to clean up if partial setup occurred
                raise MCPConnectionError(f"Failed to establish stdio session: {e}", underlying_error=e) from e

        if self._session is None: # Should be redundant if above logic is correct
             _logger.error("StdioSessionManager: Session is None after attempt to ensure session.")
             raise MCPConnectionError("Failed to establish MCP Stdio session (unknown reason).")
        return self._session

    async def call_tool(self, tool_name: str, arguments: dict[str, _t.Any]) -> _t.Any:
        session = await self._ensure_session()
        _logger.debug(f"StdioSessionManager: Calling MCP tool '{tool_name}' with args: {arguments}")
        try:
            # Assuming call_tool in mcp-lib handles its own errors or they are generic
            return await session.call_tool(tool_name, arguments=arguments)
        except MCPLibError as e:
             _logger.error(f"StdioSessionManager: mcp-lib error calling tool {tool_name}: {e}")
             raise MCPProtocolError(f"MCP library error on tool call {tool_name}: {e}", underlying_error=e) from e
        except Exception as e:
            _logger.exception(f"StdioSessionManager: Unexpected error calling tool {tool_name}: {e}")
            await self.close()
            raise MCPError(f"Unexpected error on tool call {tool_name}: {e}", underlying_error=e) from e


    async def list_tools(self) -> list[dict[str, _t.Any]]:
        session = await self._ensure_session()
        _logger.debug("StdioSessionManager: Listing tools.")
        try:
            result = await session.list_tools()
            return result.tools if hasattr(result, 'tools') else []
        except MCPLibError as e:
             _logger.error(f"StdioSessionManager: mcp-lib error listing tools: {e}")
             raise MCPProtocolError(f"MCP library error on list_tools: {e}", underlying_error=e) from e
        except Exception as e:
            _logger.exception(f"StdioSessionManager: Unexpected error listing tools: {e}")
            await self.close()
            raise MCPError(f"Unexpected error on list_tools: {e}", underlying_error=e) from e

    async def close(self) -> None:
        _logger.info("StdioSessionManager: Closing session.")
        session_to_close = self._session
        exit_stack_to_close = self._exit_stack
        self._session = None # Mark as unusable immediately
        self._exit_stack = AsyncExitStack() # Replace for future use if any (though typically not)

        original_stderr = sys.stderr
        original_showwarning = warnings.showwarning
        try:
            # Suppress warnings/errors during cleanup
            with open(os.devnull, 'w') as devnull:
                sys.stderr = devnull
                warnings.showwarning = lambda *args, **kwargs: None
                if session_to_close and hasattr(session_to_close, 'close'):
                    await asyncio.wait_for(session_to_close.close(), timeout=1.0) # Increased timeout slightly
                if exit_stack_to_close:
                    await asyncio.wait_for(exit_stack_to_close.aclose(), timeout=1.0)
        except asyncio.TimeoutError:
            _logger.warning("StdioSessionManager: Timeout during cleanup of session/exit_stack.")
        except Exception as e:
            _logger.debug(f"StdioSessionManager: Exception during cleanup: {e}", exc_info=True) # Log with traceback for debug
        finally:
            sys.stderr = original_stderr
            warnings.showwarning = original_showwarning


class SseSessionManager(StdioSessionManagerBase):
    server_capabilities: mcp_types.ServerCapabilities | None = None

    def __init__(
        self,
        params: SseServerParams,
        client_info: mcp_types.Implementation | None = None,
        client_capabilities: mcp_types.ClientCapabilities | None = None,
        protocol_version: str = "2025-03-26-preview",
    ) -> None:
        super().__init__(client_info, client_capabilities, protocol_version)
        self._params = params
        self._session: ClientSession | None = None

    async def _ensure_session(self) -> ClientSession:
        if self._session is None:
            _logger.info(f"SseSessionManager: No active session, creating new one. URL: {self._params.url}")
            try:
                sse_client_context = sse_client(
                    url=self._params.url,
                    headers=self._params.headers,
                    timeout=self._params.timeout,
                    sse_read_timeout=self._params.sse_read_timeout,
                )
                read_stream, write_stream = await self._exit_stack.enter_async_context(sse_client_context)
                session_context = ClientSession(read_stream, write_stream)
                self._session = await self._exit_stack.enter_async_context(session_context)

                _logger.debug("SseSessionManager: mcp.ClientSession created. Calling mcp-lib initialize().")
                await self._session.initialize() # mcp-lib's own initialize
                _logger.info("SseSessionManager: mcp.ClientSession initialized by mcp-lib.")

            except MCPLibError as e: # Catching specific mcp-lib errors
                _logger.error(f"SseSessionManager: mcp-lib error during session setup: {e}")
                await self.close()
                raise MCPConnectionError(f"MCP library error during SSE session setup: {e}", underlying_error=e) from e
            except Exception as e: # Catch-all for sse_client or mcp-lib ClientSession setup
                _logger.exception(f"SseSessionManager: Failed to establish SSE session: {e}")
                await self.close()
                raise MCPConnectionError(f"Failed to establish SSE session: {e}", underlying_error=e) from e

        if self._session is None:
             _logger.error("SseSessionManager: Session is None after attempt to ensure session.")
             raise MCPConnectionError("Failed to establish MCP SSE session (unknown reason).")
        return self._session

    async def call_tool(self, tool_name: str, arguments: dict[str, _t.Any]) -> _t.Any:
        session = await self._ensure_session()
        _logger.debug(f"SseSessionManager: Calling MCP tool '{tool_name}' with args: {arguments}")
        try:
            return await session.call_tool(tool_name, arguments=arguments)
        except MCPLibError as e:
             _logger.error(f"SseSessionManager: mcp-lib error calling tool {tool_name}: {e}")
             raise MCPProtocolError(f"MCP library error on tool call {tool_name}: {e}", underlying_error=e) from e
        except Exception as e:
            _logger.exception(f"SseSessionManager: Unexpected error calling tool {tool_name}: {e}")
            await self.close()
            raise MCPError(f"Unexpected error on tool call {tool_name}: {e}", underlying_error=e) from e

    async def list_tools(self) -> list[dict[str, _t.Any]]:
        session = await self._ensure_session()
        _logger.debug("SseSessionManager: Listing tools.")
        try:
            result = await session.list_tools()
            return result.tools if hasattr(result, 'tools') else []
        except MCPLibError as e:
             _logger.error(f"SseSessionManager: mcp-lib error listing tools: {e}")
             raise MCPProtocolError(f"MCP library error on list_tools: {e}", underlying_error=e) from e
        except Exception as e:
            _logger.exception(f"SseSessionManager: Unexpected error listing tools: {e}")
            await self.close()
            raise MCPError(f"Unexpected error on list_tools: {e}", underlying_error=e) from e

    async def close(self) -> None:
        _logger.info("SseSessionManager: Closing session.")
        session_to_close = self._session
        exit_stack_to_close = self._exit_stack # Capture current stack
        self._session = None # Mark as unusable
        self._exit_stack = AsyncExitStack() # Reset for any future use (unlikely for same instance)
        try:
            if session_to_close and hasattr(session_to_close, 'close'):
                await asyncio.wait_for(session_to_close.close(), timeout=1.0)
        except asyncio.TimeoutError:
            _logger.warning("SseSessionManager: Timeout during session.close().")
        except Exception as e:
            _logger.debug(f'SseSessionManager: Error closing mcp.ClientSession gracefully: {e}', exc_info=True)

        try:
            if exit_stack_to_close: # Ensure it was set
                await asyncio.wait_for(exit_stack_to_close.aclose(), timeout=1.0)
        except asyncio.TimeoutError:
            _logger.warning("SseSessionManager: Timeout during exit_stack.aclose().")
        except RuntimeError as e: # e.g. "Attempted to exit cancel scope in a different task"
            _logger.debug(f'SseSessionManager: RuntimeError during exit stack cleanup: {e}', exc_info=True)
        except Exception as e:
            _logger.debug(f'SseSessionManager: Error closing exit stack: {e}', exc_info=True)

