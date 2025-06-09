"""Unit tests for MCP Session Managers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
import json # For http client serializing requests

import httpx # For exceptions

from .. import mcp_types
from ..exceptions import MCPConnectionError, MCPHttpError, MCPProtocolError, MCPTimeoutError
from ..session_manager import (
    HttpSessionManager,
    StdioSessionManager,
    SseSessionManager,
    DEFAULT_CLIENT_CAPABILITIES, # For creating InitializeRequest
    DEFAULT_CLIENT_INFO,       # For creating InitializeRequest
)
from ..server_connection import HttpServerParams, StdioServerParams, SseServerParams

# --- Fixtures ---

@pytest.fixture
def mock_http_params():
    return HttpServerParams(base_url="http://testserver.com/mcp")

@pytest.fixture
def mock_stdio_params():
    return StdioServerParams(command="echo_server")

@pytest.fixture
def mock_sse_params():
    return SseServerParams(url="http://testserver.com/mcp/sse")

@pytest.fixture
def default_initialize_request_params():
    return mcp_types.InitializeRequestParams(
        capabilities=DEFAULT_CLIENT_CAPABILITIES,
        clientInfo=DEFAULT_CLIENT_INFO,
        protocolVersion="2025-03-26-preview" # Matches default in session managers
    )

@pytest.fixture
def sample_server_capabilities_dict():
    return {"tools": {"listChanged": True}, "prompts": {"listChanged": False}}

@pytest.fixture
def sample_initialize_result_dict(sample_server_capabilities_dict):
    return {
        "capabilities": sample_server_capabilities_dict,
        "serverInfo": {"name": "Test MCP Server", "version": "1.0.0"},
        "protocolVersion": "2025-03-26-preview",
        "instructions": "Test instructions"
    }

# --- HttpSessionManager Tests ---

@pytest.mark.asyncio
async def test_http_session_manager_initialize_success(
    mock_http_params, default_initialize_request_params, sample_initialize_result_dict
):
    manager = HttpSessionManager(params=mock_http_params)

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    # _send_request expects the "result" part of JSONRPCResponse
    # initialize method directly posts and expects InitializeResult or full JSONRPCResponse
    # The initialize method in HttpSessionManager was changed to post the InitializeRequest directly
    # and expect a JSONRPCResponse containing InitializeResult.
    # So, the mock needs to return the full JSONRPCResponse structure.
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": manager._next_id() -1, # It increments before making request
        "result": sample_initialize_result_dict
    }

    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.post = AsyncMock(return_value=mock_response)

    manager._client = mock_async_client # Inject mock client

    await manager.initialize()

    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args
    assert call_args[0][0] == "/initialize" # Endpoint for initialize
    sent_json = call_args[1]['json']

    assert sent_json['method'] == "initialize"
    assert sent_json['params']['capabilities'] == DEFAULT_CLIENT_CAPABILITIES.model_dump(exclude_none=True)
    assert sent_json['params']['clientInfo'] == DEFAULT_CLIENT_INFO.model_dump(exclude_none=True)

    assert manager.server_capabilities is not None
    assert manager.server_capabilities.tools == {"listChanged": True}
    assert manager.server_capabilities.model_dump(exclude_none=True) == sample_initialize_result_dict["capabilities"]


@pytest.mark.asyncio
async def test_http_session_manager_initialize_http_status_error(mock_http_params):
    manager = HttpSessionManager(params=mock_http_params)

    http_error = httpx.HTTPStatusError(
        message="Server Error",
        request=MagicMock(),
        response=MagicMock(spec=httpx.Response, status_code=500, text="Internal Server Error")
    )
    # Configure the response mock for json parsing inside error handling
    http_error.response.json = MagicMock(side_effect=json.JSONDecodeError("msg", "doc", 0))


    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.post = AsyncMock(side_effect=http_error)
    manager._client = mock_async_client

    with pytest.raises(MCPHttpError) as exc_info:
        await manager.initialize()

    assert exc_info.value.status_code == 500
    assert "HTTP error 500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_http_session_manager_initialize_timeout_error(mock_http_params):
    manager = HttpSessionManager(params=mock_http_params)
    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout", request=MagicMock()))
    manager._client = mock_async_client

    with pytest.raises(MCPTimeoutError):
        await manager.initialize()

@pytest.mark.asyncio
async def test_http_session_manager_initialize_connection_error(mock_http_params):
    manager = HttpSessionManager(params=mock_http_params)
    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.post = AsyncMock(side_effect=httpx.RequestError("Network Error", request=MagicMock()))
    manager._client = mock_async_client

    with pytest.raises(MCPConnectionError):
        await manager.initialize()

@pytest.mark.asyncio
async def test_http_session_manager_initialize_protocol_error_invalid_json(mock_http_params):
    manager = HttpSessionManager(params=mock_http_params)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)

    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.post = AsyncMock(return_value=mock_response)
    manager._client = mock_async_client

    with pytest.raises(MCPProtocolError) as exc_info:
        await manager.initialize()
    assert "Invalid JSON response" in str(exc_info.value)


@pytest.mark.asyncio
async def test_http_session_manager_initialize_protocol_error_missing_result(mock_http_params, sample_initialize_result_dict):
    manager = HttpSessionManager(params=mock_http_params)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    # Simulate server returning JSON-RPC error object instead of full response with result
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": manager._next_id() -1,
        "error": {"code": -32000, "message": "Some server error"}
    }

    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.post = AsyncMock(return_value=mock_response)
    manager._client = mock_async_client

    with pytest.raises(MCPProtocolError) as exc_info:
        await manager.initialize()
    assert "MCP Error from server: Some server error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_http_session_manager_list_resources_success(mock_http_params, sample_server_capabilities_dict):
    manager = HttpSessionManager(params=mock_http_params)
    manager.server_capabilities = mcp_types.ServerCapabilities.model_validate(sample_server_capabilities_dict) # Assume initialized

    expected_result_data = {"resources": [{"name": "res1", "uri": "file:///res1.txt"}]}
    mock_response_json = {
        "jsonrpc": "2.0",
        "id": manager._next_id(), # ID that _send_request would generate
        "result": expected_result_data
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_json

    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.post = AsyncMock(return_value=mock_response)
    manager._client = mock_async_client

    params = mcp_types.ListResourcesRequestParams(cursor="test_cursor")
    result = await manager.list_resources(params)

    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args
    sent_json = call_args[1]['json']
    assert sent_json['method'] == "resources/list"
    assert sent_json['params'] == params.model_dump(exclude_none=True)

    assert isinstance(result, mcp_types.ListResourcesResult)
    assert len(result.resources) == 1
    assert result.resources[0].name == "res1"


# --- StdioSessionManager & SseSessionManager Tests (Basic Structure) ---
# These will require more complex mocking of the 'mcp' library.

@pytest.fixture
def mock_mcp_client_session():
    session = AsyncMock(spec_set=['initialize', 'call_tool', 'list_tools', 'close', '_send_request_obj'] + [m.replace("/", "_") for m in mcp_types.MCP_METHOD_MAP.keys()])
    # Mock methods assumed by StdioSessionManagerBase._make_rpc_call
    # For example, if list_resources calls session.resources_list:
    session.resources_list = AsyncMock()
    session.prompts_get = AsyncMock()
    # ... etc. for all methods in MCP_METHOD_MAP
    # if _send_request_obj is used as fallback
    session._send_request_obj = AsyncMock()
    return session

# Patch 'HAS_MCP_CLIENT' to True for these tests, and mock away the client libs themselves
@patch('codin.tool.mcp.session_manager.HAS_MCP_CLIENT', True)
@patch('codin.tool.mcp.session_manager.ClientSession')
@patch('codin.tool.mcp.session_manager.stdio_client')
@pytest.mark.asyncio
async def test_stdio_session_manager_initialize_success(
    mock_stdio_client, mock_ClientSession, mock_stdio_params,
    sample_server_capabilities_dict, default_initialize_request_params
):
    mock_streams = (AsyncMock(), AsyncMock()) # read_stream, write_stream
    mock_stdio_client.return_value.__aenter__.return_value = mock_streams

    mock_session_instance = mock_ClientSession.return_value
    mock_session_instance.__aenter__.return_value = mock_session_instance # The session itself

    # Simulate mcp-lib's initialize() and how capabilities might be set
    # Option A: initialize() returns a dict that can be parsed into InitializeResult
    # mock_session_instance.initialize = AsyncMock(return_value=sample_initialize_result_dict)
    # Option B: initialize() populates an attribute on the session
    mock_session_instance.initialize = AsyncMock(return_value=None) # Basic initialize
    # Simulate that mcp-lib's initialize populates _server_info
    # StdioSessionManagerBase._initialize_mcp_session_capabilities tries to read this
    mock_session_instance._server_info = { # This is a guess on mcp-lib internal
        "capabilities": sample_server_capabilities_dict,
        "serverInfo": {"name": "Test Stdio Server", "version": "0.5"},
        "protocolVersion": "2025-03-26-preview"
    }

    manager = StdioSessionManager(params=mock_stdio_params)
    await manager.initialize() # This calls _ensure_session then _initialize_mcp_session_capabilities

    mock_stdio_client.assert_called_once()
    mock_ClientSession.assert_called_with(mock_streams[0], mock_streams[1])
    mock_session_instance.initialize.assert_called_once() # mcp-lib's initialize

    assert manager.server_capabilities is not None
    assert manager.server_capabilities.model_dump(exclude_none=True) == sample_server_capabilities_dict


@patch('codin.tool.mcp.session_manager.HAS_MCP_CLIENT', True)
@patch('codin.tool.mcp.session_manager.ClientSession')
@patch('codin.tool.mcp.session_manager.stdio_client')
@pytest.mark.asyncio
async def test_stdio_session_manager_list_resources_success(
    mock_stdio_client, mock_ClientSession, mock_stdio_params, mock_mcp_client_session
):
    # Setup _ensure_session to return our detailed mock_mcp_client_session
    mock_streams = (AsyncMock(), AsyncMock())
    mock_stdio_client.return_value.__aenter__.return_value = mock_streams

    # When ClientSession is instantiated, return our highly configured mock_mcp_client_session
    mock_ClientSession.return_value.__aenter__.return_value = mock_mcp_client_session
    mock_mcp_client_session.initialize = AsyncMock() # Basic initialize from mcp-lib

    manager = StdioSessionManager(params=mock_stdio_params)
    # Manually set session to avoid _ensure_session re-mocking if not careful
    # await manager._ensure_session() # This would set up the session with the mocks
    # Or, more directly for testing _make_rpc_call:
    manager._session = mock_mcp_client_session
    manager.server_capabilities = mcp_types.ServerCapabilities() # Assume initialized

    expected_result_data = {"resources": [{"name": "res_stdio", "uri": "file:///res_stdio.txt"}], "nextCursor": None}
    # Configure the specific method mock on mock_mcp_client_session
    mock_mcp_client_session.resources_list = AsyncMock(return_value=expected_result_data)

    params = mcp_types.ListResourcesRequestParams()
    result = await manager.list_resources(params)

    mock_mcp_client_session.resources_list.assert_called_once_with() # Or with params_dict if non-empty
    assert isinstance(result, mcp_types.ListResourcesResult)
    assert result.resources[0].name == "res_stdio"


# TODO: More tests for StdioSessionManager error cases (e.g. command not found, mcp-lib errors)
# TODO: Tests for SseSessionManager, similar structure to Stdio but mocking sse_client

# Placeholder for more tests
def test_true_placeholder(): # Keep a simple placeholder if other tests are complex to write initially
    assert True

```
