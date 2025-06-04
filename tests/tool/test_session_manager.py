from __future__ import annotations

"""Tests for the MCP session manager classes."""

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from src.codin.tool.mcp.server_connection import (
    HttpServerParams,
    StdioServerParams,
    SseServerParams,
)
from src.codin.tool.mcp.session_manager import (
    MCPSessionManager,
    HttpSessionManager,
    StdioSessionManager,
    SseSessionManager,
)


class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}
        
    def json(self):
        return self._json_data
        
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP Error {self.status_code}",
                request=httpx.Request("GET", "https://example.com"),
                response=self,
            )


@pytest.mark.asyncio
async def test_http_session_manager():
    """Test the HTTP session manager."""
    # Create a mock transport that returns test responses
    async def mock_post(url, json):
        if url == "/tools/test_tool":
            return MockResponse(json_data={"result": "success", "args": json})
        return MockResponse(status_code=404)
    
    async def mock_get(url):
        if url == "/tools":
            return MockResponse(json_data={"tools": [
                {"name": "test_tool", "description": "A test tool"}
            ]})
        return MockResponse(status_code=404)
    
    # Create a mock client
    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_client.get = AsyncMock(side_effect=mock_get)
    mock_client.is_closed = False
    
    # Create the session manager
    params = HttpServerParams(base_url="https://example.com")
    session_manager = HttpSessionManager(params)
    
    # Patch the get_client method to return our mock
    with patch.object(session_manager, "get_client", return_value=mock_client):
        # Test calling a tool
        result = await session_manager.call_tool("test_tool", {"param": "value"})
        assert result == {"result": "success", "args": {"param": "value"}}
        mock_client.post.assert_called_once_with("/tools/test_tool", json={"param": "value"})
        
        # Test listing tools
        tools = await session_manager.list_tools()
        assert tools == [{"name": "test_tool", "description": "A test tool"}]
        mock_client.get.assert_called_once_with("/tools")


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 10), reason="MCP client requires Python 3.10+")
@patch("src.codin.tool.mcp.session_manager.HAS_MCP_CLIENT", True)
@patch("src.codin.tool.mcp.session_manager.StdioServerParameters")
@patch("src.codin.tool.mcp.session_manager.stdio_client")
@patch("src.codin.tool.mcp.session_manager.ClientSession")
async def test_stdio_session_manager(
    mock_client_session,
    mock_stdio_client,
    mock_stdio_params,
):
    """Test the stdio session manager."""
    # Set up mocks
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=["transport1", "transport2"])
    
    mock_stdio_client.return_value = mock_client
    
    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value={"result": "success"})
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[
        {"name": "stdio_tool", "description": "A stdio tool"}
    ]))
    
    mock_client_session.return_value = mock_session
    mock_client_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    
    # Create session manager with mocked error log
    params = StdioServerParams(command="node", args=["mcp-server.js"])
    session_manager = StdioSessionManager(params, errlog=MagicMock())
    
    # Test calling a tool
    result = await session_manager.call_tool("stdio_tool", {"param": "value"})
    assert result == {"result": "success"}
    
    # Test listing tools
    tools = await session_manager.list_tools()
    assert tools == [{"name": "stdio_tool", "description": "A stdio tool"}]
    
    # Clean up
    await session_manager.close()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 10), reason="MCP client requires Python 3.10+")
@patch("src.codin.tool.mcp.session_manager.HAS_MCP_CLIENT", True)
@patch("src.codin.tool.mcp.session_manager.sse_client")
@patch("src.codin.tool.mcp.session_manager.ClientSession")
async def test_sse_session_manager(
    mock_client_session,
    mock_sse_client,
):
    """Test the SSE session manager."""
    # Set up mocks
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=["transport1", "transport2"])
    
    mock_sse_client.return_value = mock_client
    
    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value={"result": "success"})
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[
        {"name": "sse_tool", "description": "An SSE tool"}
    ]))
    
    mock_client_session.return_value = mock_session
    mock_client_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    
    # Create session manager
    params = SseServerParams(url="https://example.com/sse")
    session_manager = SseSessionManager(params)
    
    # Test calling a tool
    result = await session_manager.call_tool("sse_tool", {"param": "value"})
    assert result == {"result": "success"}
    
    # Test listing tools
    tools = await session_manager.list_tools()
    assert tools == [{"name": "sse_tool", "description": "An SSE tool"}]
    
    # Clean up
    await session_manager.close()


@pytest.mark.asyncio
async def test_factory_method():
    """Test the factory method for creating session managers."""
    # HTTP params should create an HttpSessionManager
    http_params = HttpServerParams(base_url="https://example.com")
    session_manager = MCPSessionManager.create(http_params)
    assert isinstance(session_manager, HttpSessionManager)
    
    # Test with invalid params type
    with pytest.raises(ValueError):
        MCPSessionManager.create("not a valid params object") 