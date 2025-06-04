from __future__ import annotations

"""Tests for `codin.tool.mcp.MCPTool`."""

import httpx
import json
import pytest

from src.codin.tool.mcp.session_manager import MCPSessionManager
from src.codin.tool.mcp.server_connection import HttpServerParams
from src.codin.tool.mcp.mcp_tool import MCPTool
from src.codin.tool.base import ToolContext


@pytest.mark.asyncio
async def test_mcp_tool_invocation():  # noqa: D401
    """`MCPTool.run` should perform an HTTP POST and return the JSON body."""

    async def _mock_send(request: httpx.Request) -> httpx.Response:  # noqa: D401
        assert request.method == "POST"
        assert request.url.path == "/tools/echo"
        # Access JSON payload from request content
        json_payload = json.loads(request.content.decode())
        return httpx.Response(status_code=200, json={"echo": json_payload})

    transport = httpx.MockTransport(_mock_send)
    
    # Create proper HTTP server parameters
    http_params = HttpServerParams(base_url="https://mcp.example.com")
    session_manager = MCPSessionManager.create(http_params, transport=transport)

    tool = MCPTool(
        name="echo",
        description="Echo tool for tests",
        session_manager=session_manager,
    )

    result = await tool.run({"value": 123}, ToolContext())
    assert result == {"echo": {"value": 123}}

    await session_manager.close() 