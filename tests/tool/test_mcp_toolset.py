from __future__ import annotations

"""Tests for the MCPToolset class."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.codin.tool.mcp.server_connection import HttpServerParams
from src.codin.tool.mcp.mcp_tool import MCPTool
from src.codin.tool.mcp.toolset import MCPToolset


@pytest.mark.asyncio
async def test_mcp_toolset():
    """Test the MCPToolset class."""
    # Create a mock session manager
    mock_session_manager = MagicMock()
    mock_session_manager.list_tools = AsyncMock(return_value=[
        {"name": "tool1", "description": "Tool 1 description"},
        {"name": "tool2", "description": "Tool 2 description"},
        {"name": "skip_me", "description": "This tool should be filtered out"},
    ])
    mock_session_manager.call_tool = AsyncMock(return_value={"result": "success"})
    mock_session_manager.close = AsyncMock()
    
    # Create a filter function
    def tool_filter(tool_info):
        return tool_info["name"] != "skip_me"
    
    # Create the toolset with mocked session manager creation
    with patch("src.codin.tool.mcp.toolset.MCPSessionManager") as mock_session_manager_cls:
        mock_session_manager_cls.create.return_value = mock_session_manager
        
        # Create the toolset
        params = HttpServerParams(base_url="https://example.com")
        toolset = MCPToolset(
            name="Test Toolset",
            description="Test toolset for MCP tools",
            connection_params=params,
            tool_filter=tool_filter
        )
        
        # Get the tools
        tools = await toolset.get_tools()
        
        # Verify that we got the filtered tools
        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"
        
        # Verify that skip_me was filtered out
        assert not any(tool.name == "skip_me" for tool in tools)
        
        # Get tools again to test caching
        tools2 = await toolset.get_tools()
        assert tools2 is tools  # Should be the same list instance (cached)
        
        # Close the toolset
        await toolset.close()
        mock_session_manager.close.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_toolset_no_filter():
    """Test MCPToolset without a filter function."""
    # Create a mock session manager
    mock_session_manager = MagicMock()
    mock_session_manager.list_tools = AsyncMock(return_value=[
        {"name": "tool1", "description": "Tool 1 description"},
        {"name": "tool2", "description": "Tool 2 description"},
    ])
    
    # Create the toolset with mocked session manager creation
    with patch("src.codin.tool.mcp.toolset.MCPSessionManager") as mock_session_manager_cls:
        mock_session_manager_cls.create.return_value = mock_session_manager
        
        # Create the toolset without a filter
        params = HttpServerParams(base_url="https://example.com")
        toolset = MCPToolset(
            name="Test Toolset",
            description="Test toolset for MCP tools",
            connection_params=params
        )
        
        # Get the tools
        tools = await toolset.get_tools()
        
        # Verify that we got all tools
        assert len(tools) == 2 