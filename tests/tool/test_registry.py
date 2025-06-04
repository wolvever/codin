"""Tests for tool registry implementation."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from codin.tool.registry import ToolRegistry, ToolRegistryConfig, ToolEndpoint
from codin.tool.base import Tool, Toolset, ToolContext
from codin.tool.core_tools import CoreToolset
from codin.sandbox.local import LocalSandbox


class MockTool(Tool):
    """Mock tool for testing."""
    
    def __init__(self, name: str, description: str = "Mock tool"):
        super().__init__(name=name, description=description)
    
    async def run(self, args: dict, tool_context: ToolContext):
        return {"tool": self.name, "args": args}


@pytest.fixture
async def sandbox():
    """Create and initialize a LocalSandbox for testing."""
    sandbox = LocalSandbox()
    await sandbox.up()
    yield sandbox
    await sandbox.down()


@pytest.fixture
def sample_config():
    return ToolRegistryConfig(
        endpoints=[
            ToolEndpoint(endpoint="fs://./tools", name="local_tools"),
            ToolEndpoint(endpoint="http://localhost:8080", name="remote_tools", enabled=False)
        ],
        tool_prefix_removal=True,
        auto_initialize=False
    )


def test_tool_registry_initialization():
    """Test basic tool registry initialization."""
    registry = ToolRegistry()
    
    assert len(registry.get_tools()) == 0
    assert len(registry.get_toolsets()) == 0


def test_tool_registry_with_config(sample_config):
    """Test tool registry initialization with config."""
    registry = ToolRegistry(sample_config)
    
    assert registry.config == sample_config
    assert registry.config.tool_prefix_removal is True
    assert registry.config.auto_initialize is False


def test_register_single_tool():
    """Test registering a single tool."""
    registry = ToolRegistry()
    tool = MockTool("test_tool")
    
    registry.register_tool(tool)
    
    assert len(registry.get_tools()) == 1
    assert registry.get_tool("test_tool") == tool


@pytest.mark.asyncio
async def test_register_toolset_without_prefix_removal(sandbox):
    """Test registering a toolset without prefix removal."""
    registry = ToolRegistry(ToolRegistryConfig(tool_prefix_removal=False))
    toolset = CoreToolset(sandbox)
    
    registry.register_toolset(toolset)
    
    # Should keep original names with prefixes
    assert registry.get_tool("list_dir") is not None
    assert registry.get_tool("read_file") is not None
    assert registry.get_tool("run_shell") is not None


def test_register_toolset_with_prefix_removal():
    """Test registering a toolset with prefix removal."""
    registry = ToolRegistry(ToolRegistryConfig(tool_prefix_removal=True))
    
    # Create a mock toolset with prefixed tools
    toolset = Toolset("sandbox", "Mock sandbox toolset")
    toolset.add_tool(MockTool("sandbox_exec"))
    toolset.add_tool(MockTool("sandbox_read"))
    
    registry.register_toolset(toolset)
    
    # Should remove prefixes when no conflicts
    assert registry.get_tool("exec") is not None
    assert registry.get_tool("read") is not None
    assert registry.get_tool("sandbox_exec") is not None  # Original names still available


def test_register_toolset_with_conflicts():
    """Test registering toolsets with naming conflicts."""
    registry = ToolRegistry(ToolRegistryConfig(tool_prefix_removal=True))
    
    # Register first toolset
    toolset1 = Toolset("toolset1", "First toolset")
    toolset1.add_tool(MockTool("toolset1_common"))
    registry.register_toolset(toolset1)
    
    # Register second toolset with conflicting simplified name
    toolset2 = Toolset("toolset2", "Second toolset")
    toolset2.add_tool(MockTool("toolset2_common"))
    registry.register_toolset(toolset2)
    
    # First tool should get simplified name, second should keep prefix
    assert registry.get_tool("common") is not None
    assert registry.get_tool("toolset2_common") is not None


def test_get_tools_with_executor():
    """Test getting tools wrapped with executor."""
    from codin.tool.executor import ToolExecutor
    
    registry = ToolRegistry()
    tool = MockTool("test_tool")
    registry.register_tool(tool)
    
    # Without executor
    tools = registry.get_tools_with_executor()
    assert len(tools) == 1
    assert tools[0] == tool
    
    # With executor
    executor = ToolExecutor(registry)
    registry.set_executor(executor)
    
    wrapped_tools = registry.get_tools_with_executor()
    assert len(wrapped_tools) == 1
    # Should be wrapped in GenericTool
    from codin.tool.generic import GenericTool
    assert isinstance(wrapped_tools[0], GenericTool)


@pytest.mark.asyncio
async def test_registry_from_config_file():
    """Test creating registry from config file."""
    config_data = {
        "endpoints": [
            {"endpoint": "fs://./tools", "name": "local"}
        ],
        "tool_prefix_removal": True,
        "auto_initialize": False
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        registry = await ToolRegistry.from_config(config_path)
        assert registry.config.tool_prefix_removal is True
        assert len(registry.config.endpoints) == 1
    finally:
        Path(config_path).unlink()


@pytest.mark.asyncio
async def test_registry_from_endpoint():
    """Test creating registry from single endpoint."""
    registry = await ToolRegistry.from_endpoint("fs://./tools", name="test")
    
    assert len(registry.config.endpoints) == 1
    assert registry.config.endpoints[0].endpoint == "fs://./tools"
    assert registry.config.endpoints[0].name == "test"


@pytest.mark.asyncio
async def test_up_all_tools(sandbox):
    """Test bringing up all tools in registry."""
    config = ToolRegistryConfig(auto_initialize=True)
    registry = ToolRegistry(config)
    
    toolset = CoreToolset(sandbox)
    registry.register_toolset(toolset)
    
    await registry.up()
    
    # Check that tools are up
    for tool in registry.get_tools():
        from codin.lifecycle import LifecycleState
        assert tool.state == LifecycleState.UP


@pytest.mark.asyncio
async def test_down_all_tools(sandbox):
    """Test bringing down all tools in registry."""
    registry = ToolRegistry()
    
    toolset = CoreToolset(sandbox)
    registry.register_toolset(toolset)
    
    # Bring up first
    await registry.up()
    
    # Then bring down
    await registry.down()
    
    # Check that tools are down
    for tool in registry.get_tools():
        from codin.lifecycle import LifecycleState
        assert tool.state == LifecycleState.DOWN


def test_tool_overwrite_warning():
    """Test that overwriting tools generates warnings."""
    import logging
    
    registry = ToolRegistry()
    tool1 = MockTool("test_tool", "First tool")
    tool2 = MockTool("test_tool", "Second tool")
    
    registry.register_tool(tool1)
    
    # This should generate a warning (we can't easily test the warning itself)
    registry.register_tool(tool2)
    
    # Second tool should overwrite first
    assert registry.get_tool("test_tool") == tool2


def test_toolset_overwrite_warning():
    """Test that overwriting toolsets generates warnings."""
    registry = ToolRegistry()
    
    toolset1 = Toolset("test_toolset", "First toolset")
    toolset1.add_tool(MockTool("tool1"))
    
    toolset2 = Toolset("test_toolset", "Second toolset")
    toolset2.add_tool(MockTool("tool2"))
    
    registry.register_toolset(toolset1)
    registry.register_toolset(toolset2)
    
    # Second toolset should overwrite first
    assert registry.get_toolset("test_toolset") == toolset2
    assert registry.get_tool("tool2") is not None


def test_endpoint_configuration():
    """Test endpoint configuration validation."""
    # Valid endpoint
    endpoint = ToolEndpoint(endpoint="http://localhost:8080")
    assert endpoint.endpoint == "http://localhost:8080"
    assert endpoint.enabled is True
    assert endpoint.timeout == 30.0
    
    # Endpoint with auth
    endpoint_with_auth = ToolEndpoint(
        endpoint="http://api.example.com",
        auth={"type": "bearer", "token": "secret"},
        timeout=60.0
    )
    assert endpoint_with_auth.auth["type"] == "bearer"
    assert endpoint_with_auth.timeout == 60.0 