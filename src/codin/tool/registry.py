"""Tool registry for managing and discovering tools.

This module provides a centralized registry for tools and toolsets,
supporting loading from various sources including filesystem and HTTP endpoints.
It handles tool name resolution, conflict detection, and lifecycle management.
"""

import json
import logging
import typing as _t

from pathlib import Path
from urllib.parse import urlparse

import httpx
import pydantic as _pyd
import yaml

from .base import Tool, Toolset


__all__ = [
    'ToolEndpoint',
    'ToolRegistry',
    'ToolRegistryConfig',
]


class ToolEndpoint(_pyd.BaseModel):
    """Configuration for a tool endpoint."""

    endpoint: str = _pyd.Field(..., description='Endpoint URL (fs://path, http://host:port, etc.)')
    name: str = _pyd.Field(None, description='Optional name for the endpoint')
    auth: dict[str, _t.Any] | None = _pyd.Field(None, description='Authentication configuration')
    timeout: float = _pyd.Field(30.0, description='Request timeout in seconds')
    enabled: bool = _pyd.Field(True, description='Whether this endpoint is enabled')


class ToolRegistryConfig(_pyd.BaseModel):
    """Configuration for the tool registry."""

    endpoints: list[ToolEndpoint] = _pyd.Field(default_factory=list, description='List of tool endpoints')
    sandbox: dict[str, _t.Any] | None = _pyd.Field(None, description='Sandbox configuration')
    tool_prefix_removal: bool = _pyd.Field(True, description='Remove toolset prefixes when no conflicts')
    auto_initialize: bool = _pyd.Field(True, description='Automatically initialize tools on load')


class ToolRegistry:
    """Registry for all available tools and toolsets.

    The registry can be initialized from config files, remote HTTP services,
    or programmatically. It supports loading tools with or without a tool executor.
    """

    def __init__(self, config: ToolRegistryConfig | None = None):
        self._toolsets = {}  # name -> toolset
        self._tools = {}  # name -> tool
        self._original_names = {}  # simplified_name -> original_name
        self.config = config or ToolRegistryConfig()
        self.logger = logging.getLogger(__name__)
        self._executor = None

    @classmethod
    async def from_config(cls, config_path: str | Path) -> 'ToolRegistry':
        """Create a tool registry from a configuration file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_path}')

        # Load config based on file extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path) as f:
                config_data = json.load(f)
        else:
            raise ValueError(f'Unsupported config file format: {config_path.suffix}')

        config = ToolRegistryConfig(**config_data)
        registry = cls(config)

        # Load tools from endpoints
        await registry._load_from_endpoints()

        return registry

    @classmethod
    async def from_endpoint(cls, endpoint: str, **kwargs) -> 'ToolRegistry':
        """Create a tool registry from a single endpoint."""
        endpoint_config = ToolEndpoint(endpoint=endpoint, **kwargs)
        config = ToolRegistryConfig(endpoints=[endpoint_config])
        registry = cls(config)
        await registry._load_from_endpoints()
        return registry

    async def _load_from_endpoints(self) -> None:
        """Load tools from configured endpoints."""
        for endpoint_config in self.config.endpoints:
            if not endpoint_config.enabled:
                continue

            try:
                await self._load_from_endpoint(endpoint_config)
            except Exception as e:
                self.logger.error(f'Failed to load from endpoint {endpoint_config.endpoint}: {e}')

    async def _load_from_endpoint(self, endpoint_config: ToolEndpoint) -> None:
        """Load tools from a specific endpoint."""
        parsed = urlparse(endpoint_config.endpoint)

        if parsed.scheme == 'fs':
            # File system endpoint
            await self._load_from_filesystem(parsed.path)
        elif parsed.scheme in ['http', 'https']:
            # HTTP endpoint
            await self._load_from_http(endpoint_config)
        else:
            raise ValueError(f'Unsupported endpoint scheme: {parsed.scheme}')

    async def _load_from_filesystem(self, path: str) -> None:
        """Load tools from filesystem path."""
        # This would load tools from a directory structure
        # For now, just log that it's not implemented
        self.logger.warning(f'Filesystem tool loading not yet implemented: {path}')

    async def _load_from_http(self, endpoint_config: ToolEndpoint) -> None:
        """Load tools from HTTP endpoint."""
        async with httpx.AsyncClient(timeout=endpoint_config.timeout) as client:
            # Add authentication if configured
            headers = {}
            if endpoint_config.auth:
                if endpoint_config.auth.get('type') == 'bearer':
                    headers['Authorization'] = f'Bearer {endpoint_config.auth["token"]}'
                elif endpoint_config.auth.get('type') == 'api_key':
                    headers[endpoint_config.auth['header']] = endpoint_config.auth['key']

            # Fetch tool definitions
            response = await client.get(f'{endpoint_config.endpoint}/tools', headers=headers)
            response.raise_for_status()

            tools_data = response.json()

            # Create remote tools (simplified implementation)
            for tool_data in tools_data.get('tools', []):
                # This would create RemoteTool instances that delegate to the HTTP endpoint
                self.logger.info(f'Would load remote tool: {tool_data.get("name")}')

    def set_executor(self, executor) -> None:
        """Set the tool executor for this registry."""
        self._executor = executor

    def register_toolset(self, toolset: Toolset, remove_prefix: bool = None) -> None:
        """Register a toolset.

        Args:
            toolset: The toolset to register
            remove_prefix: Whether to remove toolset prefix from tool names when no conflicts.
                          If None, uses the registry config setting.
        """
        if toolset.name in self._toolsets:
            self.logger.warning(f'Overwriting existing toolset: {toolset.name}')
        self._toolsets[toolset.name] = toolset

        # Register all tools in the toolset
        should_remove_prefix = remove_prefix if remove_prefix is not None else self.config.tool_prefix_removal

        for tool in toolset.tools:
            self.register_tool(tool, toolset_name=toolset.name, remove_prefix=should_remove_prefix)

    def register_tool(self, tool: Tool, toolset_name: str = None, remove_prefix: bool = False) -> None:
        """Register a single tool.

        Args:
            tool: The tool to register
            toolset_name: Name of the toolset this tool belongs to
            remove_prefix: Whether to try removing toolset prefix from tool name
        """
        original_name = tool.name
        final_name = original_name

        # Always register with original name first
        self._tools[original_name] = tool

        # Try to remove prefix if requested and no conflicts
        if remove_prefix and toolset_name and tool.name.startswith(f'{toolset_name}_'):
            simplified_name = tool.name[len(f'{toolset_name}_') :]

            # Only use simplified name if it doesn't conflict
            if simplified_name not in self._tools:
                final_name = simplified_name
                self._original_names[simplified_name] = original_name

                # Create a copy with the new name and register it too
                tool_copy = type(tool).__new__(type(tool))
                tool_copy.__dict__.update(tool.__dict__)
                tool_copy.name = final_name
                self._tools[final_name] = tool_copy

        if final_name in self._tools and final_name != original_name and self._tools[final_name].name != original_name:
            self.logger.warning(f'Tool name conflict: {final_name} (original: {original_name})')

    def get_toolset(self, name: str) -> Toolset | None:
        """Get a toolset by name."""
        return self._toolsets.get(name)

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_toolsets(self) -> list[Toolset]:
        """Get all registered toolsets."""
        return list(self._toolsets.values())

    def get_tools_with_executor(self) -> list[Tool]:
        """Get all tools wrapped with the executor if available."""
        if self._executor is None:
            return self.get_tools()

        # Return tools that delegate to the executor
        from .generic import GenericTool

        wrapped_tools = []

        for tool in self.get_tools():
            wrapped_tool = GenericTool(
                name=tool.name,
                description=tool.description,
                executor=self._executor,
                target_tool_name=tool.name,
                input_schema=tool.input_schema,
                is_generative=tool.is_generative,
            )
            wrapped_tools.append(wrapped_tool)

        return wrapped_tools

    async def up(self) -> None:
        """Bring up all registered toolsets and tools."""
        if not self.config.auto_initialize:
            return

        # Bring up toolsets first
        for toolset in self._toolsets.values():
            try:
                await toolset.up()
                self.logger.info(f'Started toolset: {toolset.name}')
            except Exception as e:
                self.logger.error(f'Failed to start toolset {toolset.name}: {e}')

        # Bring up standalone tools
        for tool in self._tools.values():
            if not any(tool in ts.tools for ts in self._toolsets.values()):
                try:
                    await tool.up()
                    self.logger.info(f'Started tool: {tool.name}')
                except Exception as e:
                    self.logger.error(f'Failed to start tool {tool.name}: {e}')

    async def down(self) -> None:
        """Bring down all registered toolsets and tools."""
        # Bring down toolsets in reverse order
        for toolset in reversed(list(self._toolsets.values())):
            try:
                await toolset.down()
            except Exception as e:
                self.logger.error(f'Failed to stop toolset {toolset.name}: {e}')

        # Bring down standalone tools
        for tool in self._tools.values():
            if not any(tool in ts.tools for ts in self._toolsets.values()):
                try:
                    await tool.down()
                except Exception as e:
                    self.logger.error(f'Failed to stop tool {tool.name}: {e}')
