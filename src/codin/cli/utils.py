"""Utility functions for CLI operations.

This module provides utility functions for command-line interface operations,
including MCP toolset creation from configuration files.
"""

import click

from ..config import get_config
from ..tool.base import Toolset
from ..tool.mcp import MCPToolset, SseServerParams, StdioServerParams

__all__ = [
    'create_mcp_toolsets_from_config',
]


def create_mcp_toolsets_from_config(config_file: str | None = None) -> list[Toolset]:
    """Create MCP toolsets from configuration.

    Args:
        config_file: Optional path to custom config file
    """
    config = get_config(config_file)
    toolsets = []

    for server_name, server_config in config.mcp_servers.items():
        try:
            if server_config.url:
                # SSE server
                connection_params = SseServerParams(url=server_config.url)
            else:
                # Stdio server
                connection_params = StdioServerParams(
                    command=server_config.command, args=server_config.args, env=server_config.env
                )

            # Create MCP toolset
            mcp_toolset = MCPToolset(
                name=f'mcp_{server_name}',
                description=server_config.description or f'MCP server: {server_name}',
                connection_params=connection_params,
            )

            toolsets.append(mcp_toolset)

        except Exception as e:
            click.echo(f"[WARN] Failed to create MCP toolset for '{server_name}': {e}", err=True)
            # Continue creating other toolsets even if one fails

    return toolsets
