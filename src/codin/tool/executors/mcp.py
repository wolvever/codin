"""MCP server executor for tool execution."""

from __future__ import annotations

import typing as _t

from ..base import ToolContext
from ..specs.base import ToolSpec, ToolType
from .base import BaseExecutor

__all__ = ['MCPExecutor']


class MCPExecutor(BaseExecutor):
    """Executor for MCP server-based tools."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_managers: dict[str, _t.Any] = {}  # spec.name -> MCPSessionManager
    
    @property
    def supported_types(self) -> set[str]:
        """Return supported tool types."""
        return {ToolType.MCP.value}
    
    async def can_execute(self, spec: ToolSpec) -> bool:
        """Check if this executor can handle the tool spec."""
        if spec.tool_type != ToolType.MCP:
            return False
        
        config = spec.implementation_config
        return bool(
            config.get('server_command') or 
            config.get('server_url') or
            config.get('session_manager')
        )
    
    async def _setup_tool(self, spec: ToolSpec) -> None:
        """Setup MCP session for the tool."""
        from ..mcp.session_manager import MCPSessionManager
        from ..mcp import StdioServerParams, HttpServerParams
        
        config = spec.implementation_config
        
        # If session manager is directly provided
        if 'session_manager' in config:
            self._session_managers[spec.name] = config['session_manager']
            return
        
        # Create session manager from config
        session_manager = None
        
        if 'server_command' in config:
            # Stdio server
            params = StdioServerParams(
                command=config['server_command'],
                args=config.get('server_args', []),
                env=config.get('server_env', {})
            )
            session_manager = MCPSessionManager(params)
        
        elif 'server_url' in config:
            # HTTP server
            params = HttpServerParams(
                url=config['server_url'],
                headers=config.get('headers', {})
            )
            session_manager = MCPSessionManager(params)
        
        if session_manager:
            await session_manager.initialize()
            self._session_managers[spec.name] = session_manager
        else:
            raise ValueError("MCP executor requires server configuration")
    
    async def _teardown_tool(self, spec: ToolSpec) -> None:
        """Clean up MCP session."""
        session_manager = self._session_managers.pop(spec.name, None)
        if session_manager:
            await session_manager.close()
    
    async def execute(
        self, 
        spec: ToolSpec, 
        args: dict[str, _t.Any], 
        context: ToolContext
    ) -> _t.Any:
        """Execute the MCP tool."""
        if spec.name not in self._session_managers:
            await self.setup_tool(spec)
        
        session_manager = self._session_managers[spec.name]
        validated_args = await self.validate_args(spec, args)
        
        # Execute via MCP session
        result = await session_manager.call_tool(spec.name, validated_args)
        
        # Process MCP result format
        from ..mcp.conversion_utils import convert_mcp_to_protocol_types
        return convert_mcp_to_protocol_types(result)