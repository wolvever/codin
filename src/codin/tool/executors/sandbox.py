"""Sandbox executor for tool execution."""

from __future__ import annotations

import typing as _t

from ..base import ToolContext
from ..specs.base import ToolSpec, ToolType
from .base import BaseExecutor

__all__ = ['SandboxExecutor']


class SandboxExecutor(BaseExecutor):
    """Executor for sandbox-based tools."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sandboxes: dict[str, _t.Any] = {}  # spec.name -> Sandbox instance
    
    @property
    def supported_types(self) -> set[str]:
        """Return supported tool types."""
        return {ToolType.SANDBOX.value, ToolType.SHELL.value}
    
    async def can_execute(self, spec: ToolSpec) -> bool:
        """Check if this executor can handle the tool spec."""
        if spec.tool_type not in {ToolType.SANDBOX, ToolType.SHELL}:
            return False
        
        config = spec.implementation_config
        return bool(
            config.get('sandbox') or 
            config.get('sandbox_type') or
            config.get('command')  # For shell commands
        )
    
    async def _setup_tool(self, spec: ToolSpec) -> None:
        """Setup sandbox for the tool."""
        config = spec.implementation_config
        
        # If sandbox is directly provided
        if 'sandbox' in config:
            sandbox = config['sandbox']
            if hasattr(sandbox, 'up') and sandbox.is_down:
                await sandbox.up()
            self._sandboxes[spec.name] = sandbox
            return
        
        # Create sandbox from config
        from ...sandbox.factory import SandboxFactory
        
        sandbox_type = config.get('sandbox_type', 'local')
        sandbox_config = config.get('sandbox_config', {})
        
        sandbox = await SandboxFactory.create(sandbox_type, **sandbox_config)
        await sandbox.up()
        self._sandboxes[spec.name] = sandbox
    
    async def _teardown_tool(self, spec: ToolSpec) -> None:
        """Clean up sandbox resources."""
        sandbox = self._sandboxes.pop(spec.name, None)
        if sandbox and hasattr(sandbox, 'down'):
            # Note: We might not want to shut down shared sandboxes
            # This depends on the sandbox lifecycle management strategy
            pass
    
    async def execute(
        self, 
        spec: ToolSpec, 
        args: dict[str, _t.Any], 
        context: ToolContext
    ) -> _t.Any:
        """Execute the sandbox tool."""
        if spec.name not in self._sandboxes:
            await self.setup_tool(spec)
        
        sandbox = self._sandboxes[spec.name]
        config = spec.implementation_config
        validated_args = await self.validate_args(spec, args)
        
        # Handle different execution modes
        if spec.tool_type == ToolType.SHELL:
            # Direct shell command execution
            command = config.get('command')
            if not command:
                raise ValueError("Shell tool requires 'command' in config")
            
            # Format command with arguments
            formatted_command = command.format(**validated_args)
            return await sandbox.run_command(formatted_command)
        
        else:
            # Sandbox method execution
            method_name = config.get('method')
            if not method_name:
                raise ValueError("Sandbox tool requires 'method' in config")
            
            method = getattr(sandbox, method_name)
            return await method(**validated_args)