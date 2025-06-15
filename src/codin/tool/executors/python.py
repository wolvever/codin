"""Python function executor for tool execution."""

from __future__ import annotations

import asyncio
import inspect
import typing as _t
from importlib import import_module

from ..base import ToolContext
from ..specs.base import ToolSpec, ToolType
from .base import BaseExecutor

__all__ = ['PythonExecutor']


class PythonExecutor(BaseExecutor):
    """Executor for Python function-based tools."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._functions: dict[str, _t.Callable] = {}
    
    @property
    def supported_types(self) -> set[str]:
        """Return supported tool types."""
        return {ToolType.PYTHON.value}
    
    async def can_execute(self, spec: ToolSpec) -> bool:
        """Check if this executor can handle the tool spec."""
        if spec.tool_type != ToolType.PYTHON:
            return False
        
        # Check if we have the function or can import it
        config = spec.implementation_config
        return bool(
            config.get('function') or 
            (config.get('module') and config.get('function_name'))
        )
    
    async def _setup_tool(self, spec: ToolSpec) -> None:
        """Setup the Python function for execution."""
        config = spec.implementation_config
        
        # If function is directly provided
        if 'function' in config:
            self._functions[spec.name] = config['function']
            return
        
        # If module and function name are provided
        module_name = config.get('module')
        function_name = config.get('function_name')
        
        if module_name and function_name:
            try:
                module = import_module(module_name)
                function = getattr(module, function_name)
                self._functions[spec.name] = function
            except (ImportError, AttributeError) as e:
                raise RuntimeError(f"Failed to import {module_name}.{function_name}: {e}")
        else:
            raise ValueError("Python executor requires 'function' or 'module'+'function_name' in config")
    
    async def _teardown_tool(self, spec: ToolSpec) -> None:
        """Clean up function resources."""
        self._functions.pop(spec.name, None)
    
    async def execute(
        self, 
        spec: ToolSpec, 
        args: dict[str, _t.Any], 
        context: ToolContext
    ) -> _t.Any:
        """Execute the Python function."""
        if spec.name not in self._functions:
            await self.setup_tool(spec)
        
        function = self._functions[spec.name]
        
        # Prepare arguments
        validated_args = await self.validate_args(spec, args)
        
        # Add context if function accepts it
        sig = inspect.signature(function)
        if 'context' in sig.parameters:
            validated_args['context'] = context
        elif 'tool_context' in sig.parameters:
            validated_args['tool_context'] = context
        
        # Execute function
        if inspect.iscoroutinefunction(function):
            return await function(**validated_args)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: function(**validated_args))