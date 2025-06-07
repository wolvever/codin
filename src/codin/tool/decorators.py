"""Tool decorators for codin agents.

This module provides decorators for easily converting Python functions
into codin tools with automatic schema generation and validation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import typing as _t

import pydantic as _pyd

from .base import Tool, ToolContext
from .registry import ToolRegistry


__all__ = [
    'ToolDecorator',
    'tool',
]

logger = logging.getLogger(__name__)


class ToolDecorator:
    """Decorator class for creating tools from functions."""

    def __init__(
        self,
        name: str = None,
        description: str = None,
        registry: ToolRegistry = None,
        is_generative: bool = False,
        version: str = '1.0.0',
    ):
        """Initialize the tool decorator.

        Args:
            name: Name for the tool, defaults to function name
            description: Description for the tool, defaults to function docstring
            registry: Optional registry to auto-register the tool
            is_generative: Whether this tool produces streaming output
            version: Tool version
        """
        self.name = name
        self.description = description
        self.registry = registry
        self.is_generative = is_generative
        self.version = version

    def __call__(self, func: _t.Callable) -> Tool:
        """Convert a function to a Tool."""
        # Get name and description
        tool_name = self.name or func.__name__
        tool_description = self.description or inspect.getdoc(func) or f'Tool based on function {func.__name__}'

        # Create input schema from function signature
        input_schema = self._create_input_schema(func, tool_name)

        # Capture decorator values
        decorator_version = self.version
        decorator_is_generative = self.is_generative
        decorator_registry = self.registry

        # Create the tool class
        class DecoratedTool(Tool):
            def __init__(self):
                super().__init__(
                    name=tool_name,
                    description=tool_description,
                    version=decorator_version,
                    input_schema=input_schema,
                    is_generative=decorator_is_generative,
                )
                self._func = func

            async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> _t.Any:
                """Execute the decorated function."""
                try:
                    # Prepare arguments for the function
                    func_args = {}
                    sig = inspect.signature(self._func)

                    for param_name, param in sig.parameters.items():
                        if param_name == 'tool_context':
                            func_args[param_name] = tool_context
                        elif param_name in args:
                            func_args[param_name] = args[param_name]
                        elif param.default != inspect.Parameter.empty:
                            func_args[param_name] = param.default

                    # Call the function
                    if inspect.iscoroutinefunction(self._func):
                        result = await self._func(**func_args)
                    else:
                        # Run sync function in executor
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, lambda: self._func(**func_args))

                    return result

                except Exception as e:
                    logger.error(f'Error executing tool {tool_name}: {e}')
                    raise

        # Create tool instance
        tool_instance = DecoratedTool()

        # Auto-register if registry provided
        if decorator_registry:
            decorator_registry.register_tool(tool_instance)

        # Store the tool instance on the function for later access
        func._tool = tool_instance

        return tool_instance

    def _create_input_schema(self, func: _t.Callable, tool_name: str) -> type[_pyd.BaseModel]:
        """Create a Pydantic model from function signature."""
        sig = inspect.signature(func)
        fields = {}

        for param_name, param in sig.parameters.items():
            # Skip special parameters
            if param_name in ('self', 'cls', 'tool_context'):
                continue

            # Get type annotation
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else _t.Any

            # Get default value
            if param.default != inspect.Parameter.empty:
                default = param.default
            else:
                default = ...  # Required field

            # Create field description from parameter name
            description = f'Parameter {param_name}'

            # Add field
            if default is ...:
                fields[param_name] = (annotation, _pyd.Field(..., description=description))
            else:
                fields[param_name] = (annotation, _pyd.Field(default, description=description))

        # Create model class
        model_name = f'{tool_name.title().replace("_", "")}Schema'
        return _pyd.create_model(model_name, **fields)


def tool(
    name: str = None,
    description: str = None,
    registry: ToolRegistry = None,
    is_generative: bool = False,
    version: str = '1.0.0',
) -> _t.Callable[[_t.Callable], Tool]:
    """Decorator to create a tool from a function.

    Args:
        name: Name for the tool, defaults to function name
        description: Description for the tool, defaults to function docstring
        registry: Optional registry to auto-register the tool
        is_generative: Whether this tool produces streaming output
        version: Tool version

    Returns:
        A decorator that converts a function to a Tool

    Example:
        ```python
        @tool(name='my_tool', description='Does something useful')
        def my_function(arg1: str, arg2: int = 42) -> str:
            return f'Got {arg1} and {arg2}'


        # my_function is now a Tool instance
        ```
    """
    return ToolDecorator(
        name=name,
        description=description,
        registry=registry,
        is_generative=is_generative,
        version=version,
    )
