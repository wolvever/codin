"""Generic tool creation utilities for codin agents.

This module provides utilities for creating tools from Python functions
and other generic tool creation patterns.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import typing as _t

import pydantic as _pyd

from .base import Tool, ToolContext
from .executor import ToolExecutor


__all__ = [
    'GenericTool',
    'create_tool_from_function',
]


class GenericTool(Tool):
    """A generic tool implementation that delegates to a ToolExecutor.

    This tool acts as a bridge between the tool interface and the actual execution
    logic handled by the ToolExecutor. It can be used to create tools that need
    access to the ToolExecutor's features like retry logic, hooks, and metrics.
    """

    def __init__(
        self,
        name: str,
        description: str,
        executor: ToolExecutor,
        target_tool_name: str,
        input_schema: type[_pyd.BaseModel] | None = None,
        is_generative: bool = False,
    ):
        """Initialize a generic tool.

        Parameters
        ----------
        name : str
            Name of the tool, this may differ from the target_tool_name
        description : str
            Human-readable description of the tool
        executor : ToolExecutor
            The tool executor instance to use for executing the target tool
        target_tool_name : str
            Name of the actual tool to execute via the executor
        input_schema : Type[_pyd.BaseModel], optional
            Optional Pydantic model for validating inputs, if None, uses the
            input schema of the target tool
        is_generative : bool, optional
            Whether this tool produces streaming/generative output
        """
        # If no input schema provided, get it from the target tool
        if input_schema is None:
            target_tool = executor.registry.get_tool(target_tool_name)
            if target_tool is None:
                raise ValueError(f"Target tool '{target_tool_name}' not found in registry")
            input_schema = target_tool.input_schema

        super().__init__(
            name=name,
            description=description,
            input_schema=input_schema,
            is_generative=is_generative,
        )
        self.executor = executor
        self.target_tool_name = target_tool_name
        self.logger = logging.getLogger(__name__)

    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> _t.Any:
        """Execute the target tool via the executor."""
        self.logger.debug(f"Executing target tool '{self.target_tool_name}' via executor")
        try:
            result = await self.executor.execute(
                self.target_tool_name,
                args,
                tool_context,
            )
            return result
        except Exception as e:
            self.logger.error(f"Error executing target tool '{self.target_tool_name}': {e!s}")
            # Rethrow to let executor handle retries if configured
            raise


def create_tool_from_function(
    func: _t.Callable,
    name: str = None,
    description: str = None,
    executor: ToolExecutor = None,
    is_generative: bool = False,
) -> GenericTool:
    """Create a tool from a Python function.

    This utility helps in creating tools from existing functions by:
    1. Generating a pydantic schema from the function's type annotations
    2. Extracting docstring for description
    3. Registering the function with the ToolExecutor
    4. Creating a GenericTool that delegates to this function

    Parameters
    ----------
    func : Callable
        The function to convert to a tool
    name : str, optional
        Name for the tool, defaults to function name
    description : str, optional
        Description for the tool, defaults to function docstring
    executor : ToolExecutor, optional
        Executor to register with, required unless just creating schema
    is_generative : bool, optional
        Whether this tool produces streaming output

    Returns:
    -------
    GenericTool
        A tool that delegates to the provided function
    """
    if name is None:
        name = func.__name__

    if description is None:
        description = inspect.getdoc(func) or f'Tool based on function {func.__name__}'

    # Create a pydantic model from function params
    sig = inspect.signature(func)
    fields = {}

    for param_name, param in sig.parameters.items():
        # Skip self, cls, and context parameters
        if param_name in ('self', 'cls', 'tool_context'):
            continue

        # Get annotation, default to Any
        annotation = param.annotation if param.annotation != inspect.Parameter.empty else _t.Any

        # Get default
        default = param.default if param.default != inspect.Parameter.empty else ...

        # Add field
        fields[param_name] = (annotation, default)

    # Create model class
    model_name = f'{name.title().replace("_", "")}Schema'
    input_schema = _pyd.create_model(model_name, **fields)

    # If no executor provided, just return the tool with schema
    if executor is None:

        class FunctionTool(Tool):
            async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> _t.Any:
                # Call the function with arguments
                if inspect.iscoroutinefunction(func):
                    return await func(**args, tool_context=tool_context)
                # Run in executor for sync functions
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: func(**args, tool_context=tool_context))

            async def _up(self) -> None:
                """Function-based tools don't need special startup."""

            async def _down(self) -> None:
                """Function-based tools don't need special shutdown."""

        return FunctionTool(
            name=name,
            description=description,
            input_schema=input_schema,
            is_generative=is_generative,
        )

    # Register function as a tool with the registry
    class FunctionBasedTool(Tool):
        async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> _t.Any:
            # Call the function with arguments
            if inspect.iscoroutinefunction(func):
                return await func(**args, tool_context=tool_context)
            # Run in executor for sync functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**args, tool_context=tool_context))

        async def _up(self) -> None:
            """Function-based tools don't need special startup."""

        async def _down(self) -> None:
            """Function-based tools don't need special shutdown."""

    tool_impl = FunctionBasedTool(
        name=name,
        description=description,
        input_schema=input_schema,
        is_generative=is_generative,
    )

    # Register with the registry
    executor.registry.register_tool(tool_impl)

    # Create and return the generic tool that delegates to our impl
    return GenericTool(
        name=name,
        description=description,
        executor=executor,
        target_tool_name=name,
        input_schema=input_schema,
        is_generative=is_generative,
    )
