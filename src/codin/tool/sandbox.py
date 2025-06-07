"""Sandbox tool integration for codin agents.

This module provides tools that integrate with various sandbox environments
for secure code execution and testing.
"""

from __future__ import annotations

import inspect
import typing as _t

import pydantic as _pyd

from ..sandbox.base import Sandbox
from .base import Tool, ToolContext, Toolset


__all__ = [
    'SandboxTool',
    'SandboxToolset',
]


class SandboxTool(Tool):
    """Base class for sandbox tools."""

    def __init__(
        self,
        name: str,
        description: str,
        sandbox: Sandbox,
        input_schema: type[_pyd.BaseModel],
        is_generative: bool = False,
    ):
        """Initialize a sandbox tool.

        Parameters
        ----------
        name : str
            The name of the tool
        description : str
            A human-readable description of what the tool does
        sandbox : Sandbox
            The sandbox instance to use
        input_schema : Type[_pyd.BaseModel]
            The Pydantic model for validating inputs
        is_generative : bool, optional
            Whether this tool produces streaming output, by default False
        """
        super().__init__(name=name, description=description, input_schema=input_schema, is_generative=is_generative)
        self.sandbox = sandbox

    async def _up(self) -> None:
        """Bring up the sandbox tool by checking sandbox availability."""
        # Check if sandbox is running/available
        if self.sandbox.is_down:
            # Try to start the sandbox if it's not running
            await self.sandbox.up()

    async def _down(self) -> None:
        """Bring down the sandbox tool."""
        # Note: We don't stop the sandbox here as it might be shared
        # The sandbox lifecycle is managed separately


class SandboxMethodTool(Tool):
    """A tool that wraps a sandbox method."""

    def __init__(self, name: str, method: _t.Callable, sandbox: Sandbox):
        """Initialize a tool from a sandbox method.

        Parameters
        ----------
        name : str
            The name of the tool (without 'tool_' prefix)
        method : Callable
            The sandbox method to wrap
        sandbox : Sandbox
            The sandbox instance
        """
        # Get method signature and create input schema
        sig = inspect.signature(method)
        fields = {}

        for param_name, param in sig.parameters.items():
            # Skip 'self' parameter
            if param_name == 'self':
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

        # Create input schema
        model_name = f'{name.title().replace("_", "")}Input'
        input_schema = _pyd.create_model(model_name, **fields)

        # Get description from method docstring
        description = inspect.getdoc(method) or f'Tool based on sandbox method {method.__name__}'

        super().__init__(
            name=name,
            description=description,
            input_schema=input_schema,
        )

        self.method = method
        self.sandbox = sandbox

    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> _t.Any:
        """Execute the sandbox method."""
        # Call the method with the provided arguments
        return await self.method(**args)

    async def _up(self) -> None:
        """Bring up the tool by ensuring sandbox is available."""
        if self.sandbox.is_down:
            await self.sandbox.up()

    async def _down(self) -> None:
        """Bring down the tool."""


class SandboxToolset(Toolset):
    """A toolset containing sandbox tools automatically generated from sandbox methods."""

    def __init__(self, sandbox: Sandbox):
        """Initialize a sandbox toolset for the given sandbox.

        This will automatically discover and create tools for all methods in the
        sandbox that start with 'tool_'.

        Parameters
        ----------
        sandbox : Sandbox
            The sandbox instance to create tools for
        """
        # Get all tool methods from the sandbox
        tool_methods = sandbox.get_tool_methods()

        # Create Tool instances for each method
        tools = []
        for tool_name, method in tool_methods.items():
            tool_instance = SandboxMethodTool(tool_name, method, sandbox)
            tools.append(tool_instance)

        super().__init__(
            name='sandbox',
            description='Tools for interacting with a sandbox environment (auto-generated from sandbox methods)',
            tools=tools,
        )

        # Store sandbox reference
        self.sandbox = sandbox

    async def _up(self) -> None:
        """Bring up the sandbox toolset by checking sandbox availability."""
        # Check if sandbox is running/available
        if self.sandbox.is_down:
            # Try to start the sandbox if it's not running
            await self.sandbox.up()

        # Bring up all tools
        await super()._up()

    async def _down(self) -> None:
        """Bring down the sandbox toolset and all its tools."""
        # Bring down all tools first
        await super()._down()

        # Note: We don't stop the sandbox here as it might be shared
        # The sandbox lifecycle is managed separately

    async def cleanup(self) -> None:
        """Cleanup the sandbox toolset and all its tools."""
        # Cleanup all tools
        for tool in self.tools:
            try:
                if hasattr(tool, 'cleanup'):
                    await tool.cleanup()
                elif hasattr(tool, '_down'):
                    await tool._down()
            except Exception:
                pass  # Ignore cleanup errors

        # Note: We don't stop the sandbox here as it might be shared
        # The sandbox lifecycle is managed separately
