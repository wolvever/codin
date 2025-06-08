"""Tool system for codin agents.

This module provides the core tool infrastructure including base classes,
tool definitions, toolsets, and execution contexts for agent capabilities.
"""

import abc
import typing as _t

import pydantic as _pyd
from pydantic import BaseModel, ConfigDict

from ..lifecycle import LifecycleMixin, LifecycleState

__all__ = [
    'LifecycleState',
    'Tool',
    'ToolContext',
    'ToolDefinition',
    'ToolSpec',
    'Toolset',
]


class ToolDefinition(BaseModel):
    """Tool definition for LLM function calling."""

    model_config = ConfigDict()

    name: str
    description: str
    parameters: dict[str, _t.Any]
    metadata: dict[str, _t.Any] | None = None

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, _t.Any],
        metadata: dict[str, _t.Any] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            metadata=metadata,
        )
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: _t.Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("ToolDefinition is frozen")
        super().__setattr__(name, value)


class ToolContext:
    """Context for tool execution."""

    def __init__(
        self,
        *,
        tool_name: str | None = None,
        arguments: dict[str, _t.Any] | None = None,
        session_id: str | None = None,
        fileids: list[str] | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, _t.Any] | None = None,
    ):
        """Initialize tool execution context.

        Args:
            tool_name: Name of the tool being executed
            arguments: Arguments passed to the tool
            session_id: Session identifier
            fileids: List of file IDs associated with the execution
            tool_call_id: Unique identifier for this tool call
            metadata: Additional metadata for the execution
        """
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.session_id = session_id
        self.fileids = fileids or []
        self.tool_call_id = tool_call_id
        self.metadata = metadata or {}


class Tool(LifecycleMixin):
    """Base class for all tools."""

    def __init__(
        self,
        name: str,
        description: str,
        version: str = '1.0.0',
        input_schema: type[_pyd.BaseModel] | None = None,
        is_generative: bool = False,
    ):
        """Initialize a tool.

        Args:
            name: Name of the tool
            description: Description of what the tool does
            version: Version of the tool
            input_schema: Pydantic model for input validation
            is_generative: Whether the tool generates streaming output
        """
        super().__init__()
        self.name = name
        self.description = description
        self.version = version
        self.input_schema = input_schema or _pyd.create_model(f'{name}Schema', __base__=_pyd.BaseModel)
        self.is_generative = is_generative
        self.metadata = {}

    def validate_input(self, args: dict[str, _t.Any]) -> dict[str, _t.Any]:
        """Validate input against schema and return validated data."""
        return self.input_schema(**args).dict()

    @abc.abstractmethod
    async def run(
        self, args: dict[str, _t.Any], tool_context: ToolContext
    ) -> _t.Any | _t.AsyncGenerator[_t.Any]:
        """Execute the tool with the given arguments."""
        raise NotImplementedError('Tool subclasses must implement run')

    async def _up(self) -> None:
        """Bring the tool up. Override in subclasses if needed."""

    async def _down(self) -> None:
        """Bring the tool down. Override in subclasses if needed."""

    def to_tool_definition(self) -> ToolDefinition:
        """Convert this Tool to a ToolDefinition for LLM function calling."""
        schema = self.input_schema.schema()
        parameters = {
            'type': 'object',
            'properties': schema.get('properties', {}),
        }
        if schema.get('required'):
            parameters['required'] = schema['required']

        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=parameters,
            metadata={'version': self.version, 'is_generative': self.is_generative, **self.metadata},
        )

    def to_mcp_schema(self) -> dict[str, _t.Any]:
        """Convert the tool to an MCP-compatible tool schema.

        The MCP spec requires that tools have an inputSchema with specific format:
        - type: "object" (required)
        - properties: Object mapping property names to JSON Schema definitions
        - required: List of required property names
        """
        schema = self.input_schema.schema()

        # Ensure the schema has the required format
        mcp_schema = {
            'type': 'object',
            'properties': schema.get('properties', {}),
        }

        # Add required properties if they exist
        if schema.get('required'):
            mcp_schema['required'] = schema['required']

        return {'name': self.name, 'description': self.description, 'inputSchema': mcp_schema}

    def to_openai_schema(self) -> dict[str, _t.Any]:
        """Convert the tool to OpenAI function format for LLM function calling."""
        schema = self.input_schema.schema()

        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': {
                    'type': 'object',
                    'properties': schema.get('properties', {}),
                    'required': schema.get('required', []),
                },
            },
        }


class Toolset(LifecycleMixin):
    """Collection of logically related tools."""

    def __init__(
        self,
        name: str,
        description: str,
        tools: list[Tool] | None = None,
    ):
        """Initialize a toolset.

        Args:
            name: Name of the toolset
            description: Description of the toolset
            tools: List of tools in this toolset
        """
        super().__init__()
        self.name = name
        self.description = description
        self.tools = tools or []
        self._tool_map = {tool.name: tool for tool in self.tools}

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to this toolset."""
        self.tools.append(tool)
        self._tool_map[tool.name] = tool

    def get_tool(self, tool_name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tool_map.get(tool_name)

    def get_tools(self) -> list[Tool]:
        """Get all tools in this toolset."""
        return self.tools

    def to_tool_definitions(self) -> list[ToolDefinition]:
        """Convert all tools to ToolDefinition objects."""
        return [tool.to_tool_definition() for tool in self.tools]

    def to_mcp_tools(self) -> list[dict[str, _t.Any]]:
        """Convert all tools to MCP-compatible tool schemas."""
        return [tool.to_mcp_schema() for tool in self.tools]

    def to_openai_tools(self) -> list[dict[str, _t.Any]]:
        """Convert all tools to OpenAI function format for LLM function calling."""
        return [tool.to_openai_schema() for tool in self.tools]

    async def _up(self) -> None:
        """Bring up the toolset and all its tools."""
        for tool in self.tools:
            await tool.up()

    async def _down(self) -> None:
        """Bring down the toolset and all its tools."""
        # Bring down tools in reverse order
        for tool in reversed(self.tools):
            try:
                await tool.down()
            except Exception:
                # Attempt to bring down all tools in the toolset, even if some individual tool.down() calls fail.
                pass  # Continue cleanup even if individual tools fail


class ToolSpec(_t.Protocol):
    """Protocol defining what a Tool specification should support."""

    def to_mcp_schema(self) -> dict[str, _t.Any]: ...
    def to_openai_schema(self) -> dict[str, _t.Any]: ...


def to_tool_definition(tool: Tool | ToolDefinition) -> ToolDefinition:
    """Convert a Tool or ToolDefinition to ToolDefinition."""
    if isinstance(tool, ToolDefinition):
        return tool
    return tool.to_tool_definition()


def to_tool_definitions(tools: list[Tool | ToolDefinition] | None) -> list[ToolDefinition]:
    """Convert a list of Tools/ToolDefinitions to ToolDefinitions."""
    if not tools:
        return []
    return [to_tool_definition(tool) for tool in tools]
