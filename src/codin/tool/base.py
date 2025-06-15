"""Tool system for codin agents.

This module provides the core tool infrastructure including base classes,
tool definitions, toolsets, and execution contexts for agent capabilities.
"""

import abc
import typing as _t
from enum import Enum

import pydantic as _pyd
from pydantic import BaseModel, ConfigDict

from ..lifecycle import LifecycleMixin, LifecycleState

__all__ = [
    'LifecycleState',
    'Tool',
    'ToolContext',
    'ToolDefinition',
    'ToolSpec',
    'ToolType',
    'ExecutionMode',
    'ToolMetadata',
    'Toolset',
]


class ToolType(str, Enum):
    """Types of tool implementations."""
    PYTHON = "python"
    MCP = "mcp"
    SANDBOX = "sandbox"
    HTTP = "http"
    SHELL = "shell"


class ExecutionMode(str, Enum):
    """Execution modes for tools."""
    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"


class ToolMetadata(BaseModel):
    """Metadata for tool specifications."""
    model_config = ConfigDict(extra='allow')
    
    version: str = "1.0.0"
    author: str | None = None
    category: str | None = None
    tags: list[str] = []
    documentation_url: str | None = None
    source_url: str | None = None
    
    # Execution hints
    estimated_duration: float | None = None  # seconds
    requires_approval: bool = False
    is_dangerous: bool = False
    
    # Dependencies
    requires_tools: list[str] = []
    conflicts_with: list[str] = []


class ToolSpec(BaseModel):
    """Tool specification that defines what a tool does."""
    model_config = ConfigDict(frozen=True)
    
    # Core identification
    name: str
    description: str
    tool_type: ToolType
    
    # Schema definition
    input_schema: dict[str, _t.Any]
    output_schema: dict[str, _t.Any] | None = None
    
    # Execution properties
    execution_mode: ExecutionMode = ExecutionMode.ASYNC
    timeout: float | None = None
    retries: int = 0
    
    # Metadata
    metadata: ToolMetadata = ToolMetadata()
    
    def to_openai_schema(self) -> dict[str, _t.Any]:
        """Convert to OpenAI function calling format."""
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.input_schema,
            },
        }
    
    def to_mcp_schema(self) -> dict[str, _t.Any]:
        """Convert to MCP tool format."""
        return {
            'name': self.name,
            'description': self.description,
            'inputSchema': self.input_schema,
        }
    
    def validate_args(self, args: dict[str, _t.Any]) -> dict[str, _t.Any]:
        """Validate arguments against input schema."""
        # Create temporary pydantic model for validation
        model_name = f"{self.name.title()}Args"
        temp_model = _pyd.create_model(
            model_name,
            **{
                prop: (self._get_python_type(prop_def), ...)
                for prop, prop_def in self.input_schema.get('properties', {}).items()
            }
        )
        return temp_model(**args).model_dump()
    
    def _get_python_type(self, prop_def: dict) -> type:
        """Convert JSON schema type to Python type."""
        type_map = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict,
        }
        return type_map.get(prop_def.get('type', 'string'), str)


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
        tool_type: ToolType = ToolType.PYTHON,
        version: str = '1.0.0',
        input_schema: type[_pyd.BaseModel] | None = None,
        is_generative: bool = False,
        execution_mode: ExecutionMode = ExecutionMode.ASYNC,
        timeout: float | None = None,
        retries: int = 0,
        metadata: dict[str, _t.Any] | None = None,
    ):
        """Initialize a tool.

        Args:
            name: Name of the tool
            description: Description of what the tool does
            tool_type: Type of tool implementation
            version: Version of the tool
            input_schema: Pydantic model for input validation
            is_generative: Whether the tool generates streaming output
            execution_mode: How the tool should be executed
            timeout: Maximum execution time in seconds
            retries: Number of retry attempts on failure
            metadata: Additional metadata for the tool
        """
        super().__init__()
        self.name = name
        self.description = description
        self.tool_type = tool_type
        self.version = version
        self.input_schema = input_schema or _pyd.create_model(f'{name}Schema', __base__=_pyd.BaseModel)
        self.is_generative = is_generative
        self.execution_mode = execution_mode
        self.timeout = timeout
        self.retries = retries
        self.metadata = metadata or {}

    def validate_input(self, args: dict[str, _t.Any]) -> dict[str, _t.Any]:
        """Validate input against schema and return validated data."""
        return self.input_schema(**args).dict()
    
    def get_spec(self) -> ToolSpec:
        """Get the tool specification."""
        # Convert Pydantic model schema to JSON schema
        schema = self.input_schema.schema()
        input_schema = {
            'type': 'object',
            'properties': schema.get('properties', {}),
        }
        if schema.get('required'):
            input_schema['required'] = schema['required']
        
        return ToolSpec(
            name=self.name,
            description=self.description,
            tool_type=self.tool_type,
            input_schema=input_schema,
            execution_mode=self.execution_mode,
            timeout=self.timeout,
            retries=self.retries,
            metadata=ToolMetadata(
                version=self.version,
                category=self.metadata.get('category'),
                tags=self.metadata.get('tags', []),
                estimated_duration=self.metadata.get('estimated_duration'),
                requires_approval=self.metadata.get('requires_approval', False),
                is_dangerous=self.metadata.get('is_dangerous', False),
                **{k: v for k, v in self.metadata.items() 
                   if k not in ['category', 'tags', 'estimated_duration', 'requires_approval', 'is_dangerous']}
            )
        )

    @abc.abstractmethod
    async def run(
        self, args: dict[str, _t.Any], tool_context: ToolContext
    ) -> _t.Any | _t.AsyncGenerator[_t.Any, None]:
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
