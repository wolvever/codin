"""Simple, elegant tool system for codin.

Provides minimal interfaces for tool execution with clear semantics.
"""

import abc
import typing as _t
from enum import Enum

import pydantic as _pyd
from pydantic import BaseModel, ConfigDict

from ..lifecycle import LifecycleMixin, LifecycleState

__all__ = [
    'Tool',
    'Toolset',
    'ToolContext',
    'ToolDefinition',
    'ToolSpec',
    'ToolMetadata',
    'LifecycleState',
    'to_tool_definition',
    'to_tool_definitions',
]


class ToolType(str, Enum):
    """Tool implementation types."""
    PYTHON = "python"
    MCP = "mcp"
    SANDBOX = "sandbox"


class ExecutionMode(str, Enum):
    """Tool execution modes."""
    SYNC = "sync"
    ASYNC = "async"


class ToolMetadata(BaseModel):
    """Tool metadata."""
    model_config = ConfigDict(extra='allow')
    
    version: str = "1.0.0"
    timeout: float | None = None
    dangerous: bool = False


class ToolSpec(BaseModel):
    """Tool specification."""
    model_config = ConfigDict(frozen=True)
    
    name: str
    description: str
    input_schema: dict[str, _t.Any]
    metadata: ToolMetadata = ToolMetadata()
    
    def to_openai_schema(self) -> dict[str, _t.Any]:
        """Convert to OpenAI function schema."""
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.input_schema,
            },
        }
    
    def to_mcp_schema(self) -> dict[str, _t.Any]:
        """Convert to MCP schema."""
        return {
            'name': self.name,
            'description': self.description,
            'inputSchema': self.input_schema,
        }


class ToolDefinition(BaseModel):
    """LLM function definition."""
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, _t.Any]


class ToolContext:
    """Tool execution context."""

    def __init__(
        self,
        session_id: str | None = None,
        working_dir: str = ".",
        timeout: float = 30.0,
    ):
        self.session_id = session_id
        self.working_dir = working_dir
        self.timeout = timeout


class Tool(LifecycleMixin):
    """Base tool interface."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: type[_pyd.BaseModel] | None = None,
        timeout: float | None = None,
    ):
        super().__init__()
        self.name = name
        self.description = description
        self.input_schema = input_schema or _pyd.create_model(f'{name}Schema', __base__=_pyd.BaseModel)
        self.timeout = timeout

    def validate_input(self, args: dict[str, _t.Any]) -> dict[str, _t.Any]:
        """Validate input arguments."""
        return self.input_schema(**args).model_dump()
    
    def get_spec(self) -> ToolSpec:
        """Get tool specification."""
        schema = self.input_schema.model_json_schema()
        input_schema = {
            'type': 'object',
            'properties': schema.get('properties', {}),
        }
        if schema.get('required'):
            input_schema['required'] = schema['required']
        
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=input_schema,
            metadata=ToolMetadata(timeout=self.timeout)
        )

    @abc.abstractmethod
    async def execute(self, args: dict[str, _t.Any], ctx: ToolContext) -> _t.Any:
        """Execute tool with arguments."""
        raise NotImplementedError

    async def _up(self) -> None:
        """Initialize tool."""
        pass

    async def _down(self) -> None:
        """Cleanup tool."""
        pass

    def to_definition(self) -> ToolDefinition:
        """Convert to LLM function definition."""
        schema = self.input_schema.model_json_schema()
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
        )

    def to_mcp_schema(self) -> dict[str, _t.Any]:
        """Convert to MCP tool schema."""
        schema = self.input_schema.model_json_schema()
        mcp_schema = {
            'type': 'object',
            'properties': schema.get('properties', {}),
        }
        if schema.get('required'):
            mcp_schema['required'] = schema['required']
        return {'name': self.name, 'description': self.description, 'inputSchema': mcp_schema}

    def to_openai_schema(self) -> dict[str, _t.Any]:
        """Convert to OpenAI function schema."""
        schema = self.input_schema.model_json_schema()
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
    """Collection of related tools."""

    def __init__(self, name: str, tools: list[Tool] | None = None):
        super().__init__()
        self.name = name
        self.tools = tools or []
        self._tool_map = {tool.name: tool for tool in self.tools}

    def add(self, tool: Tool) -> None:
        """Add tool."""
        self.tools.append(tool)
        self._tool_map[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get tool by name."""
        return self._tool_map.get(name)

    def list_tools(self) -> list[Tool]:
        """List all tools."""
        return self.tools

    def to_definitions(self) -> list[ToolDefinition]:
        """Convert tools to definitions."""
        return [tool.to_definition() for tool in self.tools]

    def to_mcp_schemas(self) -> list[dict[str, _t.Any]]:
        """Convert tools to MCP schemas."""
        return [tool.to_mcp_schema() for tool in self.tools]

    def to_openai_schemas(self) -> list[dict[str, _t.Any]]:
        """Convert tools to OpenAI schemas."""
        return [tool.to_openai_schema() for tool in self.tools]

    async def _up(self) -> None:
        """Initialize all tools."""
        for tool in self.tools:
            await tool.up()

    async def _down(self) -> None:
        """Cleanup all tools."""
        for tool in reversed(self.tools):
            try:
                await tool.down()
            except Exception:
                pass




def to_definition(tool: Tool | ToolDefinition) -> ToolDefinition:
    """Convert to tool definition."""
    if isinstance(tool, ToolDefinition):
        return tool
    return tool.to_definition()


def to_definitions(tools: list[Tool | ToolDefinition] | None) -> list[ToolDefinition]:
    """Convert to tool definitions."""
    if not tools:
        return []
    return [to_definition(tool) for tool in tools]


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------

def to_tool_definition(tool: Tool | ToolDefinition) -> ToolDefinition:
    """Alias for :func:`to_definition` for backwards compatibility."""
    return to_definition(tool)


def to_tool_definitions(tools: list[Tool | ToolDefinition] | None) -> list[ToolDefinition]:
    """Alias for :func:`to_definitions` for backwards compatibility."""
    return to_definitions(tools)

