"""Tool specification system for codin agents.

This module provides the core tool specification infrastructure that separates
tool definitions (what they do) from implementations (how they do it).
"""

from __future__ import annotations

import typing as _t
from abc import ABC, abstractmethod
from enum import Enum

import pydantic as _pyd
from pydantic import BaseModel, ConfigDict

__all__ = [
    'ToolType',
    'ToolSpec',
    'ToolSpecRegistry',
    'ExecutionMode',
    'ToolMetadata',
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
    """Tool specification that defines what a tool does, not how it does it."""
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
    
    # Implementation hints (opaque to agents/planners)
    implementation_config: dict[str, _t.Any] = {}
    
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


class SpecLoader(ABC):
    """Abstract base for loading tool specifications."""
    
    @abstractmethod
    async def load_specs(self) -> list[ToolSpec]:
        """Load tool specifications from source."""
        pass


class FileSpecLoader(SpecLoader):
    """Load tool specs from filesystem."""
    
    def __init__(self, paths: list[str]):
        self.paths = paths
    
    async def load_specs(self) -> list[ToolSpec]:
        """Load specs from YAML/JSON files."""
        specs = []
        # Implementation would parse files and create ToolSpec objects
        return specs


class HttpSpecLoader(SpecLoader):
    """Load tool specs from HTTP endpoints."""
    
    def __init__(self, endpoints: list[str]):
        self.endpoints = endpoints
    
    async def load_specs(self) -> list[ToolSpec]:
        """Load specs from HTTP endpoints."""
        specs = []
        # Implementation would fetch from HTTP and create ToolSpec objects
        return specs


class ToolSpecRegistry:
    """Registry for tool specifications (separate from implementations)."""
    
    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._loaders: list[SpecLoader] = []
    
    def add_loader(self, loader: SpecLoader) -> None:
        """Add a spec loader."""
        self._loaders.append(loader)
    
    async def load_all(self) -> None:
        """Load specs from all configured loaders."""
        for loader in self._loaders:
            specs = await loader.load_specs()
            for spec in specs:
                self.register_spec(spec)
    
    def register_spec(self, spec: ToolSpec) -> None:
        """Register a tool specification."""
        if spec.name in self._specs:
            raise ValueError(f"Tool spec already registered: {spec.name}")
        self._specs[spec.name] = spec
    
    def get_spec(self, name: str) -> ToolSpec | None:
        """Get a tool specification by name."""
        return self._specs.get(name)
    
    def list_specs(self, tool_type: ToolType | None = None) -> list[ToolSpec]:
        """List all specs, optionally filtered by type."""
        specs = list(self._specs.values())
        if tool_type:
            specs = [s for s in specs if s.tool_type == tool_type]
        return specs
    
    def to_openai_tools(self) -> list[dict[str, _t.Any]]:
        """Convert all specs to OpenAI function format."""
        return [spec.to_openai_schema() for spec in self._specs.values()]
    
    def to_mcp_tools(self) -> list[dict[str, _t.Any]]:
        """Convert all specs to MCP format."""
        return [spec.to_mcp_schema() for spec in self._specs.values()]