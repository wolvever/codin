"""Base executor interfaces for the unified tool execution system."""

from __future__ import annotations

import typing as _t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from ...lifecycle import LifecycleMixin
from ..base import ToolContext
from ..specs.base import ToolSpec

__all__ = [
    'ExecutionResult',
    'ExecutionStatus',
    'BaseExecutor',
    'ExecutorRegistry',
    'ExecutorFactory',
]


class ExecutionStatus(str, Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of tool execution."""
    status: ExecutionStatus
    result: _t.Any = None
    error: Exception | None = None
    duration: float | None = None
    metadata: dict[str, _t.Any] | None = None
    
    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.SUCCESS
    
    @property
    def is_error(self) -> bool:
        """Check if execution failed."""
        return self.status == ExecutionStatus.ERROR


class BaseExecutor(LifecycleMixin):
    """Base class for all tool executors."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_specs: dict[str, ToolSpec] = {}
    
    @property
    @abstractmethod
    def supported_types(self) -> set[str]:
        """Return the tool types this executor can handle."""
        pass
    
    @abstractmethod
    async def can_execute(self, spec: ToolSpec) -> bool:
        """Check if this executor can handle the given tool spec."""
        pass
    
    def add_tool(self, spec: ToolSpec) -> None:
        """Add a tool to this executor (will be setup when executor goes up)."""
        if spec.name in self._tool_specs:
            raise ValueError(f"Tool {spec.name} already added to executor")
        self._tool_specs[spec.name] = spec
    
    def remove_tool(self, spec_name: str) -> None:
        """Remove a tool from this executor (will be torn down if executor is up)."""
        if spec_name not in self._tool_specs:
            raise ValueError(f"Tool {spec_name} not found in executor")
        
        spec = self._tool_specs[spec_name]
        if self.is_up:
            # If executor is running, tear down the tool immediately
            import asyncio
            asyncio.create_task(self._teardown_tool(spec))
        
        del self._tool_specs[spec_name]
    
    def get_tools(self) -> list[ToolSpec]:
        """Get all tools managed by this executor."""
        return list(self._tool_specs.values())
    
    def has_tool(self, spec_name: str) -> bool:
        """Check if executor manages a specific tool."""
        return spec_name in self._tool_specs
    
    @abstractmethod
    async def _setup_tool(self, spec: ToolSpec) -> None:
        """Implementation-specific tool setup logic."""
        pass
    
    @abstractmethod
    async def _teardown_tool(self, spec: ToolSpec) -> None:
        """Implementation-specific tool teardown logic."""
        pass
    
    @abstractmethod
    async def execute(
        self, 
        spec: ToolSpec, 
        args: dict[str, _t.Any], 
        context: ToolContext
    ) -> _t.Any:
        """Execute the tool with given arguments."""
        pass
    
    async def _up(self) -> None:
        """Bring up the executor and setup all tools."""
        # First do executor-level setup
        await self._setup_executor()
        
        # Then setup all added tools
        for spec in self._tool_specs.values():
            try:
                await self._setup_tool(spec)
                self._logger.debug(f"Setup tool: {spec.name}")
            except Exception as e:
                self._logger.error(f"Failed to setup tool {spec.name}: {e}")
                # Continue with other tools
    
    async def _down(self) -> None:
        """Bring down the executor and teardown all tools."""
        # First teardown all tools
        for spec in list(self._tool_specs.values()):
            try:
                await self._teardown_tool(spec)
                self._logger.debug(f"Teardown tool: {spec.name}")
            except Exception as e:
                self._logger.error(f"Error tearing down tool {spec.name}: {e}")
        
        # Then do executor-level cleanup
        await self._teardown_executor()
    
    async def _setup_executor(self) -> None:
        """Override for executor-level setup logic."""
        pass
    
    async def _teardown_executor(self) -> None:
        """Override for executor-level teardown logic."""
        pass
    
    async def validate_args(self, spec: ToolSpec, args: dict[str, _t.Any]) -> dict[str, _t.Any]:
        """Validate arguments against tool specification."""
        return spec.validate_args(args)
    
    async def prepare_context(self, spec: ToolSpec, context: ToolContext) -> ToolContext:
        """Prepare execution context (can be overridden by subclasses)."""
        context.tool_name = spec.name
        return context


class ExecutorRegistry:
    """Registry for tool executors."""
    
    def __init__(self):
        self._executors: dict[str, BaseExecutor] = {}
        self._type_mapping: dict[str, str] = {}  # tool_type -> executor_name
    
    def register_executor(self, name: str, executor: BaseExecutor) -> None:
        """Register an executor."""
        self._executors[name] = executor
        
        # Map supported types to this executor
        for tool_type in executor.supported_types:
            if tool_type in self._type_mapping:
                raise ValueError(f"Tool type {tool_type} already handled by {self._type_mapping[tool_type]}")
            self._type_mapping[tool_type] = name
    
    def get_executor(self, tool_type: str) -> BaseExecutor | None:
        """Get executor for a tool type."""
        executor_name = self._type_mapping.get(tool_type)
        if executor_name:
            return self._executors[executor_name]
        return None
    
    async def find_executor(self, spec: ToolSpec) -> BaseExecutor | None:
        """Find the best executor for a tool spec."""
        # First try by tool type
        executor = self.get_executor(spec.tool_type.value)
        if executor and await executor.can_execute(spec):
            return executor
        
        # Fallback: check all executors
        for executor in self._executors.values():
            if await executor.can_execute(spec):
                return executor
        
        return None


class ExecutorFactory:
    """Factory for creating and configuring executors."""
    
    @staticmethod
    def create_registry() -> ExecutorRegistry:
        """Create a registry with default executors."""
        registry = ExecutorRegistry()
        
        # Register default executors
        from .python import PythonExecutor
        from .mcp import MCPExecutor
        from .sandbox import SandboxExecutor
        
        registry.register_executor("python", PythonExecutor())
        registry.register_executor("mcp", MCPExecutor())
        registry.register_executor("sandbox", SandboxExecutor())
        
        return registry