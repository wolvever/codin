"""Extension system for cross-cutting concerns in tool execution."""

from __future__ import annotations

import typing as _t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from ..base import ToolContext
from ..specs.base import ToolSpec
from ..executors.base import ExecutionResult

__all__ = [
    'ExtensionPriority',
    'Extension',
    'ExtensionManager',
    'ExtensionChain',
    'ExtensionContext',
]


class ExtensionPriority(int, Enum):
    """Priority levels for extension execution order."""
    HIGHEST = 1000
    HIGH = 750
    NORMAL = 500
    LOW = 250
    LOWEST = 0


@dataclass
class ExtensionContext:
    """Context passed to extensions during tool execution."""
    spec: ToolSpec
    args: dict[str, _t.Any]
    tool_context: ToolContext
    execution_metadata: dict[str, _t.Any] = field(default_factory=dict)
    
    def set_metadata(self, key: str, value: _t.Any) -> None:
        """Set metadata that can be shared between extensions."""
        self.execution_metadata[key] = value
    
    def get_metadata(self, key: str, default: _t.Any = None) -> _t.Any:
        """Get metadata set by other extensions."""
        return self.execution_metadata.get(key, default)


class Extension(ABC):
    """Base class for tool execution extensions."""
    
    def __init__(self, priority: ExtensionPriority = ExtensionPriority.NORMAL):
        self.priority = priority
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the extension name."""
        pass
    
    async def before_execute(self, ctx: ExtensionContext) -> None:
        """Called before tool execution."""
        pass
    
    async def after_execute(self, ctx: ExtensionContext, result: ExecutionResult) -> ExecutionResult:
        """Called after successful tool execution. Can modify the result."""
        return result
    
    async def on_error(self, ctx: ExtensionContext, error: Exception) -> Exception | None:
        """Called when tool execution fails. Return None to suppress error."""
        return error
    
    async def on_timeout(self, ctx: ExtensionContext) -> None:
        """Called when tool execution times out."""
        pass
    
    async def on_cancelled(self, ctx: ExtensionContext) -> None:
        """Called when tool execution is cancelled."""
        pass


class ExtensionManager:
    """Manages and orchestrates extensions."""
    
    def __init__(self):
        self._extensions: list[Extension] = []
        self._extension_map: dict[str, Extension] = {}
    
    def register(self, extension: Extension) -> None:
        """Register an extension."""
        if extension.name in self._extension_map:
            raise ValueError(f"Extension '{extension.name}' already registered")
        
        self._extensions.append(extension)
        self._extension_map[extension.name] = extension
        
        # Sort by priority (highest first)
        self._extensions.sort(key=lambda e: e.priority.value, reverse=True)
    
    def unregister(self, name: str) -> None:
        """Unregister an extension by name."""
        extension = self._extension_map.pop(name, None)
        if extension:
            self._extensions.remove(extension)
    
    def get_extension(self, name: str) -> Extension | None:
        """Get an extension by name."""
        return self._extension_map.get(name)
    
    def list_extensions(self) -> list[Extension]:
        """List all registered extensions in priority order."""
        return self._extensions.copy()
    
    async def before_execute(self, ctx: ExtensionContext) -> None:
        """Run all before_execute hooks."""
        for extension in self._extensions:
            try:
                await extension.before_execute(ctx)
            except Exception as e:
                # Extensions should not break tool execution
                # Log error but continue
                import logging
                logging.getLogger(__name__).error(
                    f"Extension '{extension.name}' failed in before_execute: {e}"
                )
    
    async def after_execute(self, ctx: ExtensionContext, result: ExecutionResult) -> ExecutionResult:
        """Run all after_execute hooks."""
        current_result = result
        
        for extension in self._extensions:
            try:
                current_result = await extension.after_execute(ctx, current_result)
            except Exception as e:
                # Extensions should not break tool execution
                import logging
                logging.getLogger(__name__).error(
                    f"Extension '{extension.name}' failed in after_execute: {e}"
                )
        
        return current_result
    
    async def on_error(self, ctx: ExtensionContext, error: Exception) -> Exception | None:
        """Run all error hooks. Return None if error should be suppressed."""
        current_error = error
        
        for extension in self._extensions:
            try:
                current_error = await extension.on_error(ctx, current_error)
                if current_error is None:
                    # Error was suppressed
                    break
            except Exception as e:
                # Extension error handling failed, log but continue
                import logging
                logging.getLogger(__name__).error(
                    f"Extension '{extension.name}' failed in on_error: {e}"
                )
        
        return current_error
    
    async def on_timeout(self, ctx: ExtensionContext) -> None:
        """Run all timeout hooks."""
        for extension in self._extensions:
            try:
                await extension.on_timeout(ctx)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    f"Extension '{extension.name}' failed in on_timeout: {e}"
                )
    
    async def on_cancelled(self, ctx: ExtensionContext) -> None:
        """Run all cancellation hooks."""
        for extension in self._extensions:
            try:
                await extension.on_cancelled(ctx)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    f"Extension '{extension.name}' failed in on_cancelled: {e}"
                )


class ExtensionChain:
    """Utility for creating extension chains with specific orderings."""
    
    @staticmethod
    def create_default_chain() -> ExtensionManager:
        """Create extension manager with default extensions."""
        manager = ExtensionManager()
        
        # Register default extensions in priority order
        from .logging import LoggingExtension
        from .metrics import MetricsExtension
        from .approval import ApprovalExtension
        from .auth import AuthExtension
        
        manager.register(AuthExtension(priority=ExtensionPriority.HIGHEST))
        manager.register(ApprovalExtension(priority=ExtensionPriority.HIGH))
        manager.register(LoggingExtension(priority=ExtensionPriority.NORMAL))
        manager.register(MetricsExtension(priority=ExtensionPriority.LOW))
        
        return manager