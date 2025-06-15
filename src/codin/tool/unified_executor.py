"""Unified executor that handles all tool types with extension support."""

from __future__ import annotations

import asyncio
import logging
import time
import typing as _t
from contextlib import asynccontextmanager

from .base import ToolContext, ToolSpec
from .registry import ToolRegistry
from .executors.base import BaseExecutor, ExecutorRegistry, ExecutionResult, ExecutionStatus
from .extensions.base import ExtensionManager, ExtensionContext

__all__ = ['UnifiedToolExecutor']


class UnifiedToolExecutor:
    """Unified executor that handles all tool types with extension support."""
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        executor_registry: ExecutorRegistry | None = None,
        extension_manager: ExtensionManager | None = None,
        default_timeout: float = 60.0,
        max_concurrency: int = 10,
    ):
        self.tool_registry = tool_registry
        self.executor_registry = executor_registry or self._create_default_executor_registry()
        self.extension_manager = extension_manager or self._create_default_extension_manager()
        self.default_timeout = default_timeout
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.logger = logging.getLogger(__name__)
        
        # Cache for executor setup
        self._setup_cache: dict[str, BaseExecutor] = {}
    
    def _create_default_executor_registry(self) -> ExecutorRegistry:
        """Create default executor registry."""
        from .executors.base import ExecutorFactory
        return ExecutorFactory.create_registry()
    
    def _create_default_extension_manager(self) -> ExtensionManager:
        """Create default extension manager."""
        from .extensions.base import ExtensionChain
        return ExtensionChain.create_default_chain()
    
    async def execute(
        self,
        tool_name: str,
        args: dict[str, _t.Any],
        context: ToolContext | None = None,
        timeout: float | None = None,
    ) -> _t.Any:
        """Execute a tool by name with full extension support."""
        # Get tool and specification
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        spec = tool.get_spec()
        
        # Create context if not provided
        if context is None:
            context = ToolContext(tool_name=tool_name, arguments=args)
        
        # Create extension context
        ext_context = ExtensionContext(
            spec=spec,
            args=args,
            tool_context=context
        )
        
        # Use spec timeout if not provided
        execution_timeout = timeout or spec.timeout or self.default_timeout
        
        try:
            # Execute with timeout and concurrency control
            async with self._concurrency_limit():
                result = await asyncio.wait_for(
                    self._execute_with_extensions(spec, args, context, ext_context),
                    timeout=execution_timeout
                )
                return result.result if isinstance(result, ExecutionResult) else result
                
        except asyncio.TimeoutError:
            await self.extension_manager.on_timeout(ext_context)
            raise TimeoutError(f"Tool {tool_name} execution timed out after {execution_timeout}s")
        
        except asyncio.CancelledError:
            await self.extension_manager.on_cancelled(ext_context)
            raise
        
        except Exception as e:
            # Let extensions handle the error
            processed_error = await self.extension_manager.on_error(ext_context, e)
            if processed_error is not None:
                raise processed_error
            # Error was suppressed by extension
            return None
    
    async def _execute_with_extensions(
        self,
        spec: ToolSpec,
        args: dict[str, _t.Any],
        context: ToolContext,
        ext_context: ExtensionContext,
    ) -> ExecutionResult:
        """Execute tool with full extension pipeline."""
        start_time = time.time()
        
        try:
            # Run before_execute extensions
            await self.extension_manager.before_execute(ext_context)
            
            # Get and setup executor
            executor = await self._get_executor(spec)
            
            # Execute the tool
            result = await executor.execute(spec, args, context)
            
            # Create execution result
            execution_result = ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                result=result,
                duration=time.time() - start_time,
                metadata={'executor': type(executor).__name__}
            )
            
            # Run after_execute extensions
            execution_result = await self.extension_manager.after_execute(ext_context, execution_result)
            
            return execution_result
            
        except Exception as e:
            # Create error result
            execution_result = ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=e,
                duration=time.time() - start_time
            )
            
            # Extensions will handle error in the outer try/except
            raise e
    
    async def _get_executor(self, spec: ToolSpec) -> BaseExecutor:
        """Get and setup executor for tool spec."""
        # Check cache first
        if spec.name in self._setup_cache:
            return self._setup_cache[spec.name]
        
        # Find appropriate executor
        executor = await self.executor_registry.find_executor(spec)
        if not executor:
            raise RuntimeError(f"No executor found for tool type: {spec.tool_type}")
        
        # Ensure executor is up
        if not executor.is_up:
            await executor.up()
        
        # Setup executor for this specific tool
        await executor.setup_tool(spec)
        self._setup_cache[spec.name] = executor
        
        return executor
    
    @asynccontextmanager
    async def _concurrency_limit(self):
        """Context manager for concurrency control."""
        async with self.semaphore:
            yield
    
    async def validate_args(self, tool_name: str, args: dict[str, _t.Any]) -> dict[str, _t.Any]:
        """Validate arguments for a tool."""
        spec = self.tool_registry.get_spec(tool_name)
        if not spec:
            raise ValueError(f"Tool specification not found: {tool_name}")
        
        return spec.validate_args(args)
    
    async def get_tool_info(self, tool_name: str) -> dict[str, _t.Any]:
        """Get information about a tool."""
        spec = self.tool_registry.get_spec(tool_name)
        if not spec:
            raise ValueError(f"Tool specification not found: {tool_name}")
        
        return {
            'name': spec.name,
            'description': spec.description,
            'tool_type': spec.tool_type.value,
            'input_schema': spec.input_schema,
            'output_schema': spec.output_schema,
            'execution_mode': spec.execution_mode.value,
            'timeout': spec.timeout,
            'retries': spec.retries,
            'metadata': spec.metadata.model_dump(),
        }
    
    async def list_tools(self, tool_type: str | None = None) -> list[dict[str, _t.Any]]:
        """List available tools."""
        specs = self.tool_registry.list_specs(tool_type)
        
        return [
            {
                'name': spec.name,
                'description': spec.description,
                'tool_type': spec.tool_type.value,
                'execution_mode': spec.execution_mode.value,
                'metadata': spec.metadata.model_dump(),
            }
            for spec in specs
        ]
    
    async def setup_all(self) -> None:
        """Setup all tools (pre-warm executors)."""
        specs = self.tool_registry.list_specs()
        
        for spec in specs:
            try:
                await self._get_executor(spec)
                self.logger.info(f"Setup completed for tool: {spec.name}")
            except Exception as e:
                self.logger.error(f"Failed to setup tool {spec.name}: {e}")
    
    async def teardown_all(self) -> None:
        """Teardown all executors and clean up resources."""
        specs = self.tool_registry.list_specs()
        
        for spec in specs:
            executor = self._setup_cache.pop(spec.name, None)
            if executor:
                try:
                    await executor.teardown_tool(spec)
                    self.logger.info(f"Teardown completed for tool: {spec.name}")
                except Exception as e:
                    self.logger.error(f"Failed to teardown tool {spec.name}: {e}")
        
        # Finally, bring down all executors
        executors_shutdown = set()
        for executor in self._setup_cache.values():
            if executor not in executors_shutdown:
                try:
                    await executor.down()
                    executors_shutdown.add(executor)
                except Exception as e:
                    self.logger.error(f"Failed to shutdown executor {type(executor).__name__}: {e}")
    
    def add_extension(self, extension) -> None:
        """Add an extension to the executor."""
        self.extension_manager.register(extension)
    
    def remove_extension(self, name: str) -> None:
        """Remove an extension by name."""
        self.extension_manager.unregister(name)