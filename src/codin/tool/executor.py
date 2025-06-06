from __future__ import annotations

import typing as _t
import logging
import time
import traceback
import asyncio
from contextlib import asynccontextmanager

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import Counter, UpDownCounter, Histogram

# Prometheus imports
import prometheus_client as prom

from a2a.types import TextPart, Message
from .base import Tool, ToolContext
from .registry import ToolRegistry
from ..config import ApprovalMode

__all__ = [
    "ToolExecutor",
    "ToolExecutionHook",
    "ApprovalHook",
]

# Create tracer and metrics
tracer = trace.get_tracer("codin.tool.executor")
meter = metrics.get_meter("codin.tool.executor")

# Define metrics
tool_execution_counter = meter.create_counter(
    name="tool_executions",
    description="Number of tool executions",
    unit="1",
)

tool_execution_duration = meter.create_histogram(
    name="tool_execution_duration",
    description="Duration of tool executions",
    unit="s",
)

tool_execution_errors = meter.create_counter(
    name="tool_execution_errors",
    description="Number of tool execution errors",
    unit="1",
)

# Define Prometheus metrics - use try/except to avoid duplicate registration
try:
    prom_tool_executions = prom.Counter(
        "codin_tool_executions_total",
        "Number of tool executions",
        ["tool", "status"],
    )
except ValueError:
    # Metric already exists, reuse it
    prom_tool_executions = prom.REGISTRY._names_to_collectors["codin_tool_executions_total"]

try:
    prom_tool_execution_duration = prom.Histogram(
        "codin_tool_execution_duration_seconds",
        "Duration of tool executions",
        ["tool"],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    )
except ValueError:
    # Metric already exists, reuse it
    prom_tool_execution_duration = prom.REGISTRY._names_to_collectors["codin_tool_execution_duration_seconds"]

try:
    prom_tool_errors = prom.Counter(
        "codin_tool_errors_total",
        "Number of tool execution errors",
        ["tool", "error_type"],
    )
except ValueError:
    # Metric already exists, reuse it
    prom_tool_errors = prom.REGISTRY._names_to_collectors["codin_tool_errors_total"]

try:
    prom_tool_retries = prom.Counter(
        "codin_tool_retries_total",
        "Number of tool execution retries",
        ["tool"],
    )
except ValueError:
    # Metric already exists, reuse it
    prom_tool_retries = prom.REGISTRY._names_to_collectors["codin_tool_retries_total"]

class ToolExecutionHook:
    """Base class for tool execution hooks."""
    
    async def before_execution(self, tool: Tool, args: _t.Dict[str, _t.Any], context: ToolContext) -> None:
        """Called before tool execution."""
        pass
    
    async def after_execution(self, tool: Tool, args: _t.Dict[str, _t.Any], result: _t.Any, context: "ToolContext") -> None:
        """Called after successful tool execution."""
        pass
    
    async def on_error(self, tool: Tool, args: _t.Dict[str, _t.Any], error: Exception, context: "ToolContext") -> None:
        """Called when tool execution raises an exception."""
        pass

class ApprovalHook(ToolExecutionHook):
    """Hook for tool approval."""
    
    def __init__(self, approval_mode: ApprovalMode = ApprovalMode.ALWAYS):
        self.approval_mode = approval_mode
    
    async def before_execution(self, tool: Tool, args: _t.Dict[str, _t.Any], context: ToolContext) -> None:
        if self.approval_mode == ApprovalMode.ALWAYS:
            # Implementation for manual approval, e.g. waiting for human confirmation
            await context.get_user_approval(f"Approve tool execution: {tool.name}")
        elif self.approval_mode == ApprovalMode.NEVER:
            # Auto-approve, no action needed
            pass
        elif self.approval_mode == ApprovalMode.UNSAFE_ONLY:
            # Check if tool is potentially unsafe and ask for approval if needed
            if self._is_unsafe_tool(tool):
                await context.get_user_approval(f"Approve potentially unsafe tool execution: {tool.name}")
    
    def _is_unsafe_tool(self, tool: Tool) -> bool:
        """Determine if a tool is potentially unsafe and requires approval."""
        # This is a simple heuristic - in practice, you might want more sophisticated logic
        unsafe_keywords = ['delete', 'remove', 'rm', 'kill', 'terminate', 'destroy', 'format']
        tool_name_lower = tool.name.lower()
        return any(keyword in tool_name_lower for keyword in unsafe_keywords)

class ToolExecutor:
    """Executes tools with cross-cutting concerns."""
    
    def __init__(
        self,
        registry: ToolRegistry,
        max_concurrency: int = 5,
        default_timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.registry = registry
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.hooks: list[ToolExecutionHook] = []
        self.logger = logging.getLogger(__name__)
    
    def add_hook(self, hook: ToolExecutionHook) -> None:
        """Add a hook to the executor."""
        self.hooks.append(hook)
    
    @asynccontextmanager
    async def _concurrency_limit(self):
        """Context manager for limiting concurrency."""
        async with self.semaphore:
            yield
    
    async def execute(
        self,
        tool_name: str,
        args: _t.Dict[str, _t.Any],
        context: "ToolContext",
        timeout: _t.Optional[float] = None,
        retries: _t.Optional[int] = None,
    ) -> _t.Any:
        """Execute a tool by name."""
        with tracer.start_as_current_span(f"execute_{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.args", str(args))
            
            tool = self.registry.get_tool(tool_name)
            if not tool:
                self.logger.error(f"Tool not found: {tool_name}")
                span.set_status(Status(StatusCode.ERROR, f"Tool not found: {tool_name}"))
                prom_tool_errors.labels(tool=tool_name, error_type="not_found").inc()
                raise ValueError(f"Tool not found: {tool_name}")
            
            span.set_attribute("tool.description", tool.description)
            
            timeout = timeout or self.default_timeout
            retries = retries if retries is not None else self.max_retries
            
            try:
                result = await self._execute_with_retry(tool, args, context, timeout, retries)
                prom_tool_executions.labels(tool=tool.name, status="success").inc()
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                prom_tool_executions.labels(tool=tool.name, status="error").inc()
                raise
    
    async def _execute_with_retry(
        self,
        tool: Tool,
        args: _t.Dict[str, _t.Any],
        context: "ToolContext",
        timeout: float,
        retries: int,
    ) -> _t.Any:
        """Execute with retry logic."""
        attempt = 0
        last_error = None
        
        while True:
            attempt += 1
            try:
                with tracer.start_as_current_span(f"execute_attempt_{attempt}") as span:
                    span.set_attribute("tool.attempt", attempt)
                    span.set_attribute("tool.max_attempts", retries + 1)
                    
                    return await self._execute_once(tool, args, context, timeout)
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt > retries:
                    self.logger.error(f"Tool {tool.name} timed out after {retries} retries")
                    tool_execution_errors.add(1, {"tool": tool.name, "error": "timeout"})
                    prom_tool_errors.labels(tool=tool.name, error_type="timeout").inc()
                    raise
                self.logger.warning(f"Tool {tool.name} timed out, retrying ({attempt}/{retries})")
                prom_tool_retries.labels(tool=tool.name).inc()
            except Exception as e:
                last_error = e
                if attempt > retries:
                    self.logger.error(f"Tool {tool.name} failed after {retries} retries: {str(e)}")
                    tool_execution_errors.add(1, {"tool": tool.name, "error": str(e)[:100]})
                    prom_tool_errors.labels(tool=tool.name, error_type="exception").inc()
                    raise
                self.logger.warning(f"Tool {tool.name} failed, retrying ({attempt}/{retries}): {str(e)}")
                prom_tool_retries.labels(tool=tool.name).inc()
    
    async def _execute_once(
        self,
        tool: Tool,
        args: _t.Dict[str, _t.Any],
        context: "ToolContext",
        timeout: float,
    ) -> _t.Any:
        """Execute a tool once with all hooks."""
        # Run before execution hooks
        for hook in self.hooks:
            await hook.before_execution(tool, args, context)
        
        # Validate input
        try:
            validated_args = tool.validate_input(args)
        except Exception as e:
            self.logger.error(f"Input validation failed for tool {tool.name}: {str(e)}")
            for hook in self.hooks:
                await hook.on_error(tool, args, e, context)
            tool_execution_errors.add(1, {"tool": tool.name, "error": "validation_error"})
            prom_tool_errors.labels(tool=tool.name, error_type="validation_error").inc()
            raise
        
        # Execute with timeout and concurrency limit
        start_time = time.time()
        tool_execution_counter.add(1, {"tool": tool.name})
        
        try:
            async with self._concurrency_limit():
                run_result = await asyncio.wait_for(
                    tool.run(validated_args, context),
                    timeout=timeout
                )
                
                # Handle async generator results
                if hasattr(run_result, "__aiter__"):
                    collected_result = await self._process_async_generator(tool, run_result)
                    result = collected_result
                else:
                    result = run_result
                
                # Try to convert to protocol types if possible
                result = self._try_convert_to_protocol_types(result)
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Tool {tool.name} failed after {elapsed:.2f}s: {str(e)}")
            tool_execution_duration.record(elapsed, {"tool": tool.name, "status": "error"})
            prom_tool_execution_duration.labels(tool=tool.name).observe(elapsed)
            prom_tool_errors.labels(tool=tool.name, error_type="execution_error").inc()
            
            for hook in self.hooks:
                await hook.on_error(tool, args, e, context)
            raise
        
        # Run after execution hooks
        elapsed = time.time() - start_time
        self.logger.info(f"Tool {tool.name} completed in {elapsed:.2f}s")
        tool_execution_duration.record(elapsed, {"tool": tool.name, "status": "success"})
        prom_tool_execution_duration.labels(tool=tool.name).observe(elapsed)
        
        for hook in self.hooks:
            await hook.after_execution(tool, args, result, context)
        
        return result
    
    async def _process_async_generator(
        self, 
        tool: Tool, 
        generator: _t.AsyncGenerator
    ) -> _t.Any:
        """Process an async generator result from a tool."""
        with tracer.start_as_current_span("process_async_generator") as span:
            span.set_attribute("tool.is_generative", True)
            
            self.logger.debug(f"Processing async generator from tool {tool.name}")
            
            # For simple implementation, collect all items
            collected_items = []
            try:
                async for item in generator:
                    collected_items.append(item)
            except Exception as e:
                self.logger.error(f"Error processing generator from tool {tool.name}: {str(e)}")
                span.record_exception(e)
                prom_tool_errors.labels(tool=tool.name, error_type="generator_error").inc()
                raise
                
            # If we have a single item, return it directly
            if len(collected_items) == 1:
                return collected_items[0]
                
            return collected_items
    
    def _try_convert_to_protocol_types(self, result: _t.Any) -> _t.Any:
        """Try to convert tool results to protocol types if possible."""
        if result is None:
            return None
                        
        # If already a protocol type, return as is
        from a2a.types import Message, TextPart, DataPart, FilePart, Part
        if isinstance(result, (Message, TextPart, DataPart, FilePart)):
            return result
            
        # Check if result is from MCP
        if hasattr(result, "get") and isinstance(result, dict):
            try:
                from ..tool.mcp.conversion_utils import convert_mcp_to_protocol_types
                return convert_mcp_to_protocol_types(result)
            except (ImportError, Exception) as e:
                self.logger.debug(f"Could not convert MCP result: {e}")
                
        # If string, convert to TextPart
        if isinstance(result, str):
            return TextPart(text=result)
            
        return result