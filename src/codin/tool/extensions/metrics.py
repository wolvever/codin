"""Metrics extension for tool execution."""

from __future__ import annotations

import time
from typing import Any

try:
    import prometheus_client as prom
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    from opentelemetry import metrics
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

from ..executors.base import ExecutionResult, ExecutionStatus
from .base import Extension, ExtensionContext

__all__ = ['MetricsExtension']


class MetricsExtension(Extension):
    """Extension that collects metrics for tool execution."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._start_times: dict[str, float] = {}
        self._setup_prometheus_metrics()
        self._setup_otel_metrics()
    
    @property
    def name(self) -> str:
        return "metrics"
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics."""
        if not HAS_PROMETHEUS:
            return
        
        try:
            self.prom_executions = prom.Counter(
                'codin_tool_executions_total',
                'Total number of tool executions',
                ['tool_name', 'tool_type', 'status']
            )
            
            self.prom_duration = prom.Histogram(
                'codin_tool_execution_duration_seconds',
                'Tool execution duration',
                ['tool_name', 'tool_type'],
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf'))
            )
            
            self.prom_errors = prom.Counter(
                'codin_tool_errors_total',
                'Total number of tool execution errors',
                ['tool_name', 'tool_type', 'error_type']
            )
            
            self.prom_active = prom.Gauge(
                'codin_tool_active_executions',
                'Number of currently active tool executions',
                ['tool_name', 'tool_type']
            )
            
        except ValueError:
            # Metrics already registered, get existing ones
            self.prom_executions = prom.REGISTRY._names_to_collectors.get('codin_tool_executions_total')
            self.prom_duration = prom.REGISTRY._names_to_collectors.get('codin_tool_execution_duration_seconds')
            self.prom_errors = prom.REGISTRY._names_to_collectors.get('codin_tool_errors_total')
            self.prom_active = prom.REGISTRY._names_to_collectors.get('codin_tool_active_executions')
    
    def _setup_otel_metrics(self) -> None:
        """Setup OpenTelemetry metrics."""
        if not HAS_OTEL:
            return
        
        meter = metrics.get_meter("codin.tool.execution")
        
        self.otel_executions = meter.create_counter(
            name="tool_executions",
            description="Number of tool executions",
            unit="1"
        )
        
        self.otel_duration = meter.create_histogram(
            name="tool_execution_duration",
            description="Tool execution duration",
            unit="s"
        )
        
        self.otel_errors = meter.create_counter(
            name="tool_execution_errors",
            description="Number of tool execution errors",
            unit="1"
        )
    
    async def before_execute(self, ctx: ExtensionContext) -> None:
        """Record execution start metrics."""
        execution_id = str(id(ctx))
        self._start_times[execution_id] = time.time()
        
        # Increment active executions
        if HAS_PROMETHEUS and hasattr(self, 'prom_active'):
            self.prom_active.labels(
                tool_name=ctx.spec.name,
                tool_type=ctx.spec.tool_type.value
            ).inc()
    
    async def after_execute(self, ctx: ExtensionContext, result: ExecutionResult) -> ExecutionResult:
        """Record execution completion metrics."""
        execution_id = str(id(ctx))
        start_time = self._start_times.pop(execution_id, time.time())
        duration = time.time() - start_time
        
        tool_name = ctx.spec.name
        tool_type = ctx.spec.tool_type.value
        status = result.status.value
        
        # Prometheus metrics
        if HAS_PROMETHEUS:
            if hasattr(self, 'prom_executions'):
                self.prom_executions.labels(
                    tool_name=tool_name,
                    tool_type=tool_type,
                    status=status
                ).inc()
            
            if hasattr(self, 'prom_duration'):
                self.prom_duration.labels(
                    tool_name=tool_name,
                    tool_type=tool_type
                ).observe(duration)
            
            if hasattr(self, 'prom_active'):
                self.prom_active.labels(
                    tool_name=tool_name,
                    tool_type=tool_type
                ).dec()
        
        # OpenTelemetry metrics
        if HAS_OTEL:
            attributes = {
                "tool.name": tool_name,
                "tool.type": tool_type,
                "status": status
            }
            
            if hasattr(self, 'otel_executions'):
                self.otel_executions.add(1, attributes)
            
            if hasattr(self, 'otel_duration'):
                self.otel_duration.record(duration, {
                    "tool.name": tool_name,
                    "tool.type": tool_type
                })
        
        # Add metrics to result metadata
        if result.metadata is None:
            result.metadata = {}
        result.metadata.update({
            'metrics_duration': duration,
            'metrics_start_time': start_time,
            'metrics_end_time': time.time()
        })
        
        return result
    
    async def on_error(self, ctx: ExtensionContext, error: Exception) -> Exception:
        """Record error metrics."""
        execution_id = str(id(ctx))
        start_time = self._start_times.pop(execution_id, time.time())
        duration = time.time() - start_time
        
        tool_name = ctx.spec.name
        tool_type = ctx.spec.tool_type.value
        error_type = type(error).__name__
        
        # Prometheus metrics
        if HAS_PROMETHEUS:
            if hasattr(self, 'prom_executions'):
                self.prom_executions.labels(
                    tool_name=tool_name,
                    tool_type=tool_type,
                    status="error"
                ).inc()
            
            if hasattr(self, 'prom_errors'):
                self.prom_errors.labels(
                    tool_name=tool_name,
                    tool_type=tool_type,
                    error_type=error_type
                ).inc()
            
            if hasattr(self, 'prom_duration'):
                self.prom_duration.labels(
                    tool_name=tool_name,
                    tool_type=tool_type
                ).observe(duration)
            
            if hasattr(self, 'prom_active'):
                self.prom_active.labels(
                    tool_name=tool_name,
                    tool_type=tool_type
                ).dec()
        
        # OpenTelemetry metrics
        if HAS_OTEL:
            error_attributes = {
                "tool.name": tool_name,
                "tool.type": tool_type,
                "error.type": error_type
            }
            
            if hasattr(self, 'otel_executions'):
                self.otel_executions.add(1, {**error_attributes, "status": "error"})
            
            if hasattr(self, 'otel_errors'):
                self.otel_errors.add(1, error_attributes)
            
            if hasattr(self, 'otel_duration'):
                self.otel_duration.record(duration, {
                    "tool.name": tool_name,
                    "tool.type": tool_type
                })
        
        return error
    
    async def on_timeout(self, ctx: ExtensionContext) -> None:
        """Record timeout metrics."""
        execution_id = str(id(ctx))
        self._start_times.pop(execution_id, None)
        
        tool_name = ctx.spec.name
        tool_type = ctx.spec.tool_type.value
        
        # Record as error with timeout type
        if HAS_PROMETHEUS and hasattr(self, 'prom_errors'):
            self.prom_errors.labels(
                tool_name=tool_name,
                tool_type=tool_type,
                error_type="timeout"
            ).inc()
        
        if HAS_OTEL and hasattr(self, 'otel_errors'):
            self.otel_errors.add(1, {
                "tool.name": tool_name,
                "tool.type": tool_type,
                "error.type": "timeout"
            })
    
    async def on_cancelled(self, ctx: ExtensionContext) -> None:
        """Record cancellation metrics."""
        execution_id = str(id(ctx))
        self._start_times.pop(execution_id, None)
        
        tool_name = ctx.spec.name
        tool_type = ctx.spec.tool_type.value
        
        # Decrement active executions
        if HAS_PROMETHEUS and hasattr(self, 'prom_active'):
            self.prom_active.labels(
                tool_name=tool_name,
                tool_type=tool_type
            ).dec()