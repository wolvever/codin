"""Logging extension for tool execution."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..executors.base import ExecutionResult, ExecutionStatus
from .base import Extension, ExtensionContext

__all__ = ['LoggingExtension']


class LoggingExtension(Extension):
    """Extension that provides comprehensive logging for tool execution."""
    
    def __init__(self, logger: logging.Logger | None = None, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger or logging.getLogger('codin.tool.execution')
        self._start_times: dict[str, float] = {}
    
    @property
    def name(self) -> str:
        return "logging"
    
    async def before_execute(self, ctx: ExtensionContext) -> None:
        """Log tool execution start."""
        execution_id = id(ctx)
        self._start_times[str(execution_id)] = time.time()
        
        self.logger.info(
            "Tool execution started",
            extra={
                'tool_name': ctx.spec.name,
                'tool_type': ctx.spec.tool_type.value,
                'session_id': ctx.tool_context.session_id,
                'tool_call_id': ctx.tool_context.tool_call_id,
                'args': self._sanitize_args(ctx.args),
                'execution_id': execution_id,
            }
        )
    
    async def after_execute(self, ctx: ExtensionContext, result: ExecutionResult) -> ExecutionResult:
        """Log successful tool execution."""
        execution_id = id(ctx)
        start_time = self._start_times.pop(str(execution_id), time.time())
        duration = time.time() - start_time
        
        self.logger.info(
            "Tool execution completed",
            extra={
                'tool_name': ctx.spec.name,
                'tool_type': ctx.spec.tool_type.value,
                'status': result.status.value,
                'duration_ms': round(duration * 1000, 2),
                'execution_id': execution_id,
                'result_type': type(result.result).__name__ if result.result is not None else 'None',
                'result_size': self._get_result_size(result.result),
            }
        )
        
        # Add duration to result metadata
        if result.metadata is None:
            result.metadata = {}
        result.metadata['log_duration'] = duration
        
        return result
    
    async def on_error(self, ctx: ExtensionContext, error: Exception) -> Exception:
        """Log tool execution error."""
        execution_id = id(ctx)
        start_time = self._start_times.pop(str(execution_id), time.time())
        duration = time.time() - start_time
        
        self.logger.error(
            "Tool execution failed",
            extra={
                'tool_name': ctx.spec.name,
                'tool_type': ctx.spec.tool_type.value,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'duration_ms': round(duration * 1000, 2),
                'execution_id': execution_id,
            },
            exc_info=error
        )
        
        return error
    
    async def on_timeout(self, ctx: ExtensionContext) -> None:
        """Log tool execution timeout."""
        execution_id = id(ctx)
        start_time = self._start_times.pop(str(execution_id), time.time())
        duration = time.time() - start_time
        
        self.logger.warning(
            "Tool execution timed out",
            extra={
                'tool_name': ctx.spec.name,
                'tool_type': ctx.spec.tool_type.value,
                'timeout_duration_ms': round(duration * 1000, 2),
                'execution_id': execution_id,
                'specified_timeout': ctx.spec.timeout,
            }
        )
    
    async def on_cancelled(self, ctx: ExtensionContext) -> None:
        """Log tool execution cancellation."""
        execution_id = id(ctx)
        start_time = self._start_times.pop(str(execution_id), time.time())
        duration = time.time() - start_time
        
        self.logger.info(
            "Tool execution cancelled",
            extra={
                'tool_name': ctx.spec.name,
                'tool_type': ctx.spec.tool_type.value,
                'duration_before_cancel_ms': round(duration * 1000, 2),
                'execution_id': execution_id,
            }
        )
    
    def _sanitize_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Sanitize arguments for logging (remove sensitive data)."""
        sanitized = {}
        sensitive_keys = {'password', 'token', 'secret', 'key', 'api_key', 'auth'}
        
        for key, value in args.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = f"[TRUNCATED: {len(value)} chars]"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _get_result_size(self, result: Any) -> int | None:
        """Get approximate size of result for logging."""
        try:
            if isinstance(result, str):
                return len(result)
            elif isinstance(result, (list, dict)):
                return len(str(result))
            elif hasattr(result, '__len__'):
                return len(result)
            else:
                return None
        except Exception:
            return None