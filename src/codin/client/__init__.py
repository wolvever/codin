"""HTTP client infrastructure for codin agents.

This module provides HTTP client functionality with tracing, metrics,
and observability features for external API integrations.
"""

from .base import Client, ClientConfig, RequestTracer
from .tracers import LoggingTracer, MetricsTracer, RequestHistoryTracer


__all__ = [
    'Client',
    'ClientConfig',
    'LoggingTracer',
    'MetricsTracer',
    'RequestHistoryTracer',
    'RequestTracer',
]
