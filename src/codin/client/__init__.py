"""Client module for making HTTP requests.

This module provides a robust HTTP client with retry, logging, and tracing support.
"""

from .base import Client, ClientConfig, RequestTracer
from .tracers import LoggingTracer, MetricsTracer, RequestHistoryTracer

__all__ = [
    "Client", 
    "ClientConfig", 
    "RequestTracer",
    "LoggingTracer",
    "MetricsTracer",
    "RequestHistoryTracer",
] 