from __future__ import annotations

import logging
import time
import typing as _t

from pydantic import BaseModel, Field

from .base import RequestTracer

__all__ = [
    "LoggingTracer",
    "MetricsTracer",
    "RequestHistoryTracer",
]

logger = logging.getLogger("codin.client.tracers")


class LoggingTracer(RequestTracer):
    """A simple tracer that logs all requests and responses."""
    
    def __init__(self, log_level: int = logging.DEBUG):
        self.log_level = log_level
    
    async def on_request_start(self, method: str, url: str, headers: dict, data: _t.Any | None = None) -> None:
        logger.log(self.log_level, f"Starting {method} request to {url}")
    
    async def on_request_end(self, method: str, url: str, status_code: int, elapsed: float, 
                         response_headers: dict, response_data: _t.Any | None = None) -> None:
        logger.log(self.log_level, f"Completed {method} request to {url} with status {status_code} in {elapsed:.3f}s")
    
    async def on_request_error(self, method: str, url: str, error: Exception, elapsed: float) -> None:
        logger.log(logging.ERROR, f"Failed {method} request to {url} after {elapsed:.3f}s: {str(error)}")


class RequestMetrics(BaseModel):
    """Metrics for HTTP requests."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0
    
    # Status code counters
    status_codes: dict[int, int] = Field(default_factory=dict)
    
    # Timing stats
    min_time: float = float('inf')
    max_time: float = 0.0
    
    @property
    def average_time(self) -> float:
        """Calculate the average request time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_time / self.total_requests
    
    def record_request(self, status_code: int | None, elapsed: float) -> None:
        """Record metrics for a request."""
        self.total_requests += 1
        self.total_time += elapsed
        
        if status_code is not None:
            self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1
            
            if 200 <= status_code < 400:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
        else:
            # No status code means an error occurred
            self.failed_requests += 1
        
        # Update min/max times
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)


class MetricsTracer(RequestTracer):
    """A tracer that collects metrics about HTTP requests."""
    
    def __init__(self):
        self.metrics = RequestMetrics()
        self.current_requests: dict[tuple[str, str], float] = {}
    
    async def on_request_start(self, method: str, url: str, headers: dict, data: _t.Any | None = None) -> None:
        self.current_requests[(method, url)] = time.time()
    
    async def on_request_end(self, method: str, url: str, status_code: int, elapsed: float, 
                         response_headers: dict, response_data: _t.Any | None = None) -> None:
        self.metrics.record_request(status_code, elapsed)
        self.current_requests.pop((method, url), None)
    
    async def on_request_error(self, method: str, url: str, error: Exception, elapsed: float) -> None:
        self.metrics.record_request(None, elapsed)
        self.current_requests.pop((method, url), None)
    
    def get_metrics(self) -> RequestMetrics:
        """Get the collected metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = RequestMetrics()


class RequestRecord(BaseModel):
    """Record of a single HTTP request."""
    
    method: str
    url: str
    start_time: float
    end_time: float | None = None
    status_code: int | None = None
    error: str | None = None
    
    @property
    def elapsed(self) -> float:
        """Calculate the elapsed time for the request."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def successful(self) -> bool:
        """Check if the request was successful."""
        return self.error is None and self.status_code is not None and 200 <= self.status_code < 400


class RequestHistoryTracer(RequestTracer):
    """A tracer that keeps a history of recent requests."""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history: list[RequestRecord] = []
        self.current_requests: dict[tuple[str, str], RequestRecord] = {}
    
    async def on_request_start(self, method: str, url: str, headers: dict, data: _t.Any | None = None) -> None:
        record = RequestRecord(
            method=method,
            url=url,
            start_time=time.time()
        )
        self.current_requests[(method, url)] = record
    
    async def on_request_end(self, method: str, url: str, status_code: int, elapsed: float, 
                         response_headers: dict, response_data: _t.Any | None = None) -> None:
        key = (method, url)
        if key in self.current_requests:
            record = self.current_requests.pop(key)
            record.end_time = time.time()
            record.status_code = status_code
            self._add_to_history(record)
    
    async def on_request_error(self, method: str, url: str, error: Exception, elapsed: float) -> None:
        key = (method, url)
        if key in self.current_requests:
            record = self.current_requests.pop(key)
            record.end_time = time.time()
            record.error = str(error)
            self._add_to_history(record)
    
    def _add_to_history(self, record: RequestRecord) -> None:
        """Add a record to the history, maintaining the max history size."""
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(self) -> list[RequestRecord]:
        """Get the request history."""
        return self.history
    
    def get_recent_failures(self) -> list[RequestRecord]:
        """Get recent failed requests."""
        return [r for r in self.history if not r.successful]
    
    def clear_history(self) -> None:
        """Clear the request history."""
        self.history = [] 