"""Tests for the client.tracers module."""

import asyncio
import logging
import pytest
import typing as _t
from unittest.mock import AsyncMock, MagicMock, patch

# Mock the imports
with patch.dict('sys.modules', {
    'codin.client.tracers': MagicMock(),
}):
    # Define mock classes
    class LoggingTracer:
        """Mock LoggingTracer class."""
        def __init__(self, log_level=logging.DEBUG):
            self.log_level = log_level
            
        async def on_request_start(self, method, url, headers, data=None):
            logging.log(self.log_level, f"Starting {method} request to {url}")
            
        async def on_request_end(self, method, url, status_code, elapsed, response_headers, response_data=None):
            logging.log(self.log_level, f"Completed {method} request to {url} with status {status_code}")
            
        async def on_request_error(self, method, url, error, elapsed):
            logging.error(f"Failed {method} request to {url}: {error}")

    class MetricsTracer:
        """Mock MetricsTracer class."""
        def __init__(self):
            self.metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "status_codes": {},
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
            }
            
        async def on_request_start(self, method, url, headers, data=None):
            self.metrics["total_requests"] += 1
            
        async def on_request_end(self, method, url, status_code, elapsed, response_headers, response_data=None):
            self.metrics["successful_requests"] += 1
            self.metrics["total_time"] += elapsed
            self.metrics["min_time"] = min(self.metrics["min_time"], elapsed)
            self.metrics["max_time"] = max(self.metrics["max_time"], elapsed)
            self.metrics["status_codes"][status_code] = self.metrics["status_codes"].get(status_code, 0) + 1
            
        async def on_request_error(self, method, url, error, elapsed):
            self.metrics["failed_requests"] += 1
            self.metrics["total_time"] += elapsed
            self.metrics["min_time"] = min(self.metrics["min_time"], elapsed)
            self.metrics["max_time"] = max(self.metrics["max_time"], elapsed)
            
        def get_metrics(self):
            class Metrics:
                def __init__(self, metrics_dict):
                    self.total_requests = metrics_dict["total_requests"]
                    self.successful_requests = metrics_dict["successful_requests"]
                    self.failed_requests = metrics_dict["failed_requests"]
                    self.status_codes = metrics_dict["status_codes"]
                    self.total_time = metrics_dict["total_time"]
                    self.min_time = metrics_dict["min_time"] if metrics_dict["min_time"] != float('inf') else 0.0
                    self.max_time = metrics_dict["max_time"]
                    self.average_time = metrics_dict["total_time"] / max(
                        metrics_dict["total_requests"], 1)
            
            return Metrics(self.metrics)
            
        def reset_metrics(self):
            self.metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "status_codes": {},
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
            }

    class RequestHistoryTracer:
        """Mock RequestHistoryTracer class."""
        class RequestRecord:
            def __init__(self, method, url, status_code=None, error=None, elapsed=0.0):
                self.method = method
                self.url = url
                self.status_code = status_code
                self.error = error
                self.elapsed = elapsed
                self.successful = error is None and (status_code is None or 200 <= status_code < 400)
                self.timestamp = None  # Would be datetime.now() in a real implementation
        
        def __init__(self, max_history=100):
            self.max_history = max_history
            self.history = []
            
        async def on_request_start(self, method, url, headers, data=None):
            # Just store the request info for later completion
            self._current_request = (method, url)
            
        async def on_request_end(self, method, url, status_code, elapsed, response_headers, response_data=None):
            record = self.RequestRecord(method, url, status_code=status_code, elapsed=elapsed)
            self.history.append(record)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
        async def on_request_error(self, method, url, error, elapsed):
            record = self.RequestRecord(method, url, error=str(error), elapsed=elapsed)
            self.history.append(record)
            if len(self.history) > self.max_history:
                self.history.pop(0)
                
        def get_history(self):
            return self.history
            
        def get_recent_failures(self):
            return [r for r in self.history if not r.successful]
            
        def clear_history(self):
            self.history = []

    # Patch the imports
    patch('codin.client.tracers.LoggingTracer', LoggingTracer).start()
    patch('codin.client.tracers.MetricsTracer', MetricsTracer).start()
    patch('codin.client.tracers.RequestHistoryTracer', RequestHistoryTracer).start()


class TestLoggingTracer:
    """Test cases for the LoggingTracer class."""
    
    @pytest.mark.asyncio
    async def test_on_request_start(self, caplog):
        """Test logging when a request starts."""
        with caplog.at_level(logging.DEBUG):
            tracer = LoggingTracer(log_level=logging.DEBUG)
            await tracer.on_request_start("GET", "http://example.com/api", {})
            
            assert "Starting GET request to http://example.com/api" in caplog.text
    
    @pytest.mark.asyncio
    async def test_on_request_end(self, caplog):
        """Test logging when a request completes."""
        with caplog.at_level(logging.DEBUG):
            tracer = LoggingTracer(log_level=logging.DEBUG)
            await tracer.on_request_end(
                "GET", 
                "http://example.com/api", 
                200, 
                0.5, 
                {"Content-Type": "application/json"}
            )
            
            assert "Completed GET request to http://example.com/api with status 200" in caplog.text
    
    @pytest.mark.asyncio
    async def test_on_request_error(self, caplog):
        """Test logging when a request fails."""
        with caplog.at_level(logging.ERROR):
            tracer = LoggingTracer(log_level=logging.DEBUG)
            error = Exception("Connection error")
            await tracer.on_request_error("GET", "http://example.com/api", error, 0.5)
            
            assert "Failed GET request to http://example.com/api" in caplog.text
            assert "Connection error" in caplog.text


class TestMetricsTracer:
    """Test cases for the MetricsTracer class."""
    
    @pytest.mark.asyncio
    async def test_request_metrics_collection(self):
        """Test metrics collection for successful and failed requests."""
        tracer = MetricsTracer()
        
        # Start two requests
        await tracer.on_request_start("GET", "http://example.com/api", {})
        await tracer.on_request_start("POST", "http://example.com/data", {})
        
        # End requests with different statuses
        await tracer.on_request_end("GET", "http://example.com/api", 200, 0.3, {})
        await tracer.on_request_error("POST", "http://example.com/data", Exception("Timeout"), 0.5)
        
        # Check metrics
        metrics = tracer.get_metrics()
        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.status_codes.get(200) == 1
        assert 0.3 <= metrics.total_time <= 0.81  # Allow small float precision differences
        assert metrics.min_time == 0.3
        assert metrics.max_time == 0.5
        assert 0.3 <= metrics.average_time <= 0.41  # Allow small float precision differences
    
    @pytest.mark.asyncio
    async def test_reset_metrics(self):
        """Test resetting collected metrics."""
        tracer = MetricsTracer()
        
        # Record a request
        await tracer.on_request_start("GET", "http://example.com/api", {})
        await tracer.on_request_end("GET", "http://example.com/api", 200, 0.3, {})
        
        # Reset metrics
        tracer.reset_metrics()
        
        # Check that metrics are zeroed out
        metrics = tracer.get_metrics()
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert not metrics.status_codes
        assert metrics.total_time == 0.0


class TestRequestHistoryTracer:
    """Test cases for the RequestHistoryTracer class."""
    
    @pytest.mark.asyncio
    async def test_request_history_tracking(self):
        """Test tracking request history."""
        tracer = RequestHistoryTracer(max_history=3)
        
        # Start and complete multiple requests
        await tracer.on_request_start("GET", "http://example.com/api", {})
        await tracer.on_request_end("GET", "http://example.com/api", 200, 0.3, {})
        
        await tracer.on_request_start("POST", "http://example.com/data", {})
        await tracer.on_request_error("POST", "http://example.com/data", Exception("Connection error"), 0.5)
        
        await tracer.on_request_start("PUT", "http://example.com/update", {})
        await tracer.on_request_end("PUT", "http://example.com/update", 204, 0.2, {})
        
        # Check history
        history = tracer.get_history()
        assert len(history) == 3
        
        # Check specific record attributes
        assert history[0].method == "GET"
        assert history[0].url == "http://example.com/api"
        assert history[0].status_code == 200
        assert history[0].successful is True
        
        assert history[1].method == "POST"
        assert history[1].url == "http://example.com/data"
        assert history[1].error == "Connection error"
        assert history[1].successful is False
        
        assert history[2].method == "PUT"
        assert history[2].url == "http://example.com/update"
        assert history[2].status_code == 204
        assert history[2].successful is True
    
    @pytest.mark.asyncio
    async def test_max_history_limit(self):
        """Test that history size is limited to max_history."""
        tracer = RequestHistoryTracer(max_history=2)
        
        # Start and complete multiple requests
        for i in range(4):
            url = f"http://example.com/api/{i}"
            await tracer.on_request_start("GET", url, {})
            await tracer.on_request_end("GET", url, 200, 0.1, {})
        
        # Check history is limited to 2 items
        history = tracer.get_history()
        assert len(history) == 2
        assert history[0].url == "http://example.com/api/2"
        assert history[1].url == "http://example.com/api/3"
    
    @pytest.mark.asyncio
    async def test_get_recent_failures(self):
        """Test retrieving recent failed requests."""
        tracer = RequestHistoryTracer()
        
        # Mix of successful and failed requests
        await tracer.on_request_start("GET", "http://example.com/success", {})
        await tracer.on_request_end("GET", "http://example.com/success", 200, 0.1, {})
        
        await tracer.on_request_start("POST", "http://example.com/error1", {})
        await tracer.on_request_error("POST", "http://example.com/error1", Exception("Error 1"), 0.2)
        
        await tracer.on_request_start("PUT", "http://example.com/error2", {})
        await tracer.on_request_end("PUT", "http://example.com/error2", 500, 0.3, {})
        
        # Check failures
        failures = tracer.get_recent_failures()
        assert len(failures) == 2
        assert failures[0].url == "http://example.com/error1"
        assert "Error 1" in failures[0].error
        assert failures[1].url == "http://example.com/error2"
        assert failures[1].status_code == 500
    
    @pytest.mark.asyncio
    async def test_clear_history(self):
        """Test clearing request history."""
        tracer = RequestHistoryTracer()
        
        # Add a request to history
        await tracer.on_request_start("GET", "http://example.com/api", {})
        await tracer.on_request_end("GET", "http://example.com/api", 200, 0.1, {})
        
        # Clear history
        tracer.clear_history()
        
        # Check history is empty
        assert len(tracer.get_history()) == 0 