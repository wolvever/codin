"""Tests for the client.base module."""

import asyncio
import pytest
import typing as _t
from unittest.mock import AsyncMock, MagicMock, patch, call

import httpx
from codin.client.base import Client, ClientConfig, RequestTracer


class MockTracer(RequestTracer):
    """Mock tracer for testing."""
    
    def __init__(self):
        self.start_calls = []
        self.end_calls = []
        self.error_calls = []
    
    async def on_request_start(self, method, url, headers, data=None):
        self.start_calls.append((method, url, headers, data))
    
    async def on_request_end(self, method, url, status_code, elapsed, response_headers, response_data=None):
        self.end_calls.append((method, url, status_code, elapsed, response_headers, response_data))
    
    async def on_request_error(self, method, url, error, elapsed):
        self.error_calls.append((method, url, error, elapsed))


@pytest.fixture
def mock_httpx_response():
    """Create a mock HTTPX response."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.text = '{"result": "success"}'
    mock_response.json.return_value = {"result": "success"}
    mock_response.raise_for_status = MagicMock()
    
    return mock_response


@pytest.fixture
def mock_httpx_client():
    """Create a mock HTTPX client with patched request method."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        
        # Set up the request method
        mock_client.request = AsyncMock()
        
        yield mock_client


class TestClientConfig:
    """Test cases for the ClientConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ClientConfig()
        
        assert config.base_url == ""
        assert config.timeout == 30.0
        assert config.connect_timeout == 10.0
        assert config.auth_header is None
        assert config.auth_token is None
        assert config.max_retries == 3
        assert config.retry_min_wait == 1.0
        assert config.retry_max_wait == 10.0
        assert 429 in config.retry_on_status_codes
        assert 500 in config.retry_on_status_codes
        assert not config.tracers
        assert config.run_mode == "local"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        tracer = MockTracer()
        config = ClientConfig(
            base_url="https://api.example.com",
            timeout=60.0,
            connect_timeout=5.0,
            auth_header="X-API-Key",
            auth_token="secret-token",
            max_retries=5,
            retry_min_wait=0.5,
            retry_max_wait=5.0,
            retry_on_status_codes=[429, 503],
            tracers=[tracer],
            default_headers={"User-Agent": "Test Client"},
            run_mode="remote"
        )
        
        assert config.base_url == "https://api.example.com"
        assert config.timeout == 60.0
        assert config.connect_timeout == 5.0
        assert config.auth_header == "X-API-Key"
        assert config.auth_token == "secret-token"
        assert config.max_retries == 5
        assert config.retry_min_wait == 0.5
        assert config.retry_max_wait == 5.0
        assert config.retry_on_status_codes == [429, 503]
        assert len(config.tracers) == 1
        assert config.tracers[0] is tracer
        assert config.default_headers == {"User-Agent": "Test Client"}
        assert config.run_mode == "remote"


class TestClient:
    """Test cases for the Client class."""
    
    @pytest.mark.asyncio
    async def test_prepare(self, mock_httpx_client):
        """Test client preparation."""
        config = ClientConfig(
            base_url="https://api.example.com",
            auth_header="Authorization",
            auth_token="token123"
        )
        
        client = Client(config)
        await client.prepare()
        
        # Check that httpx client was created with proper config
        httpx.AsyncClient.assert_called_once()
        call_kwargs = httpx.AsyncClient.call_args.kwargs
        assert call_kwargs["base_url"] == "https://api.example.com"
        assert call_kwargs["headers"] == {"Authorization": "Bearer token123"}
        assert call_kwargs["follow_redirects"] is True
        
        # Check that timeout was set correctly
        timeout = call_kwargs["timeout"]
        assert timeout.connect == 10.0
        assert timeout.pool == 30.0
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_httpx_client):
        """Test client as context manager."""
        mock_httpx_client.aclose = AsyncMock()
        
        async with Client() as client:
            assert client._client is mock_httpx_client
        
        # Check that client was closed
        mock_httpx_client.aclose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_request_with_tracing(self, mock_httpx_client, mock_httpx_response):
        """Test request with tracing."""
        # Set up mock response
        mock_httpx_client.request.return_value = mock_httpx_response
        
        # Create tracer and client
        tracer = MockTracer()
        config = ClientConfig(
            base_url="https://api.example.com",
            tracers=[tracer]
        )
        
        client = Client(config)
        await client.prepare()
        
        # Make request
        response = await client.get("/users", params={"id": 123})
        
        # Check response
        assert response is mock_httpx_response
        
        # Verify request was made
        mock_httpx_client.request.assert_called_once_with("GET", "/users", params={"id": 123})
        
        # Verify tracer was called
        assert len(tracer.start_calls) == 1
        assert tracer.start_calls[0][0] == "GET"
        assert tracer.start_calls[0][1] == "https://api.example.com/users"
        
        assert len(tracer.end_calls) == 1
        assert tracer.end_calls[0][0] == "GET"
        assert tracer.end_calls[0][1] == "https://api.example.com/users"
        assert tracer.end_calls[0][2] == 200
    
    @pytest.mark.asyncio
    async def test_request_error_handling(self, mock_httpx_client):
        """Test request error handling and tracing."""
        # Set up mock to raise exception
        error = httpx.RequestError("Connection failed")
        mock_httpx_client.request.side_effect = error
        
        # Create tracer and client
        tracer = MockTracer()
        config = ClientConfig(
            base_url="https://api.example.com",
            tracers=[tracer]
        )
        
        client = Client(config)
        await client.prepare()
        
        # Make request and catch exception
        with pytest.raises(httpx.RequestError) as excinfo:
            await client.get("/users")
        
        assert str(excinfo.value) == "Connection failed"
        
        # Verify tracer was called for error (3 times due to retries)
        assert len(tracer.start_calls) == 3  # 3 attempts due to max_retries=3
        assert len(tracer.error_calls) == 3
        assert tracer.error_calls[0][0] == "GET"
        assert tracer.error_calls[0][1] == "https://api.example.com/users"
        assert tracer.error_calls[0][2] is error
    
    @pytest.mark.asyncio
    async def test_convenience_methods(self, mock_httpx_client, mock_httpx_response):
        """Test convenience methods (get, post, etc.)."""
        # Set up mock response
        mock_httpx_client.request.return_value = mock_httpx_response
        
        client = Client(ClientConfig(base_url="https://api.example.com"))
        await client.prepare()
        
        # Test different methods
        await client.get("/users", params={"id": 123})
        await client.post("/users", json={"name": "Test User"})
        await client.put("/users/123", json={"name": "Updated User"})
        await client.patch("/users/123", json={"status": "active"})
        await client.delete("/users/123")
        
        # Verify all requests were made with correct methods
        assert mock_httpx_client.request.call_count == 5
        
        calls = [
            call("GET", "/users", params={"id": 123}),
            call("POST", "/users", json={"name": "Test User"}),
            call("PUT", "/users/123", json={"name": "Updated User"}),
            call("PATCH", "/users/123", json={"status": "active"}),
            call("DELETE", "/users/123")
        ]
        
        mock_httpx_client.request.assert_has_calls(calls) 