"""Base HTTP client implementation for codin agents.

This module provides a robust HTTP client with retry logic, tracing,
and observability features for making external API calls.
"""

import abc
import asyncio
import logging
import time
import typing as _t

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

__all__ = [
    'Client',
    'ClientConfig',
    'RequestTracer',
]

logger = logging.getLogger('codin.client')


class RequestTracer(abc.ABC):
    """Abstract base class for request tracing."""

    @abc.abstractmethod
    async def on_request_start(self, method: str, url: str, headers: dict, data: _t.Any | None = None) -> None:
        """Called when a request is about to be sent."""

    @abc.abstractmethod
    async def on_request_end(
        self,
        method: str,
        url: str,
        status_code: int,
        elapsed: float,
        response_headers: dict,
        response_data: _t.Any | None = None,
    ) -> None:
        """Called when a response has been received."""

    @abc.abstractmethod
    async def on_request_error(self, method: str, url: str, error: Exception, elapsed: float) -> None:
        """Called when a request raises an exception."""


class ClientConfig(BaseModel):
    """Configuration for the HTTP client."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Connection settings
    base_url: str = ''
    timeout: float = 30.0
    connect_timeout: float = 10.0

    # Authentication
    auth_header: str | None = None
    auth_token: str | None = None

    # Retry settings
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0
    retry_on_status_codes: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])

    # Logging and tracing
    log_level: int = logging.INFO
    tracers: list[RequestTracer] = Field(default_factory=list)

    # Request settings
    default_headers: dict[str, str] = Field(default_factory=dict)

    # Mode (local or remote)
    run_mode: str = 'local'


class Client:
    """HTTP client with retry, logging, and tracing support.

    This client wraps httpx and provides:
    - Automatic retries with exponential backoff
    - Request/response logging
    - Tracing for metrics and debugging
    - Configurable timeouts and other settings
    """

    def __init__(self, config: ClientConfig | None = None):
        """Initialize the client with the given configuration.

        Args:
            config: Configuration for the client
        """
        self.config = config or ClientConfig()
        self.run_mode = self.config.run_mode

        # Configure logging
        logger.setLevel(self.config.log_level)

        # Client initialization moved from prepare() to __init__()
        headers = dict(self.config.default_headers)
        if self.config.auth_header and self.config.auth_token:
            headers[self.config.auth_header] = f'Bearer {self.config.auth_token}'

        self._client: httpx.AsyncClient = httpx.AsyncClient( # Initialize directly
            base_url=self.config.base_url,
            timeout=httpx.Timeout(
                timeout=self.config.timeout,
                connect=self.config.connect_timeout,
            ),
            headers=headers,
            follow_redirects=True,
        )
        # No await self._client.prepare() here, as httpx.AsyncClient doesn't have it.
        # If it did, __init__ would need to be async.

    async def __aenter__(self) -> 'Client':
        """Set up the client for use in an async context manager."""
        # await self.prepare() # Removed, client is initialized in __init__
        # If self._client.prepare() was a thing for httpx.AsyncClient, it would be awaited in __init__
        # or here if __init__ was to remain sync and prepare() was still separate for async parts.
        # For now, assuming httpx.AsyncClient is ready after sync instantiation.
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up resources when exiting the async context manager."""
        await self.close()

    # prepare() method is now removed.

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _should_retry_response(self, response: httpx.Response) -> bool:
        """Determine if a request should be retried based on response status."""
        return response.status_code in self.config.retry_on_status_codes

    def _create_retry_decorator(self):
        """Create a retry decorator for HTTP requests."""
        return retry(
            retry=(
                retry_if_exception_type(httpx.RequestError)
                | retry_if_exception_type(httpx.TimeoutException)
                | retry_if_exception_type(asyncio.TimeoutError)
            ),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            reraise=True,
        )

    async def _trace_request(self, method, url, kwargs, fn):
        """Wrap a request with tracing and logging."""
        if not self.config.tracers and logger.level > logging.DEBUG:
            # Skip all the tracing overhead if not needed
            return await fn(**kwargs)

        # Extract headers and data for tracing
        headers = kwargs.get('headers', {})
        data = kwargs.get('json', kwargs.get('data'))

        # Generate a clean URL for logging - handle relative URLs
        if url.startswith('http'):
            # Already a full URL
            log_url = url
        # Relative URL - combine with base_url
        elif self.config.base_url:
            log_url = self.config.base_url.rstrip('/') + url
        else:
            log_url = url

        start_time = time.time()

        # Notify tracers of request start
        for tracer in self.config.tracers:
            await tracer.on_request_start(method, log_url, headers, data)

        logger.debug(f'{method} {log_url}')

        try:
            response = await fn(**kwargs)
            elapsed = time.time() - start_time

            # Check for retry status codes
            if self._should_retry_response(response):
                logger.warning(
                    f'{method} {log_url} returned {response.status_code}, may retry (elapsed: {elapsed:.3f}s)'
                )
                raise httpx.RequestError(f'Retryable status code: {response.status_code}')

            logger.debug(f'{method} {log_url} returned {response.status_code} in {elapsed:.3f}s')

            # Notify tracers of request completion
            for tracer in self.config.tracers:
                await tracer.on_request_end(
                    method, log_url, response.status_code, elapsed, dict(response.headers), response.text
                )

            return response

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f'{method} {log_url} failed after {elapsed:.3f}s: {e!s}')

            # Notify tracers of request error
            for tracer in self.config.tracers:
                await tracer.on_request_error(method, log_url, e, elapsed)

            raise

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Send an HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            **kwargs: Additional arguments to pass to httpx

        Returns:
            The HTTP response

        Raises:
            httpx.RequestError: If the request fails after retries
        """
        # await self.prepare() # Removed, client is initialized in __init__

        if self._client is None: # Should ideally not happen if __init__ succeeded
            raise RuntimeError("HTTP client is not initialized. This should not occur.")

        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        async def _make_request(**request_kwargs):
            if self._client is None:
                raise RuntimeError('HTTP client not initialized')

            # Use the trace wrapper
            return await self._trace_request(
                method, url, request_kwargs, lambda **kw: self._client.request(method, url, **kw)
            )

        return await _make_request(**kwargs)

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Send a GET request.

        Args:
            url: URL to request
            **kwargs: Additional arguments to pass to httpx

        Returns:
            The HTTP response
        """
        return await self.request('GET', url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Send a POST request.

        Args:
            url: URL to request
            **kwargs: Additional arguments to pass to httpx

        Returns:
            The HTTP response
        """
        return await self.request('POST', url, **kwargs)

    async def put(self, url: str, **kwargs) -> httpx.Response:
        """Send a PUT request.

        Args:
            url: URL to request
            **kwargs: Additional arguments to pass to httpx

        Returns:
            The HTTP response
        """
        return await self.request('PUT', url, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """Send a DELETE request.

        Args:
            url: URL to request
            **kwargs: Additional arguments to pass to httpx

        Returns:
            The HTTP response
        """
        return await self.request('DELETE', url, **kwargs)

    async def patch(self, url: str, **kwargs) -> httpx.Response:
        """Send a PATCH request.

        Args:
            url: URL to request
            **kwargs: Additional arguments to pass to httpx

        Returns:
            The HTTP response
        """
        return await self.request('PATCH', url, **kwargs)
