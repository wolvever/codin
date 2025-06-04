"""Global pytest configuration and fixtures."""

import os
import asyncio
import logging
import pytest
import typing as _t
from unittest.mock import AsyncMock, MagicMock, patch

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def env_setup(monkeypatch):
    """Set up common environment variables for testing."""
    # Model API keys
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    
    # Prompt configuration
    monkeypatch.setenv("PROMPT_RUN_MODE", "local")
    monkeypatch.setenv("PROMPT_TEMPLATE_DIR", "./tests/fixtures/prompts")
    
    yield


@pytest.fixture
def async_mock():
    """Create an AsyncMock with specified return value."""
    def _async_mock(return_value=None):
        mock = AsyncMock()
        mock.return_value = return_value
        return mock
    return _async_mock


@pytest.fixture
def temp_asyncio_event_loop():
    """Create a temporary asyncio event loop for tests.
    
    This is useful when we need to run asyncio code in a synchronous test.
    """
    old_loop = asyncio.get_event_loop_policy().get_event_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    yield loop
    
    loop.close()
    asyncio.set_event_loop(old_loop)


@pytest.fixture
def json_response_mock():
    """Create a mock HTTP response with JSON data."""
    def _json_response_mock(data, status_code=200):
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = data
        response.raise_for_status = MagicMock()
        response.text = "Mock response text"
        response.headers = {"Content-Type": "application/json"}
        return response
    return _json_response_mock


@pytest.fixture
def run_async():
    """Helper to run async code in sync tests."""
    def _run_async(coroutine, loop=None):
        """Run an async coroutine and return its result."""
        if loop is None:
            loop = asyncio.get_event_loop()
            
        if loop.is_running():
            # Create a new loop if the current one is already running
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coroutine)
            finally:
                new_loop.close()
        else:
            return loop.run_until_complete(coroutine)
            
    return _run_async 