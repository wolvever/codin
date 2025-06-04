"""Tests for the model.openai_llm module."""

import asyncio
import json
import os
import pytest
import typing as _t
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from codin.model.openai_llm import OpenAILLM
from codin.client import Client


@pytest.fixture
def mock_response():
    """Create a mock response from OpenAI API."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "This is a test response.",
                    "role": "assistant"
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }
    return response


@pytest.fixture
def mock_stream_response():
    """Create a mock streaming response from OpenAI API."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    
    # Stream response lines
    stream_data = [
        'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":"This"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":" is"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":" a"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":" test"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":" response."},"index":0}]}',
        'data: [DONE]'
    ]
    
    # Mock aiter_lines to return the stream data
    async def mock_aiter_lines():
        for line in stream_data:
            yield line
    
    response.aiter_lines = mock_aiter_lines
    
    return response


@pytest.fixture
def mock_function_call_response():
    """Create a mock function call response from OpenAI API."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"New York","unit":"celsius"}'
                            }
                        }
                    ]
                },
                "index": 0,
                "finish_reason": "tool_calls"
            }
        ]
    }
    return response


class TestOpenAILLM:
    """Test cases for the OpenAILLM class."""
    
    @pytest.fixture(autouse=True)
    def setup_environment(self, monkeypatch):
        """Set up environment variables for tests."""
        # Clear any existing environment variables first
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        
        # Set test environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.test.com/v1")
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock client for API requests."""
        with patch("codin.model.openai_llm.Client") as mock_client_cls:
            client_instance = MagicMock()
            mock_client_cls.return_value = client_instance
            client_instance.prepare = AsyncMock()
            client_instance.post = AsyncMock()
            client_instance.close = AsyncMock()
            
            yield client_instance
    
    def test_supported_models(self):
        """Test the supported models patterns."""
        patterns = OpenAILLM.supported_models()
        
        assert "gpt-3.5-turbo.*" in patterns
        assert "gpt-4.*" in patterns
        assert "gpt-4o.*" in patterns
        assert "claude-.*" in patterns  # For compatibility layer
    
    @pytest.mark.asyncio
    async def test_prepare(self, mock_client):
        """Test preparing the model."""
        with patch("codin.model.openai_llm.Client") as mock_client_cls:
            mock_client_cls.return_value = mock_client
            
            llm = OpenAILLM("gpt-4")
            await llm.prepare()
            
            # Check that client was created with correct config
            mock_client_cls.assert_called_once()
            
            # Get the config passed to Client
            config = mock_client_cls.call_args[0][0]
            assert config.base_url == "https://api.test.com/v1"
            assert config.default_headers["Authorization"] == "Bearer test-api-key"
            assert config.default_headers["Content-Type"] == "application/json"
            
            # Check that client was prepared
            mock_client.prepare.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_prepare_missing_api_key(self, monkeypatch):
        """Test preparing the model with missing API key."""
        # Clear both possible API key environment variables
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        llm = OpenAILLM("gpt-4")
        with pytest.raises(ValueError, match="LLM_API_KEY or OPENAI_API_KEY environment variable is required"):
            await llm.prepare()
    
    @pytest.mark.asyncio
    async def test_generate_string_prompt(self, mock_client, mock_response):
        """Test generating text with a string prompt."""
        with patch("codin.model.openai_llm.Client") as mock_client_cls:
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response
            
            llm = OpenAILLM("gpt-4")
            await llm.prepare()
            
            # Generate text
            response = await llm.generate("What is the capital of France?")
            
            # Check the response
            assert response == "This is a test response."
            
            # Verify the API request
            mock_client.post.assert_called_once()
            args, kwargs = mock_client.post.call_args
            assert args[0] == "/chat/completions"
            assert kwargs["json"]["model"] == "gpt-4"
            assert kwargs["json"]["messages"] == [
                {"role": "user", "content": "What is the capital of France?"}
            ]
            assert not kwargs["json"]["stream"]
    
    @pytest.mark.asyncio
    async def test_generate_message_prompt(self, mock_client, mock_response):
        """Test generating text with a message list prompt."""
        with patch("codin.model.openai_llm.Client") as mock_client_cls:
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response
            
            llm = OpenAILLM("gpt-4")
            await llm.prepare()
            
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
            
            # Generate text
            response = await llm.generate(messages)
            
            # Check the response
            assert response == "This is a test response."
            
            # Verify the API request
            mock_client.post.assert_called_once()
            args, kwargs = mock_client.post.call_args
            assert args[0] == "/chat/completions"
            assert kwargs["json"]["model"] == "gpt-4"
            assert kwargs["json"]["messages"] == messages
            assert not kwargs["json"]["stream"]
    
    @pytest.mark.asyncio
    async def test_generate_with_parameters(self, mock_client, mock_response):
        """Test generating text with additional parameters."""
        with patch("codin.model.openai_llm.Client") as mock_client_cls:
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response
            
            llm = OpenAILLM("gpt-4")
            await llm.prepare()
            
            # Generate text with parameters
            response = await llm.generate(
                "What is the capital of France?",
                temperature=0.7,
                max_tokens=100,
                stop_sequences=[".", "!"]
            )
            
            # Check the response
            assert response == "This is a test response."
            
            # Verify the API request
            mock_client.post.assert_called_once()
            args, kwargs = mock_client.post.call_args
            assert args[0] == "/chat/completions"
            assert kwargs["json"]["temperature"] == 0.7
            assert kwargs["json"]["max_tokens"] == 100
            assert kwargs["json"]["stop"] == [".", "!"]
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, mock_client, mock_stream_response):
        """Test streaming response generation."""
        with patch("codin.model.openai_llm.Client") as mock_client_cls:
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_stream_response
            
            llm = OpenAILLM("gpt-4")
            await llm.prepare()
            
            # Generate streaming response
            stream = await llm.generate("What is the capital of France?", stream=True)
            
            # Collect streamed chunks
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            
            # Check the collected chunks
            assert chunks == ["This", " is", " a", " test", " response."]
            
            # Verify the API request
            mock_client.post.assert_called_once()
            args, kwargs = mock_client.post.call_args
            assert args[0] == "/chat/completions"
            assert kwargs["json"]["stream"] is True
    
    @pytest.mark.asyncio
    async def test_generate_with_tools(self, mock_client, mock_function_call_response):
        """Test generating function calls with tools."""
        with patch("codin.model.openai_llm.Client") as mock_client_cls:
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_function_call_response
            
            llm = OpenAILLM("gpt-4")
            await llm.prepare()
            
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state"
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"]
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
            
            # Generate response with tools
            response = await llm.generate_with_tools(
                "What's the weather in New York?",
                tools=tools
            )
            
            # Check the response
            assert "tool_calls" in response
            assert response["tool_calls"][0]["function"]["name"] == "get_weather"
            assert json.loads(response["tool_calls"][0]["function"]["arguments"]) == {
                "location": "New York",
                "unit": "celsius"
            }
            
            # Verify the API request
            mock_client.post.assert_called_once()
            args, kwargs = mock_client.post.call_args
            assert args[0] == "/chat/completions"
            assert kwargs["json"]["tools"] == tools
    
    @pytest.mark.asyncio
    async def test_close(self, mock_client):
        """Test closing the client."""
        with patch("codin.model.openai_llm.Client") as mock_client_cls:
            mock_client_cls.return_value = mock_client
            
            llm = OpenAILLM("gpt-4")
            await llm.prepare()
            await llm.close()
            
            mock_client.close.assert_called_once() 