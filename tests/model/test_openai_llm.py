"""Tests for the model.openai_llm module."""

import asyncio
import json
import os
import pytest
import typing as _t
from unittest.mock import AsyncMock, MagicMock, patch

import httpx # Keep this if it's used by Client or other parts, otherwise can be removed
from codin.model.openai_llm import OpenAILLM
from codin.client import Client # Assuming Client is imported for spec or type hinting
from src.codin.model.config import ModelConfig
from src.codin.model.http_utils import ModelResponseParsingError, ContentExtractionError, StreamProcessingError # Added for new tests


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
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    }
    return response


@pytest.fixture
def mock_stream_response():
    """Create a mock streaming response from OpenAI API."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()

    stream_data = [
        'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":"This"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":" is"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":" a"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":" test"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":" response."},"index":0}]}',
        'data: [DONE]'
    ]

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
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.api.test.com/v1")

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for API requests."""
        client_instance = MagicMock(spec=Client)
        client_instance.prepare = AsyncMock()
        client_instance.post = AsyncMock()
        client_instance.close = AsyncMock()
        return client_instance

    def test_supported_models(self):
        """Test the supported models patterns."""
        patterns = OpenAILLM.supported_models()
        assert "gpt-3.5-turbo.*" in patterns
        assert "gpt-4.*" in patterns
        assert "gpt-4o.*" in patterns
        assert "claude-.*" in patterns

    @pytest.mark.asyncio
    async def test_init_with_env_vars(self, mock_client, setup_environment): # Renamed from test_prepare_...
        """Test initializing the model using environment variables."""
        with patch("codin.model.openai_llm.Client", return_value=mock_client) as mock_client_cls:
            llm = await OpenAILLM(model="gpt-4-env", config=None) # Await instantiation

            mock_client_cls.assert_called_once()
            client_constructor_config = mock_client_cls.call_args[0][0]
            assert client_constructor_config.base_url == "https://env.api.test.com/v1"
            assert client_constructor_config.default_headers["Authorization"] == "Bearer env-api-key"
            mock_client.prepare.assert_called_once() # Client.prepare is called in __init__

    @pytest.mark.asyncio
    async def test_init_with_model_config_only(self, mock_client, monkeypatch): # Renamed
        """Test initializing the model using only ModelConfig, no env vars."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        with patch("codin.model.openai_llm.Client", return_value=mock_client) as mock_client_cls:
            model_config = ModelConfig(
                api_key="config-key-123",
                base_url="https://config.url/v1",
                timeout=99.0,
                max_retries=5
            )
            llm = await OpenAILLM(model="gpt-4-config", config=model_config) # Await

            mock_client_cls.assert_called_once()
            client_constructor_config = mock_client_cls.call_args[0][0]
            assert client_constructor_config.base_url == "https://config.url/v1"
            assert client_constructor_config.default_headers["Authorization"] == "Bearer config-key-123"
            assert client_constructor_config.timeout == 99.0
            assert client_constructor_config.max_retries == 5
            mock_client.prepare.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_with_config_overriding_env(self, mock_client, setup_environment): # Renamed
        """Test ModelConfig overrides environment variables during initialization."""
        with patch("codin.model.openai_llm.Client", return_value=mock_client) as mock_client_cls:
            model_config = ModelConfig(api_key="config-override-key", base_url="https://config.override/v1")
            llm = await OpenAILLM(model="gpt-4-override", config=model_config) # Await

            mock_client_cls.assert_called_once()
            client_constructor_config = mock_client_cls.call_args[0][0]
            assert client_constructor_config.base_url == "https://config.override/v1"
            assert client_constructor_config.default_headers["Authorization"] == "Bearer config-override-key"
            mock_client.prepare.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_with_partial_config_and_env(self, mock_client, setup_environment): # Renamed
        """Test partial ModelConfig merges with environment variables during initialization."""
        with patch("codin.model.openai_llm.Client", return_value=mock_client) as mock_client_cls:
            model_config = ModelConfig(api_key="partial-config-key")
            llm = await OpenAILLM(model="gpt-4-partial", config=model_config) # Await

            mock_client_cls.assert_called_once()
            client_constructor_config = mock_client_cls.call_args[0][0]
            assert client_constructor_config.base_url == "https://env.api.test.com/v1"
            assert client_constructor_config.default_headers["Authorization"] == "Bearer partial-config-key"
            mock_client.prepare.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_missing_api_key_in_config_and_env(self, monkeypatch): # Renamed
        """Test init with API key missing in ModelConfig and environment raises ValueError."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        model_config_no_key = ModelConfig(base_url="https://some.url", api_key=None)

        expected_error_msg = 'API key not found. Set in ModelConfig, LLM_API_KEY, or OPENAI_API_KEY environment variable.'
        with pytest.raises(ValueError, match=expected_error_msg):
            await OpenAILLM(model="gpt-4-no-key", config=model_config_no_key) # Await

    @pytest.mark.asyncio
    async def test_generate_string_prompt_with_direct_config(self, mock_client, mock_response, monkeypatch):
        """Test generating text with ModelConfig passed directly."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        with patch("codin.model.openai_llm.Client", return_value=mock_client) as mock_client_cls:
            mock_client.post.return_value = mock_response

            model_config = ModelConfig(api_key="direct-cfg-key", base_url="https://direct.cfg/v1")
            llm = await OpenAILLM(model="gpt-4-direct", config=model_config) # Await

            response_content = await llm.generate("Hello from config test")
            assert response_content == "This is a test response."

            mock_client_cls.assert_called_once()
            client_constructor_config = mock_client_cls.call_args[0][0]
            assert client_constructor_config.base_url == "https://direct.cfg/v1"
            assert client_constructor_config.default_headers["Authorization"] == "Bearer direct-cfg-key"

            mock_client.post.assert_called_once()
            args, kwargs = mock_client.post.call_args
            assert kwargs["json"]["messages"] == [{"role": "user", "content": "Hello from config test"}]

    @pytest.mark.asyncio
    async def test_generate_string_prompt(self, mock_client, mock_response, setup_environment):
        """Test generating text with a string prompt (using env vars via default config)."""
        with patch("codin.model.openai_llm.Client", return_value=mock_client):
            mock_client.post.return_value = mock_response

            llm = await OpenAILLM("gpt-4") # Await, relies on env vars

            response = await llm.generate("What is the capital of France?")
            assert response == "This is a test response."

            mock_client.post.assert_called_once()
            args, kwargs = mock_client.post.call_args
            assert args[0] == "/chat/completions"
            assert kwargs["json"]["model"] == "gpt-4"
            assert kwargs["json"]["messages"] == [{"role": "user", "content": "What is the capital of France?"}]

    @pytest.mark.asyncio
    async def test_generate_message_prompt(self, mock_client, mock_response, setup_environment):
        """Test generating text with a message list prompt."""
        with patch("codin.model.openai_llm.Client", return_value=mock_client):
            mock_client.post.return_value = mock_response

            llm = await OpenAILLM("gpt-4", config=None) # Await

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
            response = await llm.generate(messages)
            assert response == "This is a test response."
            mock_client.post.assert_called_once()
            _, kwargs = mock_client.post.call_args
            assert kwargs["json"]["messages"] == messages

    @pytest.mark.asyncio
    async def test_generate_with_parameters(self, mock_client, mock_response, setup_environment):
        """Test generating text with additional parameters."""
        with patch("codin.model.openai_llm.Client", return_value=mock_client):
            mock_client.post.return_value = mock_response
            llm = await OpenAILLM("gpt-4") # Await

            await llm.generate(
                "What is the capital of France?",
                temperature=0.7,
                max_tokens=100,
                stop_sequences=[".", "!"]
            )
            _, kwargs = mock_client.post.call_args
            assert kwargs["json"]["temperature"] == 0.7
            assert kwargs["json"]["max_tokens"] == 100
            assert kwargs["json"]["stop"] == [".", "!"]

    @pytest.mark.asyncio
    async def test_streaming_response(self, mock_client, mock_stream_response, setup_environment):
        """Test streaming response generation."""
        with patch("codin.model.openai_llm.Client", return_value=mock_client):
            mock_client.post.return_value = mock_stream_response
            llm = await OpenAILLM("gpt-4") # Await

            stream = await llm.generate("What is the capital of France?", stream=True)
            chunks = [chunk async for chunk in stream]
            assert chunks == ["This", " is", " a", " test", " response."]
            mock_client.post.assert_called_once()
            _, kwargs = mock_client.post.call_args
            assert kwargs["json"]["stream"] is True

    @pytest.mark.asyncio
    async def test_generate_with_tools(self, mock_client, mock_function_call_response, setup_environment):
        """Test generating function calls with tools."""
        with patch("codin.model.openai_llm.Client", return_value=mock_client):
            mock_client.post.return_value = mock_function_call_response
            llm = await OpenAILLM("gpt-4") # Await

            tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather"}}]
            response = await llm.generate_with_tools("Weather in NY?", tools=tools)

            assert "tool_calls" in response
            if response.get("tool_calls"):
                 assert response["tool_calls"][0]["function"]["name"] == "get_weather"

            mock_client.post.assert_called_once()
            _, kwargs = mock_client.post.call_args
            assert kwargs["json"]["tools"] == tools

    @pytest.mark.asyncio
    async def test_close(self, mock_client, setup_environment):
        """Test closing the client."""
        with patch("codin.model.openai_llm.Client", return_value=mock_client):
            llm = await OpenAILLM("gpt-4") # Await
            await llm.close() # Close should still work
            mock_client.close.assert_called_once()

# Ensure an empty line at the end of the file if required by linters/formatters

    @pytest.mark.asyncio
    async def test_complete_response_json_decode_error(self, mock_client, setup_environment):
        """Test ModelResponseParsingError (or StreamProcessingError) on JSON decode error in _complete_response."""
        llm = await OpenAILLM("gpt-4")

        mock_http_response = AsyncMock(spec=httpx.Response)
        mock_http_response.json = MagicMock(side_effect=json.JSONDecodeError("err", "doc", 0))
        mock_http_response.text = "invalid json" # For logging

        with patch("src.codin.model.openai_llm.make_post_request", AsyncMock(return_value=mock_http_response)):
            with pytest.raises((ModelResponseParsingError, StreamProcessingError)): # Current _complete_response raises StreamProcessingError
                await llm._complete_response({"some": "payload"})

    @pytest.mark.asyncio
    async def test_complete_response_content_extraction_none(self, mock_client, setup_environment):
        """Test ContentExtractionError when extractor returns None in _complete_response."""
        llm = await OpenAILLM("gpt-4")

        mock_http_response = AsyncMock(spec=httpx.Response)
        # Valid JSON, but structure will make extractor return None (e.g. missing 'content')
        mock_http_response.json = MagicMock(return_value={"choices": [{"message": {"role": "assistant"}}]})

        with patch("src.codin.model.openai_llm.make_post_request", AsyncMock(return_value=mock_http_response)):
            with pytest.raises(ContentExtractionError, match="OpenAI content extraction for model gpt-4 failed: Extractor returned None"):
                await llm._complete_response({"some": "payload"})

    @pytest.mark.asyncio
    async def test_complete_tool_response_json_decode_error(self, mock_client, setup_environment):
        """Test ModelResponseParsingError (or StreamProcessingError) on JSON decode error in _complete_tool_response."""
        llm = await OpenAILLM("gpt-4")

        mock_http_response = AsyncMock(spec=httpx.Response)
        mock_http_response.json = MagicMock(side_effect=json.JSONDecodeError("err", "doc", 0))
        mock_http_response.text = "invalid json"

        with patch("src.codin.model.openai_llm.make_post_request", AsyncMock(return_value=mock_http_response)):
            with pytest.raises((ModelResponseParsingError, StreamProcessingError)): # Current _complete_tool_response raises StreamProcessingError
                await llm._complete_tool_response({"some": "payload"})

    @pytest.mark.asyncio
    async def test_complete_tool_response_missing_choices(self, mock_client, setup_environment):
        """Test ModelResponseParsingError for missing 'choices' in _complete_tool_response."""
        llm = await OpenAILLM("gpt-4")

        mock_http_response = AsyncMock(spec=httpx.Response)
        mock_http_response.json = MagicMock(return_value={"no_choices_here": True}) # Missing 'choices'

        with patch("src.codin.model.openai_llm.make_post_request", AsyncMock(return_value=mock_http_response)):
            with pytest.raises(ModelResponseParsingError, match="Missing 'choices' in OpenAI tool response for model gpt-4."):
                await llm._complete_tool_response({"some": "payload"})

    @pytest.mark.asyncio
    async def test_complete_tool_response_missing_message(self, mock_client, setup_environment):
        """Test ModelResponseParsingError for missing 'message' in _complete_tool_response."""
        llm = await OpenAILLM("gpt-4")

        mock_http_response = AsyncMock(spec=httpx.Response)
        mock_http_response.json = MagicMock(return_value={"choices": [{"no_message_here": True}]}) # Missing 'message'

        with patch("src.codin.model.openai_llm.make_post_request", AsyncMock(return_value=mock_http_response)):
            with pytest.raises(ModelResponseParsingError, match="Missing 'message' in OpenAI tool response choice for model gpt-4."):
                await llm._complete_tool_response({"some": "payload"})

    @pytest.mark.asyncio
    async def test_stream_response_delta_extraction_error(self, mock_client, setup_environment):
        """Test StreamProcessingError if delta_extractor fails in _stream_response."""
        llm = await OpenAILLM("gpt-4")

        # Mock make_post_request to return a response that can be iterated
        mock_http_stream_response = AsyncMock(spec=httpx.Response)
        async def faulty_aiter_lines():
            yield "data: {\"choices\": [{\"delta\": \"not_a_dict\"}]}" # This will cause extractor to fail if it expects a dict
            # yield "data: [DONE]" # Not strictly needed if extractor fails first
        mock_http_stream_response.aiter_lines = faulty_aiter_lines
        mock_http_stream_response.aclose = AsyncMock()


        with patch("src.codin.model.openai_llm.make_post_request", AsyncMock(return_value=mock_http_stream_response)):
            # Patch the specific helper method that might fail
            with patch.object(llm, '_extract_delta_from_stream_chunk', side_effect=TypeError("Simulated type error in delta extraction")):
                with pytest.raises(StreamProcessingError, match="Delta extraction failed for chunk"):
                    stream = await llm._stream_response({"some": "payload"})
                    async for _ in stream: pass # Consume the stream to trigger error
