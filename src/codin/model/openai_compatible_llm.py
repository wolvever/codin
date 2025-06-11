"""
Base class for LLMs that are compatible with the OpenAI Chat Completions API.
"""
from __future__ import annotations

import json
import logging
import os
import typing as _t

from ..client import Client, ClientConfig
from .base import BaseLLM
from .config import ModelConfig
from .http_utils import (
    make_post_request,
    extract_content_from_json,
    process_sse_stream,
    ContentExtractionError,
    ModelResponseParsingError,
    StreamProcessingError
)

logger = logging.getLogger(__name__)

class OpenAICompatibleBaseLLM(BaseLLM):
    """
    Base class for language models that adhere to the OpenAI Chat Completions API format.
    This includes models from OpenAI itself, Azure OpenAI, and other providers offering
    OpenAI-compatible endpoints.
    """

    # Default values, subclasses like OpenAILLM can override these
    DEFAULT_MODEL = 'gpt-4o-mini' # A generic default
    DEFAULT_BASE_URL = 'https://api.openai.com/v1' # Standard OpenAI
    DEFAULT_TIMEOUT = 120.0
    DEFAULT_CONNECT_TIMEOUT = 30.0
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_MIN_WAIT = 1.0
    DEFAULT_RETRY_MAX_WAIT = 10.0
    DEFAULT_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

    API_TYPE_ENV_VAR: _t.Optional[str] = None # e.g. "OPENAI_API_TYPE" for Azure
    API_VERSION_ENV_VAR: _t.Optional[str] = None # e.g. "OPENAI_API_VERSION" for Azure
    API_KEY_ENV_VAR = 'OPENAI_API_KEY' # Fallback generic OpenAI key
    LLM_API_KEY_ENV_VAR = 'LLM_API_KEY' # Generic key
    BASE_URL_ENV_VAR = 'OPENAI_API_BASE' # Fallback generic OpenAI base
    LLM_BASE_URL_ENV_VAR = 'LLM_BASE_URL' # Generic base
    MODEL_ENV_VAR = 'OPENAI_MODEL' # Fallback generic OpenAI model
    LLM_MODEL_ENV_VAR = 'LLM_MODEL' # Generic model

    def __init__(self, config: _t.Optional[ModelConfig] = None, model: str | None = None):
        # This __init__ must be async due to client setup.
        # The prompt says "async def __init__ ... similar to OpenAILLM"
        # This was a mistake in my plan, as I just refactored OpenAILLM to have async __init__.
        # This class's __init__ must also be async.
        # Re-correcting the thought process for the actual implementation.
        pass # Will be implemented as async in the actual tool call.


    async def _init_client(self, config: _t.Optional[ModelConfig] = None, model_name_arg: str | None = None):
        """
        Asynchronous part of initialization. Called by __init__.
        Sets up model name, configuration, and HTTP client.
        """
        self.config = config or ModelConfig()

        chosen_model = model_name_arg
        if chosen_model is None and self.config.model_name:
            chosen_model = self.config.model_name
        if chosen_model is None:
            chosen_model = os.getenv(self.LLM_MODEL_ENV_VAR) or os.getenv(self.MODEL_ENV_VAR)

        self.model = chosen_model or self.DEFAULT_MODEL
        super().__init__(self.model)

        api_key = self.config.api_key or os.getenv(self.LLM_API_KEY_ENV_VAR) or os.getenv(self.API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(
                f"API key not found. Set in ModelConfig or via {self.LLM_API_KEY_ENV_VAR} or {self.API_KEY_ENV_VAR}."
            )

        base_url = self.config.base_url or \
                   os.getenv(self.LLM_BASE_URL_ENV_VAR) or \
                   os.getenv(self.BASE_URL_ENV_VAR) or \
                   self.DEFAULT_BASE_URL

        client_kwargs = self.config.get_client_config_kwargs()
        client_kwargs.setdefault('timeout', self.DEFAULT_TIMEOUT)
        client_kwargs.setdefault('connect_timeout', self.DEFAULT_CONNECT_TIMEOUT)
        client_kwargs.setdefault('max_retries', self.DEFAULT_MAX_RETRIES)
        client_kwargs.setdefault('retry_min_wait', self.DEFAULT_RETRY_MIN_WAIT)
        client_kwargs.setdefault('retry_max_wait', self.DEFAULT_RETRY_MAX_WAIT)
        client_kwargs.setdefault('retry_on_status_codes', self.DEFAULT_RETRY_STATUS_CODES)

        client_kwargs['base_url'] = base_url
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}

        # For Azure compatibility primarily
        if self.API_TYPE_ENV_VAR and os.getenv(self.API_TYPE_ENV_VAR):
            headers['api-type'] = os.environ[self.API_TYPE_ENV_VAR]
        if self.API_VERSION_ENV_VAR and os.getenv(self.API_VERSION_ENV_VAR) and self.config.api_version is None: # config takes precedence
            # This is usually passed as a query param for Azure, e.g. ?api-version=YYYY-MM-DD
            # Or handled by specific Azure clients. For generic OpenAI-compatible, it might be a header or not used.
            # For now, if an api_version is in config, it will be used for header construction if needed by a subclass.
            # Let's assume it's a header if specified in config for generic case, or subclass handles it.
             pass
        if self.config.api_version: # If api_version is in ModelConfig
            # How it's used depends on the provider. OpenAI doesn't use it in headers.
            # Azure might use it as a query param "api-version". Some custom OpenAI-compatible might use a header.
            # For now, let's assume it's not a default header unless a subclass adds it.
            # If using Azure, specific Azure client or logic in subclass would handle it.
            pass


        client_kwargs['default_headers'] = headers

        client_config = ClientConfig(**client_kwargs)
        self._client = Client(client_config)
        # No await self._client.prepare() as Client.__init__ is now synchronous and complete.
        logger.info(f'{self.__class__.__name__} initialized for model {self.model} at {base_url}')

    # Actual __init__ must be async to call _init_client
    async def __init__(self, config: _t.Optional[ModelConfig] = None, model: str | None = None):
        await self._init_client(config=config, model_name_arg=model)

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r'.*'] # Generic, subclasses should override

    def _prepare_messages(self, prompt: str | list[dict[str, str]]) -> list[dict[str, str]]:
        """Convert prompt to OpenAI message format."""
        if isinstance(prompt, str):
            return [{'role': 'user', 'content': prompt}]
        # TODO: Potentially validate message structure here
        return prompt

    def _extract_content_from_response(self, response_data: dict) -> _t.Optional[str]:
        """Helper to extract content from OpenAI's non-streaming chat response JSON."""
        choices = response_data.get('choices')
        if choices and isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get('message')
            if message and isinstance(message, dict):
                return message.get('content')
        return None

    def _extract_delta_from_stream_chunk(self, data_chunk: dict) -> _t.Optional[str]:
        """Helper to extract content delta from OpenAI's streaming chat response chunk."""
        choices = data_chunk.get('choices')
        if choices and isinstance(choices, list) and len(choices) > 0:
            delta = choices[0].get('delta')
            if delta and isinstance(delta, dict):
                return delta.get('content')
        return None

    async def _handle_completion_response(self, payload: dict) -> str:
        """Shared logic for handling complete (non-streaming) chat responses."""
        response = await make_post_request(
            self._client,
            '/chat/completions',
            payload,
            error_message_prefix=f"{self.__class__.__name__} API request for model {self.model} failed"
        )
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from {self.__class__.__name__} for model {self.model}: {e}. Response text: {response.text}")
            raise ModelResponseParsingError(f"Failed to decode JSON response: {e}") from e

        return extract_content_from_json(
            response_data,
            self._extract_content_from_response,
            error_message_prefix=f"{self.__class__.__name__} content extraction for model {self.model} failed"
        )

    async def _handle_streaming_response(self, payload: dict) -> _t.AsyncIterator[str]:
        """Shared logic for handling streaming chat responses."""
        response = await make_post_request(
            self._client,
            '/chat/completions',
            payload,
            error_message_prefix=f"{self.__class__.__name__} streaming API request for model {self.model} failed"
        )
        return process_sse_stream(
            response,
            delta_extractor=self._extract_delta_from_stream_chunk,
            stop_marker="[DONE]", # Common for OpenAI and compatible APIs
            error_message_prefix=f"{self.__class__.__name__} streaming processing for model {self.model} failed"
        )

    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        # Add other common OpenAI params like top_p, presence_penalty, frequency_penalty if needed
    ) -> _t.AsyncIterator[str] | str:
        if not self._client:
            raise RuntimeError(f'{self.__class__.__name__} client not initialized. This should not happen after async __init__.')

        messages = self._prepare_messages(prompt)
        payload: dict[str, _t.Any] = {'model': self.model, 'messages': messages, 'stream': stream}

        if temperature is not None: payload['temperature'] = temperature
        if max_tokens is not None: payload['max_tokens'] = max_tokens
        if stop_sequences: payload['stop'] = stop_sequences

        if stream:
            return await self._handle_streaming_response(payload)
        return await self._handle_completion_response(payload)

    def _parse_tool_calls_from_response(self, response_data: dict) -> _t.Optional[list[dict]]:
        """Helper to extract tool_calls from OpenAI's non-streaming response."""
        choices = response_data.get('choices')
        if choices and isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get('message')
            if message and isinstance(message, dict) and message.get('tool_calls'):
                return message['tool_calls']
        return None

    async def _handle_tool_call_response(self, payload: dict) -> dict:
        """Shared logic for handling complete (non-streaming) tool call responses."""
        response = await make_post_request(
            self._client,
            '/chat/completions',
            payload,
            error_message_prefix=f"{self.__class__.__name__} tool API request for model {self.model} failed"
        )
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON tool response from {self.__class__.__name__} for model {self.model}: {e}. Response text: {response.text}")
            raise ModelResponseParsingError(f"Failed to decode JSON tool response: {e}") from e

        result = {}
        content = self._extract_content_from_response(response_data) # Check for text content
        if content:
            result['content'] = content

        tool_calls = self._parse_tool_calls_from_response(response_data)
        if tool_calls:
            result['tool_calls'] = tool_calls

        if not result:
            logger.warning(f"{self.__class__.__name__} tool response for model {self.model} had no content or tool_calls.")
            # Raise error if neither content nor tool_calls are present, as it's unexpected for a tool call scenario
            raise ModelResponseParsingError(f"No content or tool_calls found in tool response for model {self.model}")

        return result

    async def _handle_tool_call_streaming_response(self, payload: dict) -> _t.AsyncIterator[dict]:
        """Shared logic for handling streaming tool call responses."""
        response = await make_post_request(
            self._client,
            '/chat/completions',
            payload,
            error_message_prefix=f"{self.__class__.__name__} streaming tool API for model {self.model} failed"
        )

        # Simplified streaming for tool calls: yield content delta or full tool_calls part.
        # More complex aggregation of tool call deltas might be needed for some applications.
        async def stream_generator():
            async for line in response.aiter_lines():
                line = line.strip()
                if not line: continue
                if line.startswith("data: "):
                    data_str = line[len("data: "):].strip()
                    if data_str == "[DONE]": break
                    try:
                        data_chunk = json.loads(data_str)
                        content_delta = self._extract_delta_from_stream_chunk(data_chunk)
                        if content_delta:
                            yield {'content': content_delta}

                        # Check for tool_calls in delta (OpenAI streams these)
                        choices = data_chunk.get('choices')
                        if choices and isinstance(choices, list) and len(choices) > 0:
                            delta = choices[0].get('delta')
                            if delta and isinstance(delta, dict) and delta.get('tool_calls'):
                                yield {'tool_calls': delta['tool_calls']}
                    except json.JSONDecodeError:
                        logger.warning(f"{self.__class__.__name__} streaming tool: Failed to parse SSE JSON for model {self.model}: {data_str}")
                        continue
        try:
            async for item in stream_generator():
                yield item
        except Exception as e:
            logger.error(f'{self.__class__.__name__} streaming tool processing for model {self.model} failed: {e}', exc_info=True)
            raise StreamProcessingError(f"Streaming tool processing for model {self.model} failed: {e}") from e
        finally:
            await response.aclose()


    async def generate_with_tools(
        self,
        prompt: str | list[dict[str, str]],
        tools: list[dict],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: _t.Optional[str | dict] = "auto", # OpenAI specific: "auto", "none", or {"type": "function", "function": {"name": "my_function"}}
    ) -> dict | _t.AsyncIterator[dict]:
        if not self._client:
            raise RuntimeError(f'{self.__class__.__name__} client not initialized. This should not happen after async __init__.')

        messages = self._prepare_messages(prompt)
        payload: dict[str, _t.Any] = {
            'model': self.model,
            'messages': messages,
            'tools': tools,
            'tool_choice': tool_choice, # Add tool_choice
            'stream': stream
        }

        if temperature is not None: payload['temperature'] = temperature
        if max_tokens is not None: payload['max_tokens'] = max_tokens

        if stream:
            return self._handle_tool_call_streaming_response(payload)
        return await self._handle_tool_call_response(payload)

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if hasattr(self, '_client') and self._client:
            await self._client.close()
            self._client = None # type: ignore [assignment] # To satisfy type checker if _client is not Optional

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_client') and self._client:
            logger.warning(f'{self.__class__.__name__} (model: {self.model}) was deleted without calling close(). This may leak resources.')
