"""OpenAI LLM integration for codin agents.

This module provides integration with OpenAI's language models
through their API, supporting both streaming and non-streaming responses.
"""

from __future__ import annotations

import json
import logging
import os
import typing as _t

# Removed direct Client, ClientConfig imports if not used directly by OpenAILLM itself
# from ..client import Client, ClientConfig
from .openai_compatible_llm import OpenAICompatibleBaseLLM # Import new base class
from .config import ModelConfig # Already here
from .registry import register # Changed from ModelRegistry to module-level register
# HTTP utils are used by the new base class, so direct imports might not be needed here
# from .http_utils import (...)

__all__ = [
    'OpenAILLM',
]

logger = logging.getLogger('codin.model.openai_llm') # Logger can remain


@register # Changed from @ModelRegistry.register
class OpenAILLM(OpenAICompatibleBaseLLM): # Inherit from new base
    """
    Specific OpenAI LLM implementation.
    This class leverages OpenAICompatibleBaseLLM and specializes it for OpenAI models,
    primarily by defining specific supported model patterns and potentially any OpenAI-specific
    default configurations or minor behavior overrides if needed in the future.

    Environment variables:
        LLM_PROVIDER: The LLM provider (should be 'openai' for this class)
        LLM_API_KEY: The API key for the LLM provider
        LLM_BASE_URL: Base URL for the API (defaults to https://api.openai.com/v1)
        LLM_MODEL: The model to use (defaults to gpt-4o)

        Legacy environment variables (deprecated, will be removed):
        OPENAI_API_KEY: The API key for OpenAI
        OPENAI_API_BASE: Base URL for the API
    """

    DEFAULT_MODEL = 'gpt-4o-mini'
    DEFAULT_BASE_URL = 'https://api.openai.com/v1'
    DEFAULT_TIMEOUT = 120.0
    DEFAULT_CONNECT_TIMEOUT = 30.0
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_MIN_WAIT = 1.0
    DEFAULT_RETRY_MAX_WAIT = 10.0
    DEFAULT_RETRY_STATUS_CODES = [429, 500, 502, 503, 504] # Kept for reference, base might use them

    # Override ENV VAR names if OpenAI uses specific ones not covered by the generic LLM_ ones
    # For OpenAI, the generic ones in OpenAICompatibleBaseLLM are often sufficient.
    # API_KEY_ENV_VAR = 'OPENAI_API_KEY' # Already a default in the new base
    # BASE_URL_ENV_VAR = 'OPENAI_API_BASE' # Already a default
    # MODEL_ENV_VAR = 'OPENAI_MODEL' # Already a default

    async def __init__(self, config: _t.Optional[ModelConfig] = None, model: str | None = None):
        """
        Initialize the OpenAI LLM.
        Relies on OpenAICompatibleBaseLLM for client setup and core logic.
        """
        # Determine model name with OpenAILLM defaults before calling super().__init__
        # This ensures that if 'model' is None and config.model_name is None,
        # OpenAILLM's specific defaults (OPENAI_MODEL, self.DEFAULT_MODEL) are used.

        final_config = config or ModelConfig()

        chosen_model = model
        if chosen_model is None and final_config.model_name:
            chosen_model = final_config.model_name
        if chosen_model is None: # Fallback to env vars specific to OpenAILLM or general LLM
            chosen_model = os.getenv(self.MODEL_ENV_VAR) or \
                           os.getenv(self.LLM_MODEL_ENV_VAR) or \
                           os.getenv('OPENAI_MODEL') # Explicitly check old one too for this class

        final_model = chosen_model or self.DEFAULT_MODEL

        # Call the async __init__ of the new base class
        await super().__init__(config=final_config, model=final_model)
        # Logger message from base class __init__ will cover initialization.
        # logger.info(f'OpenAILLM specific initialization complete for model {self.model}') # Optional

    @classmethod
    def supported_models(cls) -> list[str]: # Reinstated
        """Supported models for OpenAI LLM."""
        return [
            # GPT models
            r'gpt-3.5-turbo.*',
            r'gpt-4.*',
            r'gpt-4o.*',
            r'gpt-4-vision.*',
            # Claude models through OpenAI compat
            r'claude-.*',
            # Any o1- models (OpenAI compatible models)
            r'o1-.*',
        ]

    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> _t.AsyncIterator[str] | str:
        if not self._client: # Should not happen if __init__ completes successfully
            raise RuntimeError('OpenAI client not initialized. This should not happen after async __init__.')

        # Convert string prompt to messages format if needed
        messages = self._prepare_messages(prompt)

        # Prepare request parameters
        payload = {'model': self.model, 'messages': messages, 'stream': stream}

        # Add optional parameters if provided
        if temperature is not None:
            payload['temperature'] = temperature
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens
        if stop_sequences:
            payload['stop'] = stop_sequences

        if stream:
            return await self._stream_response(payload)
        return await self._complete_response(payload)

    def _extract_content_from_response(self, response_data: dict) -> _t.Optional[str]:
        """Helper to extract content from OpenAI's non-streaming response JSON."""
        choices = response_data.get('choices')
        if choices and isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get('message')
            if message and isinstance(message, dict):
                return message.get('content')
        return None

    def _extract_delta_from_stream_chunk(self, data_chunk: dict) -> _t.Optional[str]:
        """Helper to extract content delta from OpenAI's streaming response chunk."""
        choices = data_chunk.get('choices')
        if choices and isinstance(choices, list) and len(choices) > 0:
            delta = choices[0].get('delta')
            if delta and isinstance(delta, dict):
                return delta.get('content')
        return None

    async def _complete_response(self, payload: dict) -> str:
        """Handle a complete (non-streaming) response."""
        response = await make_post_request(
            self._client,
            '/chat/completions',
            payload,
            error_message_prefix=f"OpenAI API request for model {self.model} failed"
        )
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from OpenAI for model {self.model}: {e}. Response text: {response.text}")
            raise StreamProcessingError(f"Failed to decode JSON response: {e}") from e # Or ModelResponseParsingError

        try:
            return extract_content_from_json(
                response_data,
                self._extract_content_from_response, # Use helper method
                error_message_prefix=f"OpenAI content extraction for model {self.model} failed"
            )
        except ContentExtractionError as e: # ContentExtractionError or ModelResponseParsingError
            logger.error(f"OpenAI content extraction error for model {self.model}: {e}")
            raise

    async def _stream_response(self, payload: dict) -> _t.AsyncIterator[str]:
        """Handle a streaming response."""
        response = await make_post_request(
            self._client,
            '/chat/completions',
            payload,
            error_message_prefix=f"OpenAI streaming API request for model {self.model} failed"
        )

        try:
            return process_sse_stream(
                response,
                delta_extractor=self._extract_delta_from_stream_chunk, # Use helper method
                stop_marker="[DONE]",
                error_message_prefix="OpenAI streaming processing failed"
            )
        except StreamProcessingError as e:
            # Log and re-raise to match previous behavior
            logger.error(f"OpenAI stream processing error: {e}")
            raise


    def _prepare_messages(self, prompt: str | list[dict[str, str]]) -> list[dict[str, str]]:
        """Convert prompt to OpenAI message format."""
        if isinstance(prompt, str):
            return [{'role': 'user', 'content': prompt}]
        return prompt

    async def generate_with_tools(
        self,
        prompt: str | list[dict[str, str]],
        tools: list[dict],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict | _t.AsyncIterator[dict]:
        if not self._client: # Should not happen if __init__ completes successfully
            raise RuntimeError('OpenAI client not initialized. This should not happen after async __init__.')

        # Convert string prompt to messages format if needed
        messages = self._prepare_messages(prompt)

        # Prepare request parameters
        payload = {'model': self.model, 'messages': messages, 'tools': tools, 'stream': stream}

        # Add optional parameters if provided
        if temperature is not None:
            payload['temperature'] = temperature
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens

        if stream:
            return await self._stream_tool_response(payload)
        return await self._complete_tool_response(payload)

    async def _complete_tool_response(self, payload: dict) -> dict:
        """Handle a complete (non-streaming) response with tool calls."""
        response = await make_post_request(
            self._client,
            '/chat/completions',
            payload,
            error_message_prefix=f"OpenAI tool API request for model {self.model} failed"
        )
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON tool response from OpenAI for model {self.model}: {e}. Response text: {response.text}")
            raise StreamProcessingError(f"Failed to decode JSON tool response: {e}") from e # Or ModelResponseParsingError

        # Custom extraction logic for tool responses
        try:
            choices = response_data.get('choices')
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                # Using ModelResponseParsingError as structure is unexpected
                raise ModelResponseParsingError(f"Missing 'choices' in OpenAI tool response for model {self.model}.")

            message = choices[0].get('message')
            if not message or not isinstance(message, dict):
                raise ModelResponseParsingError(f"Missing 'message' in OpenAI tool response choice for model {self.model}.")

            result = {}
            if message.get('content'): # Content can be None
                result['content'] = message['content']
            if message.get('tool_calls'):
                result['tool_calls'] = message['tool_calls']

            if not result.get('content') and not result.get('tool_calls'):
                logger.warning(f"OpenAI tool response for model {self.model} had no content or tool_calls.")
                # This might be a valid empty response, or an issue.
                # Depending on strictness, one might raise ContentExtractionError if content/tool_calls are expected.
                # For now, allow empty result dict.

            return result
        except ModelResponseParsingError: # Re-raise specific parsing errors
            raise
        except Exception as e: # Catch broader errors during extraction
            logger.error(f'OpenAI tool response processing for model {self.model} failed: {e}.', exc_info=True)
            logger.debug(f"Problematic data for tool response processing: {response_data}") # Log full data at DEBUG
            raise ContentExtractionError(f'Failed to process tool response for model {self.model}: {e}') from e


    async def _stream_tool_response(self, payload: dict) -> _t.AsyncIterator[dict]:
        """Handle a streaming response with tool calls."""
        response = await make_post_request(
            self._client,
            '/chat/completions',
            payload,
            error_message_prefix="OpenAI streaming tool API request failed"
        )

        # SSE processing for tool calls is complex and stateful.
        # The generic process_sse_stream is designed for simpler string content streams.
        async def stream_generator():
            # accumulated_content and accumulated_tool_calls can be useful if a full final
            # object needs to be constructed or for debugging, but primary yield is per delta.
            # For this refactoring, we focus on yielding deltas as they come.
            # accumulated_content = ''
            # accumulated_tool_calls = []

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith("data: "):
                    data_str = line[len("data: "):].strip()
                    if data_str == "[DONE]":
                        break # Signal end of stream.

                    try:
                        data_chunk = json.loads(data_str)
                        choices = data_chunk.get('choices')
                        if choices and isinstance(choices, list) and len(choices) > 0:
                            delta = choices[0].get('delta', {})

                            content_delta = delta.get('content')
                            if content_delta:
                                # accumulated_content += content_delta # Accumulate if needed for other logic
                                yield {'content': content_delta}

                            tool_calls_delta = delta.get('tool_calls')
                            if tool_calls_delta:
                                # OpenAI streams tool calls, potentially in parts.
                                # Yielding them directly as they arrive. Robust consumer should handle aggregation if needed.
                                yield {'tool_calls': tool_calls_delta}
                                # accumulated_tool_calls.extend(tool_calls_delta) # Accumulate if needed

                    except json.JSONDecodeError:
                        logger.warning(f"OpenAI streaming tool: Failed to parse SSE JSON: {data_str}")
                        continue

        # Ensure the response is closed after the generator is exhausted.
        try:
            async for item in stream_generator():
                yield item
        except Exception as e:
            logger.error(f'OpenAI streaming tool API processing failed: {e}')
            raise StreamProcessingError(f"OpenAI streaming tool processing failed: {e}") from e
        finally:
            await response.aclose()

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client:
            await self._client.close()
            self._client = None

    def __del__(self):
        """Cleanup on deletion."""
        if self._client:
            # Can't await in __del__, so just log a warning
            logger.warning('OpenAILLM was deleted without calling close(). This may leak resources.')
