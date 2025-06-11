"""Anthropic Claude LLM integration for codin agents.

This module provides integration with Anthropic's Claude language models
through their API, supporting both streaming and non-streaming responses.
"""

from __future__ import annotations

import json
import logging
import os
import typing as _t

from ..client import Client, ClientConfig, LoggingTracer
from .base import BaseLLM
from .config import ModelConfig
from .registry import register # Changed
from .http_utils import (
    make_post_request,
    extract_content_from_json,
    process_sse_stream,
    ContentExtractionError,
    StreamProcessingError
)

__all__ = [
    'AnthropicLLM',
]

logger = logging.getLogger('codin.model.anthropic_llm')


@register # Changed
class AnthropicLLM(BaseLLM):
    """Implementation of BaseLLM for Anthropic API.

    Supports both streaming and non-streaming generation.

    Environment variables:
        LLM_PROVIDER: The LLM provider (should be 'anthropic' for this class)
        LLM_API_KEY: The API key for the LLM provider
        LLM_BASE_URL: Base URL for the API (defaults to https://api.anthropic.com)
        LLM_MODEL: The model to use (defaults to claude-3-sonnet-20240229)

        Legacy environment variables (deprecated, will be removed):
        ANTHROPIC_API_KEY: The API key for Anthropic
        ANTHROPIC_API_BASE: Base URL for the API
        ANTHROPIC_API_VERSION: API version
    """
    DEFAULT_MODEL = 'claude-3-sonnet-20240229'
    DEFAULT_BASE_URL = 'https://api.anthropic.com'
    DEFAULT_API_VERSION = '2023-06-01'
    DEFAULT_TIMEOUT = 60.0
    # Anthropic doesn't have explicit connect_timeout in ClientConfig, but we can use ModelConfig's
    # Default retry policies are often handled by the underlying HTTP client library if not specified.

    async def __init__(self, config: _t.Optional[ModelConfig] = None, model: str | None = None): # Changed to async
        """Initialize and prepare the Anthropic LLM.

        Args:
            config: Optional ModelConfig instance. If None, a default config is used,
                    and settings are primarily sourced from environment variables.
            model: Optional model name to override config or environment settings.
        """
        self.config = config or ModelConfig()

        # Determine model name: constructor arg > config > env > default
        chosen_model = model
        if chosen_model is None and self.config.model_name:
            chosen_model = self.config.model_name
        if chosen_model is None:
            chosen_model = os.getenv('LLM_MODEL') # LLM_MODEL is generic

        self.model = chosen_model or self.DEFAULT_MODEL
        super().__init__(self.model) # Call super class __init__

        # Client initialization logic moved from prepare()
        # API Key: config > env (generic) > env (specific)
        api_key = self.config.api_key or \
                  os.getenv('LLM_API_KEY') or \
                  os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError(
                'API key not found. Set in ModelConfig, LLM_API_KEY, or ANTHROPIC_API_KEY environment variable.'
            )

        # Base URL: config > env (generic) > env (specific) > default
        base_url = self.config.base_url or \
                   os.getenv('LLM_BASE_URL') or \
                   os.getenv('ANTHROPIC_API_BASE') or \
                   self.DEFAULT_BASE_URL
        if base_url.endswith('/'): # Ensure no trailing slash
            base_url = base_url[:-1]

        # API Version: config > env (specific) > default
        api_version = self.config.api_version or \
                      os.getenv('ANTHROPIC_API_VERSION') or \
                      self.DEFAULT_API_VERSION

        client_kwargs = self.config.get_client_config_kwargs()
        client_kwargs.setdefault('timeout', self.DEFAULT_TIMEOUT)
        client_kwargs['default_headers'] = {
            'x-api-key': api_key,
            'anthropic-version': api_version,
            'Content-Type': 'application/json',
        }
        client_kwargs['base_url'] = base_url

        if logger.isEnabledFor(logging.DEBUG) and 'tracers' not in client_kwargs:
            client_kwargs['tracers'] = [LoggingTracer()]
        elif 'tracers' not in client_kwargs:
             client_kwargs['tracers'] = []

        client_config = ClientConfig(**client_kwargs)
        self._client = Client(client_config)
        # await self._client.prepare() # Removed, Client.__init__ now handles full setup

        logger.info(f'Anthropic LLM initialized and client prepared for model {self.model} at {base_url}')

    # prepare() method is now removed.

    @classmethod
    def supported_models(cls) -> list[str]:
        """Supported models for Anthropic LLM."""
        return [
            r'claude-.*',
        ]

    def _prepare_messages(self, prompt: str | list[dict[str, str]]) -> tuple[list[dict], str | None]:
        """Convert prompt to Anthropic message format.

        Args:
            prompt: Either a string or a list of message dictionaries

        Returns:
            List of messages in Anthropic format and optional system prompt
        """
        if isinstance(prompt, str):
            return [{'role': 'user', 'content': prompt}], None

        # Map OpenAI roles to Anthropic roles
        anthropic_messages = []
        system_prompt = None

        for msg in prompt:
            role = msg['role']
            content = msg['content']

            if role == 'system':
                system_prompt = content
            elif role == 'user':
                anthropic_messages.append({'role': 'user', 'content': content})
            elif role == 'assistant':
                anthropic_messages.append({'role': 'assistant', 'content': content})
            else:
                logger.warning(f'Unsupported role: {role}, treating as user')
                anthropic_messages.append({'role': 'user', 'content': content})

        return anthropic_messages, system_prompt

    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> _t.AsyncIterator[str] | str:
        """Generate text using Anthropic API."""
        if not self._client: # Should not happen if __init__ completes successfully
            raise RuntimeError('Anthropic client not initialized. This should not happen after async __init__.')
        # client variable is no longer needed as methods will use self._client
        # client = self._client

        # Extract the system prompt if any
        messages, system_prompt = self._prepare_messages(prompt)

        # Prepare request payload
        payload = {
            'model': self.model,
            'messages': messages,
            'stream': stream,
        }

        # Add system prompt if present
        if system_prompt:
            payload['system'] = system_prompt

        # Add optional parameters
        if temperature is not None:
            payload['temperature'] = temperature

        if max_tokens is not None:
            payload['max_tokens'] = max_tokens

        if stop_sequences:
            payload['stop_sequences'] = stop_sequences

        # Send request
        if stream:
            return self._stream_response(payload) # Use self._client internally
        return await self._complete_response(payload) # Use self._client internally

    def _extract_content_from_response(self, response_data: dict) -> _t.Optional[str]:
        """Helper to extract content from Anthropic's non-streaming response JSON."""
        content_blocks = response_data.get('content')
        if not content_blocks or not isinstance(content_blocks, list):
            # Return None if no content blocks, let extract_content_from_json handle raising error if content is expected.
            # Or, if "" is acceptable for "no content", return that. Given current behavior, None is better.
            return None

        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get('type') == 'text':
                text_parts.append(block.get('text', ''))
        return "".join(text_parts)

    def _extract_delta_from_stream_chunk(self, data_chunk: dict) -> _t.Optional[str]:
        """Helper to extract content delta from Anthropic's streaming response chunk."""
        if data_chunk.get('type') == 'content_block_delta':
            delta = data_chunk.get('delta')
            if isinstance(delta, dict) and delta.get('type') == 'text_delta':
                return delta.get('text')
        return None

    async def _complete_response(self, payload: dict) -> str:
        """Handle a complete (non-streaming) response."""
        if not self._client:
            raise RuntimeError(f"Anthropic client for model {self.model} not initialized in _complete_response.")

        response = await make_post_request(
            self._client,
            '/v1/messages',
            payload,
            error_message_prefix=f"Anthropic API request for model {self.model} failed"
        )
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from Anthropic for model {self.model}: {e}. Response text: {response.text}")
            raise StreamProcessingError(f"Failed to decode JSON response: {e}") from e # Or ModelResponseParsingError

        try:
            return extract_content_from_json(
                response_data,
                self._extract_content_from_response, # Use helper
                error_message_prefix=f"Anthropic content extraction for model {self.model} failed"
            )
        except ContentExtractionError as e: # Catches ModelResponseParsingError too if it's a base
            logger.error(f"Anthropic content extraction error for model {self.model}: {e}")
            raise

    async def _stream_response(self, payload: dict) -> _t.AsyncIterator[str]:
        """Handle a streaming response."""
        if not self._client:
            raise RuntimeError(f"Anthropic client for model {self.model} not initialized in _stream_response.")

        response = await make_post_request(
            self._client,
            '/v1/messages',
            payload,
            error_message_prefix=f"Anthropic streaming API request for model {self.model} failed"
        )

        try:
            return process_sse_stream(
                response,
                delta_extractor=self._extract_delta_from_stream_chunk, # Use helper
                stop_marker=None, # Anthropic doesn't use a data line like "[DONE]"
                error_message_prefix="Anthropic streaming processing failed"
            )
        except StreamProcessingError as e:
            logger.error(f"Anthropic stream processing error: {e}")
            raise

    async def generate_with_tools(
        self,
        prompt: str | list[dict[str, str]],
        tools: list[dict],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict | _t.AsyncIterator[dict]:
        """Generate text with function/tool calling capabilities.

        Note: This is a placeholder implementation as Anthropic's tool calling
        API might differ from OpenAI's. This needs to be updated when Anthropic
        releases their tool calling API.
        """
        raise NotImplementedError('Tool calling is not yet implemented for Anthropic')

    async def close(self):
        """Close the client and release resources."""
        if self._client:
            await self._client.close()
            self._client = None

    def __del__(self):
        """Cleanup on deletion."""
        if self._client:
            # Can't await in __del__, so just log a warning if not closed properly
            # This matches the pattern in OpenAILLM for consistency
            # The original asyncio.create_task can lead to issues if the loop is closed
            logger.warning('AnthropicLLM was deleted without calling close(). This may leak resources.')
