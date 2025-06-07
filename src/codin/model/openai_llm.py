"""OpenAI LLM integration for codin agents.

This module provides integration with OpenAI's language models
through their API, supporting both streaming and non-streaming responses.
"""

from __future__ import annotations

import json
import logging
import os
import typing as _t

from ..client import Client, ClientConfig
from .base import BaseLLM
from .registry import ModelRegistry


__all__ = [
    'OpenAILLM',
]

logger = logging.getLogger('codin.model.openai_llm')


@ModelRegistry.register
class OpenAILLM(BaseLLM):
    """Implementation of BaseLLM for OpenAI API and compatible services (e.g., Azure OpenAI).

    Supports both streaming and non-streaming generation, and function/tool calling.

    Environment variables:
        LLM_PROVIDER: The LLM provider (should be 'openai' for this class)
        LLM_API_KEY: The API key for the LLM provider
        LLM_BASE_URL: Base URL for the API (defaults to https://api.openai.com/v1)
        LLM_MODEL: The model to use (defaults to gpt-4o)

        Legacy environment variables (deprecated, will be removed):
        OPENAI_API_KEY: The API key for OpenAI
        OPENAI_API_BASE: Base URL for the API
    """

    def __init__(self, model: str | None = None):
        # Get model from environment or use provided model or default
        # Support both LLM_MODEL and OPENAI_MODEL for backward compatibility
        env_model = os.getenv('LLM_MODEL') or os.getenv('OPENAI_MODEL')
        self.model = model or env_model or 'gpt-4o-mini'

        # Initialize client as None - will be set up in prepare()
        self._client: Client | None = None
        self._prepared: bool = False  # Track preparation state

        logger.info(f'Initialized OpenAI LLM with model {self.model}')

    async def prepare(self) -> None:
        """Prepare the LLM by setting up the HTTP client with proper configuration."""
        if self._prepared:
            return  # Already prepared, skip

        # Get API key from environment - support both LLM_API_KEY and OPENAI_API_KEY
        api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('LLM_API_KEY or OPENAI_API_KEY environment variable is required')

        # Get base URL from environment - support both LLM_BASE_URL and OPENAI_BASE_URL
        base_url = os.getenv('LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1'

        # Create client configuration
        config = ClientConfig(
            base_url=base_url,
            timeout=120.0,  # 2 minutes timeout for completion
            connect_timeout=30.0,  # 30 seconds to connect
            default_headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            max_retries=3,
            retry_min_wait=1.0,
            retry_max_wait=10.0,
            retry_on_status_codes=[429, 500, 502, 503, 504],
        )

        # Initialize and prepare the client
        self._client = Client(config)
        await self._client.prepare()

        # Mark as prepared
        self._prepared = True

        logger.info(f'Using OpenAI-compatible API at {base_url} with model {self.model}')

    @classmethod
    def supported_models(cls) -> list[str]:
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
        if not self._client:
            raise RuntimeError('OpenAI client not initialized. Call prepare() first.')

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

    async def _complete_response(self, payload: dict) -> str:
        """Handle a complete (non-streaming) response."""
        try:
            # Make API request
            response = await self._client.post('/chat/completions', json=payload)
            response.raise_for_status()

            # Parse response
            data = response.json()
            return data['choices'][0]['message']['content'] or ''

        except Exception as e:
            logger.error(f'OpenAI API request failed: {e}')
            raise

    async def _stream_response(self, payload: dict) -> _t.AsyncIterator[str]:
        """Handle a streaming response."""

        async def stream_generator():
            try:
                # Make streaming API request
                response = await self._client.post('/chat/completions', json=payload)
                response.raise_for_status()

                # Process SSE stream
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == '[DONE]':
                            break

                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue

            except Exception as e:
                logger.error(f'OpenAI streaming API request failed: {e}')
                raise

        return stream_generator()

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
        if not self._client:
            raise RuntimeError('OpenAI client not initialized. Call prepare() first.')

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
        try:
            # Make API request
            response = await self._client.post('/chat/completions', json=payload)
            response.raise_for_status()

            # Parse response
            data = response.json()
            message = data['choices'][0]['message']

            result = {}
            if message.get('content'):
                result['content'] = message['content']
            if message.get('tool_calls'):
                result['tool_calls'] = message['tool_calls']

            return result

        except Exception as e:
            logger.error(f'OpenAI tool API request failed: {e}')
            raise

    async def _stream_tool_response(self, payload: dict) -> _t.AsyncIterator[dict]:
        """Handle a streaming response with tool calls."""

        async def stream_generator():
            try:
                # Make streaming API request
                response = await self._client.post('/chat/completions', json=payload)
                response.raise_for_status()

                # For tool calls with streaming, we need to accumulate the response
                accumulated_content = ''
                accumulated_tool_calls = []

                # Process SSE stream
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == '[DONE]':
                            # Yield final accumulated result
                            result = {}
                            if accumulated_content:
                                result['content'] = accumulated_content
                            if accumulated_tool_calls:
                                result['tool_calls'] = accumulated_tool_calls
                            if result:
                                yield result
                            break

                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})

                                # Handle content
                                content = delta.get('content')
                                if content:
                                    accumulated_content += content
                                    yield {'content': content}

                                # Handle tool calls
                                tool_calls = delta.get('tool_calls')
                                if tool_calls:
                                    # This is a simplified handling - in practice, tool calls
                                    # in streaming are more complex with partial updates
                                    accumulated_tool_calls.extend(tool_calls)

                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue

            except Exception as e:
                logger.error(f'OpenAI streaming tool API request failed: {e}')
                raise

        return stream_generator()

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
