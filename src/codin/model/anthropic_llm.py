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
from .registry import ModelRegistry

__all__ = [
    'AnthropicLLM',
]

logger = logging.getLogger('codin.model.anthropic_llm')


@ModelRegistry.register
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

    def __init__(self, model: str | None = None):
        # Get model from environment or use provided model or default
        env_model = os.environ.get('LLM_MODEL')
        model_name = model or env_model or 'claude-3-sonnet-20240229'

        super().__init__(model_name)

        # Try new environment variables first, fall back to legacy ones
        self.api_key = os.environ.get('LLM_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
        self.api_base = os.environ.get('LLM_BASE_URL') or os.environ.get(
            'ANTHROPIC_API_BASE', 'https://api.anthropic.com'
        )
        self.api_version = os.environ.get('ANTHROPIC_API_VERSION', '2023-06-01')

        # Remove trailing slash if present
        if self.api_base.endswith('/'):
            self.api_base = self.api_base[:-1]

        self._client: Client | None = None

        # Initialize the client
        if not self.api_key:
            raise ValueError('LLM_API_KEY or ANTHROPIC_API_KEY environment variable not set')

        # Configure the HTTP client
        config = ClientConfig(
            base_url=self.api_base,
            default_headers={
                'x-api-key': self.api_key,
                'anthropic-version': self.api_version,
                'Content-Type': 'application/json',
            },
            timeout=60.0,
            # Add tracing in debug mode
            tracers=[LoggingTracer()] if logger.isEnabledFor(logging.DEBUG) else [],
        )
        self._client = Client(config)

        logger.info(f'Using Anthropic API at {self.api_base} with model {self.model}')

    @classmethod
    def supported_models(cls) -> list[str]:
        """Supported models for Anthropic LLM."""
        return [
            r'claude-.*',
        ]

    async def _ensure_client(self) -> Client:
        """Ensure the HTTP client is prepared and return it."""
        if not self._client:
            raise RuntimeError('Failed to initialize Anthropic client')

        # Make sure the client is prepared
        await self._client.prepare()
        return self._client

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
        client = await self._ensure_client()

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
            return self._stream_response(client, payload)
        return await self._complete_response(client, payload)

    async def _complete_response(self, client: Client, payload: dict) -> str:
        """Handle a complete (non-streaming) response."""
        response = await client.post('/v1/messages', json=payload)
        response.raise_for_status()

        data = response.json()

        # Extract content from response
        try:
            content = data.get('content', [])
            if not content:
                return ''

            # Join all text blocks
            text = ''
            for block in content:
                if block.get('type') == 'text':
                    text += block.get('text', '')

            return text
        except (KeyError, IndexError) as e:
            logger.error(f'Error parsing Anthropic response: {e}')
            logger.debug(f'Response data: {data}')
            raise ValueError(
                f'Failed to parse Anthropic response: {e}'
            ) from e

    async def _stream_response(self, client: Client, payload: dict) -> _t.AsyncIterator[str]:
        """Handle a streaming response."""

        async def stream_generator():
            response = await client.post('/v1/messages', json=payload)
            response.raise_for_status()

            text_buffer = ''

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line == 'data: [DONE]':
                    continue

                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])

                        # Check if this is a content block
                        if data.get('type') == 'content_block_delta':
                            delta = data.get('delta', {})
                            if delta.get('type') == 'text_delta':
                                text = delta.get('text', '')
                                if text:
                                    text_buffer += text
                                    yield text

                        # Check if this is a message stop
                        elif data.get('type') == 'message_stop':
                            break

                    except json.JSONDecodeError:
                        logger.warning(f'Failed to parse SSE line: {line}')

        return stream_generator()

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
        """Clean up resources when the object is garbage collected."""
        if self._client:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._client.close())
            except Exception:
                pass
