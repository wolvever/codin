"""LiteLLM adapter for codin agents.

This module provides an adapter that integrates LiteLLM for unified
access to multiple language model providers through a single interface.
"""

from __future__ import annotations

import asyncio
import logging
import os
import typing as _t


try:
    import litellm

    _HAS_LITELLM = True
except ImportError:
    _HAS_LITELLM = False

from .base import BaseLLM
from .registry import ModelRegistry


__all__ = [
    'LiteLLMAdapter',
]

logger = logging.getLogger('codin.model.litellm_adapter')


@ModelRegistry.register
class LiteLLMAdapter(BaseLLM):
    """Adapter for LiteLLM which supports numerous LLM providers.

    This adapter requires litellm to be installed: pip install litellm

    Environment variables:
        LITELLM_API_KEY: The API key for the specific provider (if not setting per provider)
        LITELLM_CACHE: Whether to enable LiteLLM caching (true/false)
    """

    def __init__(self, model: str):
        super().__init__(model)

        self._prepared = False

        # Check if litellm is installed
        if not _HAS_LITELLM:
            raise ImportError("LiteLLM is not installed. Install it with 'pip install litellm'.")

        # Configure litellm from environment variables if provided
        if os.environ.get('LITELLM_CACHE', '').lower() == 'true':
            litellm.cache = True

    @classmethod
    def supported_models(cls) -> list[str]:
        # Catch-all patterns for various provider prefixes
        # These will match models from these providers when used with litellm's prefix notation
        return [
            # Common model patterns with provider prefixes
            r'litellm/.*',  # Custom litellm prefix for any model
            r'openai/.*',  # OpenAI models
            r'anthropic/.*',  # Anthropic models
            r'google/.*',  # Google models
            r'mistral/.*',  # Mistral models
            r'cohere/.*',  # Cohere models
            r'azure/.*',  # Azure models
            r'together/.*',  # Together AI models
            r'perplexity/.*',  # Perplexity models
            r'groq/.*',  # Groq models
            r'anyscale/.*',  # Anyscale models
            r'ollama/.*',  # Ollama models
        ]

    async def prepare(self) -> None:
        """Prepare the LiteLLM adapter for use."""
        # Nothing to prepare as LiteLLM handles connections per request
        self._prepared = True

    def _prepare_messages(self, prompt: str | list[dict[str, str]]) -> list[dict[str, str]]:
        """Convert prompt to LiteLLM message format."""
        if isinstance(prompt, str):
            return [{'role': 'user', 'content': prompt}]
        return prompt

    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> _t.AsyncIterator[str] | str:
        """Generate text using LiteLLM."""
        messages = self._prepare_messages(prompt)

        # Prepare kwargs
        completion_kwargs = {
            'model': self.model,
            'messages': messages,
            'stream': stream,
        }

        if temperature is not None:
            completion_kwargs['temperature'] = temperature

        if max_tokens is not None:
            completion_kwargs['max_tokens'] = max_tokens

        if stop_sequences:
            completion_kwargs['stop'] = stop_sequences

        # Execute in a thread to avoid blocking
        loop = asyncio.get_event_loop()

        if not stream:
            # Non-streaming response
            completion = await loop.run_in_executor(None, lambda: litellm.completion(**completion_kwargs))
            return completion.choices[0].message.content or ''
        # Streaming response - we'll use an async generator
        async def stream_generator() -> _t.AsyncIterator[str]:
            # Start the stream in a separate thread
            stream_response = await loop.run_in_executor(None, lambda: litellm.completion(**completion_kwargs))

            # Iterate through chunks
            for chunk in stream_response:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

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
        """Generate text with function/tool calling capabilities."""
        messages = self._prepare_messages(prompt)

        # Prepare kwargs
        completion_kwargs = {
            'model': self.model,
            'messages': messages,
            'tools': tools,
            'stream': stream,
        }

        if temperature is not None:
            completion_kwargs['temperature'] = temperature

        if max_tokens is not None:
            completion_kwargs['max_tokens'] = max_tokens

        # Execute in a thread to avoid blocking
        loop = asyncio.get_event_loop()

        if not stream:
            # Non-streaming response
            completion = await loop.run_in_executor(None, lambda: litellm.completion(**completion_kwargs))

            message = completion.choices[0].message
            result = {}

            if message.content:
                result['content'] = message.content

            if hasattr(message, 'tool_calls') and message.tool_calls:
                result['tool_calls'] = message.tool_calls

            return result
        # Streaming response with tools
        async def stream_tool_results() -> _t.AsyncIterator[dict]:
            # Use litellm's implementation of streaming
            stream_response = await loop.run_in_executor(None, lambda: litellm.completion(**completion_kwargs))

            for chunk in stream_response:
                # Handle content chunks
                if (
                    hasattr(chunk.choices[0], 'delta')
                    and hasattr(chunk.choices[0].delta, 'content')
                    and chunk.choices[0].delta.content
                ):
                    yield {'content': chunk.choices[0].delta.content}

                # Handle tool call chunks
                if (
                    hasattr(chunk.choices[0], 'delta')
                    and hasattr(chunk.choices[0].delta, 'tool_calls')
                    and chunk.choices[0].delta.tool_calls
                ):
                    yield {'tool_calls': chunk.choices[0].delta.tool_calls}

        return stream_tool_results()
