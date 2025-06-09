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
from .config import ModelConfig
from .registry import register # Changed

__all__ = [
    'LiteLLMAdapter',
]

logger = logging.getLogger('codin.model.litellm_adapter')


@register # Changed
class LiteLLMAdapter(BaseLLM):
    """Adapter for LiteLLM which supports numerous LLM providers.

    This adapter requires litellm to be installed: pip install litellm

    Environment variables:
        LITELLM_API_KEY: The API key for the specific provider (if not setting per provider)
        LITELLM_CACHE: Whether to enable LiteLLM caching (true/false)
    """

    def __init__(self, model: str, config: _t.Optional[ModelConfig] = None):
        """Initialize the LiteLLM Adapter.

        Args:
            model: The model string that LiteLLM will use (e.g., "gpt-3.5-turbo", "claude-2", "gemini/gemini-pro").
                   This often includes the provider prefix if not using a globally set model.
            config: Optional ModelConfig instance. LiteLLM primarily uses environment variables
                    for provider-specific API keys and base URLs. However, common parameters like
                    timeout or max_retries from the config might be applicable to litellm.completion calls.
        """
        super().__init__(model)
        self.config = config or ModelConfig()
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

        # Populate from ModelConfig if available and applicable to litellm.completion
        client_settings = self.config.get_client_config_kwargs()
        if client_settings.get('timeout') is not None: # Check if timeout is set in config
            completion_kwargs['request_timeout'] = client_settings['timeout'] # LiteLLM uses request_timeout
        if client_settings.get('max_retries') is not None: # Check if max_retries is set
            completion_kwargs['num_retries'] = client_settings['max_retries'] # LiteLLM uses num_retries

        # Allow overriding api_key and base_url from config if provided
        # LiteLLM gives precedence to kwargs over environment variables
        if self.config.api_key:
            completion_kwargs['api_key'] = self.config.api_key
        if self.config.base_url:
            completion_kwargs['base_url'] = self.config.base_url
        # We don't need to pass 'api_version' as LiteLLM handles this based on model string or other env vars.

        logger.debug(
            f"Calling litellm.completion for model '{self.model}' (stream={stream}). "
            f"Options: {{k:v for k,v in completion_kwargs.items() if k not in ['messages', 'model', 'stream']}}"
        )

        if not stream:
            # Non-streaming response
            try:
                completion = await loop.run_in_executor(None, lambda: litellm.completion(**completion_kwargs))
                logger.debug(f"LiteLLM non-streaming response received for model '{self.model}'.")
                # Ensure choice and message exist before accessing content
                if completion.choices and completion.choices[0].message:
                    return completion.choices[0].message.content or ''
                logger.warning(f"LiteLLM response for model '{self.model}' had no choices or message content.")
                return '' # Return empty string if no valid content
            except Exception as e:
                logger.error(f"LiteLLM non-streaming completion for model '{self.model}' failed: {e}")
                raise # Re-raise the exception to allow higher-level handling

        # Streaming response
        async def stream_generator() -> _t.AsyncIterator[str]:
            try:
                stream_response = await loop.run_in_executor(None, lambda: litellm.completion(**completion_kwargs))
                logger.debug(f"LiteLLM streaming response initiated for model '{self.model}'.")
                for chunk in stream_response:
                    if (hasattr(chunk.choices[0], 'delta') and
                        hasattr(chunk.choices[0].delta, 'content') and
                        chunk.choices[0].delta.content is not None): # Ensure content is not None
                        yield chunk.choices[0].delta.content
            except Exception as e:
                logger.error(f"LiteLLM streaming completion for model '{self.model}' failed: {e}")
                raise # Re-raise the exception

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

        # Populate from ModelConfig if available
        client_settings = self.config.get_client_config_kwargs()
        if client_settings.get('timeout') is not None:
            completion_kwargs['request_timeout'] = client_settings['timeout']
        if client_settings.get('max_retries') is not None:
            completion_kwargs['num_retries'] = client_settings['max_retries']
        if self.config.api_key:
            completion_kwargs['api_key'] = self.config.api_key
        if self.config.base_url:
            completion_kwargs['base_url'] = self.config.base_url

        logger.debug(
            f"Calling litellm.completion_with_tools for model '{self.model}' (stream={stream}). "
            f"Options: {{k:v for k,v in completion_kwargs.items() if k not in ['messages', 'tools', 'model', 'stream']}}"
        )

        if not stream:
            # Non-streaming response
            try:
                completion = await loop.run_in_executor(None, lambda: litellm.completion(**completion_kwargs))
                logger.debug(f"LiteLLM non-streaming tool response received for model '{self.model}'.")

                result = {}
                if completion.choices and completion.choices[0].message:
                    message = completion.choices[0].message
                    if message.content:
                        result['content'] = message.content

                    # LiteLLM tool_calls are objects, convert to dict if necessary for consistency
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        raw_tool_calls = message.tool_calls
                        # Ensure tool_calls are in the expected list-of-dicts format
                        result['tool_calls'] = [
                            {'id': tc.id, 'type': tc.type,
                             'function': {'name': tc.function.name, 'arguments': tc.function.arguments}}
                            for tc in raw_tool_calls if tc.id and tc.function and tc.function.name and tc.function.arguments is not None
                        ]
                else:
                    logger.warning(f"LiteLLM tool response for model '{self.model}' had no choices or message.")

                return result
            except Exception as e:
                logger.error(f"LiteLLM non-streaming tool completion for model '{self.model}' failed: {e}")
                raise

        # Streaming response with tools
        async def stream_tool_results() -> _t.AsyncIterator[dict]:
            try:
                stream_response = await loop.run_in_executor(None, lambda: litellm.completion(**completion_kwargs))
                logger.debug(f"LiteLLM streaming tool response initiated for model '{self.model}'.")
                for chunk in stream_response:
                    if not (chunk.choices and chunk.choices[0]):
                        continue # Skip empty choices

                    delta = chunk.choices[0].delta
                    if not delta:
                        continue

                    # Handle content chunks
                    if hasattr(delta, 'content') and delta.content is not None:
                        yield {'content': delta.content}

                    # Handle tool call chunks
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        # Assuming delta.tool_calls is a list of tool call delta objects.
                        # Each object might have index, id, type, function (name, arguments).
                        # Convert to a consistent dict structure.
                        formatted_tool_calls = []
                        for tc_delta in delta.tool_calls:
                            if tc_delta and tc_delta.function: # Ensure function part exists
                                formatted_tool_calls.append({
                                    'id': tc_delta.id, # ID might be None in deltas until full
                                    'type': tc_delta.type or 'function', # type might be None
                                    'function': {
                                        'name': tc_delta.function.name,
                                        'arguments': tc_delta.function.arguments
                                    }
                                })
                        if formatted_tool_calls:
                             yield {'tool_calls': formatted_tool_calls}
            except Exception as e:
                logger.error(f"LiteLLM streaming tool completion for model '{self.model}' failed: {e}")
                raise

        return stream_tool_results()
