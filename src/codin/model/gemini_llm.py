"""Google Gemini LLM integration for codin agents.

This module provides integration with Google's Gemini language models
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
    'GeminiLLM',
]

logger = logging.getLogger('codin.model.gemini_llm')


@register # Changed
class GeminiLLM(BaseLLM):
    """Implementation of BaseLLM for Google's Gemini API.

    Supports both streaming and non-streaming generation.

    Environment variables:
        LLM_PROVIDER: The LLM provider (should be 'gemini' for this class)
        LLM_API_KEY: The API key for the LLM provider
        LLM_BASE_URL: Base URL for the API (defaults to https://generativelanguage.googleapis.com)
        LLM_MODEL: The model to use (defaults to gemini-1.5-pro)

        Legacy environment variables (deprecated, will be removed):
        GOOGLE_API_KEY: The API key for Google AI
        GOOGLE_API_BASE: Base URL for the API
    """
    DEFAULT_MODEL = 'gemini-1.5-pro'
    DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com'
    DEFAULT_TIMEOUT = 60.0

    async def __init__(self, config: _t.Optional[ModelConfig] = None, model: str | None = None): # Changed to async
        """Initialize and prepare the Gemini LLM.

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
            chosen_model = os.getenv('LLM_MODEL') # Generic env var

        self.model = chosen_model or self.DEFAULT_MODEL
        super().__init__(self.model)

        # Client initialization logic moved from prepare()
        # API Key: config > env (generic) > env (specific)
        self._resolved_api_key = self.config.api_key or \
                                 os.getenv('LLM_API_KEY') or \
                                 os.getenv('GOOGLE_API_KEY')
        if not self._resolved_api_key:
            raise ValueError(
                'API key not found. Set in ModelConfig, LLM_API_KEY, or GOOGLE_API_KEY environment variable.'
            )

        # Base URL: config > env (generic) > env (specific) > default
        base_url = self.config.base_url or \
                   os.getenv('LLM_BASE_URL') or \
                   os.getenv('GOOGLE_API_BASE') or \
                   self.DEFAULT_BASE_URL
        if base_url.endswith('/'): # Ensure no trailing slash
            base_url = base_url[:-1]

        client_kwargs = self.config.get_client_config_kwargs()
        client_kwargs.setdefault('timeout', self.DEFAULT_TIMEOUT)
        client_kwargs['default_headers'] = {'Content-Type': 'application/json'}
        client_kwargs['base_url'] = base_url

        if logger.isEnabledFor(logging.DEBUG) and 'tracers' not in client_kwargs:
            client_kwargs['tracers'] = [LoggingTracer()]
        elif 'tracers' not in client_kwargs:
             client_kwargs['tracers'] = []

        client_config = ClientConfig(**client_kwargs)
        self._client = Client(client_config)
        # await self._client.prepare() # Removed, Client.__init__ now handles full setup

        logger.info(f'Gemini LLM initialized and client prepared for model {self.model} at {base_url}')

    # prepare() method is now removed.

    @classmethod
    def supported_models(cls) -> list[str]:
        """Supported models for Gemini LLM."""
        return [
            r'gemini-.*',
        ]

    def _prepare_messages(self, prompt: str | list[dict[str, str]]) -> list[dict]:
        """Convert prompt to Gemini message format.

        Args:
            prompt: Either a string or a list of message dictionaries

        Returns:
            List of messages in Gemini format
        """
        if isinstance(prompt, str):
            return [{'role': 'user', 'parts': [{'text': prompt}]}]

        # Map OpenAI roles to Gemini roles
        gemini_messages = []

        for msg in prompt:
            role = msg['role']
            content = msg['content']

            if role == 'system':
                # Gemini doesn't have system messages, prepend to first user message
                if gemini_messages and gemini_messages[0]['role'] == 'user':
                    gemini_messages[0]['parts'].insert(0, {'text': f'System: {content}\n\n'})
                else:
                    gemini_messages.append({'role': 'user', 'parts': [{'text': f'System: {content}\n\n'}]})
            elif role == 'user':
                gemini_messages.append({'role': 'user', 'parts': [{'text': content}]})
            elif role == 'assistant':
                gemini_messages.append({'role': 'model', 'parts': [{'text': content}]})
            else:
                logger.warning(f'Unsupported role: {role}, treating as user')
                gemini_messages.append({'role': 'user', 'parts': [{'text': content}]})

        return gemini_messages

    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> _t.AsyncIterator[str] | str:
        """Generate text using Gemini API."""
        if not self._client: # Should not happen if __init__ completes successfully
            raise RuntimeError('Gemini client not initialized. This should not happen after async __init__.')
        # client variable removed, self._client used directly

        if not self._resolved_api_key: # Should be set by __init__
            raise RuntimeError("Gemini API key not resolved. This should not happen after async __init__.")

        # Convert prompt to Gemini format
        messages = self._prepare_messages(prompt)

        # Prepare request payload
        payload = {'contents': messages, 'generationConfig': {}}

        # Add optional parameters
        if temperature is not None:
            payload['generationConfig']['temperature'] = temperature

        if max_tokens is not None:
            payload['generationConfig']['maxOutputTokens'] = max_tokens

        if stop_sequences:
            payload['generationConfig']['stopSequences'] = stop_sequences

        # Add API key to URL
        endpoint = f'/v1beta/models/{self.model}:generateContent?key={self._resolved_api_key}'

        # Add streaming parameter to URL if needed
        if stream:
            endpoint += '&alt=sse'
            return self._stream_response(endpoint, payload) # Use self._client internally
        return await self._complete_response(endpoint, payload) # Use self._client internally

    def _extract_content_from_response(self, response_data: dict) -> _t.Optional[str]:
        """Helper to extract content from Gemini's non-streaming response JSON."""
        candidates = response_data.get('candidates')
        if not candidates or not isinstance(candidates, list) or len(candidates) == 0:
            return None # Let extract_content_from_json handle if this means error

        candidate = candidates[0]
        if not isinstance(candidate, dict): return None

        content = candidate.get('content')
        if not content or not isinstance(content, dict):
            return None

        parts = content.get('parts')
        if not parts or not isinstance(parts, list):
            return None

        text_parts = []
        for part in parts:
            if isinstance(part, dict) and 'text' in part:
                text_parts.append(part['text'])
        # If no text parts found, join will be empty string, which is fine for "no content".
        # If content is expected, extract_content_from_json will raise if this returns "" and it expected something.
        # However, it's better to return None if truly no text parts, to let caller decide if "" or error.
        return "".join(text_parts) if text_parts else None

    def _extract_delta_from_stream_chunk(self, data_chunk: dict) -> _t.Optional[str]:
        """Helper to extract content delta from Gemini's streaming response chunk."""
        candidates = data_chunk.get('candidates')
        if candidates and isinstance(candidates, list) and len(candidates) > 0:
            candidate = candidates[0]
            if not isinstance(candidate, dict): return None

            content = candidate.get('content')
            if content and isinstance(content, dict):
                parts = content.get('parts')
                if parts and isinstance(parts, list) and len(parts) > 0:
                    part = parts[0]
                    if isinstance(part, dict) and 'text' in part:
                        return part['text']
        return None

    async def _complete_response(self, endpoint: str, payload: dict) -> str:
        """Handle a complete (non-streaming) response."""
        if not self._client:
            raise RuntimeError(f"Gemini client for model {self.model} not initialized in _complete_response.")

        response = await make_post_request(
            self._client,
            endpoint,
            payload,
            error_message_prefix=f"Gemini API request for model {self.model} failed"
        )
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from Gemini for model {self.model}: {e}. Response text: {response.text}")
            raise StreamProcessingError(f"Failed to decode JSON response: {e}") from e # Or ModelResponseParsingError


        try:
            return extract_content_from_json(
                response_data,
                self._extract_content_from_response, # Use helper
                error_message_prefix=f"Gemini content extraction for model {self.model} failed"
            )
        except ContentExtractionError as e:
            logger.error(f"Gemini content extraction error for model {self.model}: {e}")
            raise


    async def _stream_response(self, endpoint: str, payload: dict) -> _t.AsyncIterator[str]:
        """Handle a streaming response."""
        if not self._client:
            raise RuntimeError(f"Gemini client for model {self.model} not initialized in _stream_response.")

        response = await make_post_request(
            self._client,
            endpoint,
            payload,
            error_message_prefix=f"Gemini streaming API request for model {self.model} failed"
        )

        try:
            return process_sse_stream(
                response,
                delta_extractor=self._extract_delta_from_stream_chunk, # Use helper
                stop_marker=None, # Gemini doesn't use a specific data line like "[DONE]"
                error_message_prefix=f"Gemini streaming processing for model {self.model} failed"
            )
        except StreamProcessingError as e:
            logger.error(f"Gemini stream processing error for model {self.model}: {e}")
            raise

    async def generate_with_tools(
                stop_marker=None, # Gemini doesn't use a specific data line like "[DONE]"
                error_message_prefix="Gemini streaming processing failed"
            )
        except StreamProcessingError as e:
            logger.error(f"Gemini stream processing error: {e}")
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

        Note: This is a placeholder implementation as Gemini's tool calling
        API might differ from OpenAI's. This needs to be updated when Gemini's
        function calling API is fully documented.
        """
        raise NotImplementedError('Tool calling is not yet implemented for Gemini')

    async def close(self):
        """Close the client and release resources."""
        if self._client:
            await self._client.close()
            self._client = None

    def __del__(self):
        """Cleanup on deletion."""
        if self._client:
            logger.warning('GeminiLLM was deleted without calling close(). This may leak resources.')
