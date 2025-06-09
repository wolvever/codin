"""OpenAI embedding model integration for codin agents.

This module provides integration with OpenAI's embedding models
for text vectorization and similarity search capabilities.
"""

import asyncio
import logging
import os
import typing as _t # Ensure _t is imported if not already for ModelConfig hint

from ..client import Client, ClientConfig, LoggingTracer
from .base import BaseEmbedding
from .config import ModelConfig
from .registry import register # Changed
from .http_utils import make_post_request, ContentExtractionError, ModelResponseParsingError, StreamProcessingError

__all__ = [
    'OpenAIEmbedding',
]

logger = logging.getLogger('codin.model.openai_embedding')


@register # Changed
class OpenAIEmbedding(BaseEmbedding):
    """Implementation of BaseEmbedding for OpenAI API and compatible services.

    Environment variables:
        OPENAI_API_KEY: The API key for OpenAI
        OPENAI_API_BASE: Base URL for the API (defaults to https://api.openai.com/v1)
    """

    DEFAULT_BASE_URL = 'https://api.openai.com/v1'
    DEFAULT_TIMEOUT = 30.0
    # Note: OpenAIEmbedding often uses a specific model like 'text-embedding-ada-002'
    # The `model` parameter in __init__ is usually provided directly.

    async def __init__(self, model: str, config: _t.Optional[ModelConfig] = None): # Changed to async
        """Initialize and prepare the OpenAI Embedding model.

        Args:
            model: The specific embedding model name (e.g., "text-embedding-ada-002").
            config: Optional ModelConfig instance. If None, a default config is used,
                    and settings are primarily sourced from environment variables.
        """
        super().__init__(model)
        self.config = config or ModelConfig()

        # Client initialization logic moved from prepare()
        # API Key: config > env (specific OPENAI_API_KEY)
        api_key = self.config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                'API key not found. Set in ModelConfig or OPENAI_API_KEY environment variable.'
            )

        # Base URL: config > env (specific OPENAI_API_BASE) > default
        base_url = self.config.base_url or \
                   os.getenv('OPENAI_API_BASE') or \
                   self.DEFAULT_BASE_URL
        if base_url.endswith('/'): # Ensure no trailing slash
            base_url = base_url[:-1]

        client_kwargs = self.config.get_client_config_kwargs()
        client_kwargs.setdefault('timeout', self.DEFAULT_TIMEOUT)
        client_kwargs['default_headers'] = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        client_kwargs['base_url'] = base_url

        if logger.isEnabledFor(logging.DEBUG) and 'tracers' not in client_kwargs:
            client_kwargs['tracers'] = [LoggingTracer()]
        elif 'tracers' not in client_kwargs:
             client_kwargs['tracers'] = []

        client_config_obj = ClientConfig(**client_kwargs)
        self._client = Client(client_config_obj)
        # await self._client.prepare() # Removed, Client.__init__ now handles full setup

        logger.info(f'OpenAI Embedding client initialized and prepared for model {self.model} at {base_url}')

    # prepare() method is now removed

    @classmethod
    def supported_models(cls) -> list[str]:
        """Supported models for OpenAI embedding."""
        return [
            r'text-embedding-.*',
            r'text-embedding-ada-.*',
        ]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts."""
        if not self._client: # Should not happen if __init__ completes successfully
            raise RuntimeError('OpenAI Embedding client not initialized. This should not happen after async __init__.')

        # Prepare request payload
        payload = {
            'model': self.model,
            'input': texts,
        }

        try:
            response = await make_post_request(
                self._client,
                '/embeddings',
                payload,
                error_message_prefix=f"OpenAI Embedding API request for model {self.model} failed"
            )
            response_data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from OpenAI Embedding for model {self.model}: {e}. Response text: {response.text}")
            raise ModelResponseParsingError(f"Failed to decode JSON response for model {self.model}: {e}") from e
        except Exception as e: # Catch other errors from make_post_request (like HTTPStatusError)
            logger.error(f"OpenAI Embedding request for model {self.model} failed: {e}")
            # Re-raise directly as make_post_request already logs and handles retries/status errors
            raise

        try:
            embedding_data = response_data.get('data')
            if not isinstance(embedding_data, list):
                logger.error(f"OpenAI Embedding response for model {self.model} missing 'data' list or not a list.")
                logger.debug(f"Problematic embedding data for model {self.model}: {response_data}")
                raise ModelResponseParsingError(f"Invalid response format from model {self.model}: 'data' field missing or not a list.")

            # Sort by index to ensure embeddings are returned in the same order as input
            # Use .get('embedding') for safer access, though it should ideally be present.
            embeddings = []
            for item in sorted(embedding_data, key=lambda x: x.get('index', -1)):
                if isinstance(item, dict) and 'embedding' in item:
                    embeddings.append(item['embedding'])
                else:
                    logger.warning(f"Skipping item in embedding response for model {self.model} due to missing 'embedding' key or wrong type: {item}")

            if len(embeddings) != len(texts):
                logger.warning(
                    f"Number of embeddings received ({len(embeddings)}) from model {self.model} "
                    f"does not match number of input texts ({len(texts)})."
                )

            return embeddings
        except (KeyError, TypeError, IndexError) as e_parse: # More specific parsing errors
            logger.error(f"OpenAI Embedding response parsing error for model {self.model}: {type(e_parse).__name__} - {e_parse}.")
            logger.debug(f"Problematic data for model {self.model}: {response_data}")
            raise ModelResponseParsingError(f"Invalid response format from model {self.model}, parsing error: {e_parse}") from e_parse
        except Exception as e_other: # Catch any other unexpected error during processing
            logger.error(f"Unexpected error processing OpenAI Embedding response for model {self.model}: {e_other}.", exc_info=True)
            logger.debug(f"Problematic data for model {self.model}: {response_data}")
            raise ContentExtractionError(f"Unexpected error processing embeddings for model {self.model}: {e_other}") from e_other


    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None

    def __del__(self):
        """Ensure client is closed on garbage collection."""
        if self._client:
            logger.warning('OpenAIEmbedding was deleted without calling close(). This may leak resources.')
