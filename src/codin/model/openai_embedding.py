"""OpenAI embedding model integration for codin agents.

This module provides integration with OpenAI's embedding models
for text vectorization and similarity search capabilities.
"""

import asyncio
import logging
import os

from ..client import Client, ClientConfig, LoggingTracer
from .base import BaseEmbedding
from .registry import ModelRegistry

__all__ = [
    'OpenAIEmbedding',
]

logger = logging.getLogger('codin.model.openai_embedding')


@ModelRegistry.register
class OpenAIEmbedding(BaseEmbedding):
    """Implementation of BaseEmbedding for OpenAI API and compatible services.

    Environment variables:
        OPENAI_API_KEY: The API key for OpenAI
        OPENAI_API_BASE: Base URL for the API (defaults to https://api.openai.com/v1)
    """

    def __init__(self, model: str):
        super().__init__(model)

        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.api_base = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')

        # Remove trailing slash if present
        if self.api_base.endswith('/'):
            self.api_base = self.api_base[:-1]

        self._client: Client | None = None

    @classmethod
    def supported_models(cls) -> list[str]:
        """Supported models for OpenAI embedding."""
        return [
            r'text-embedding-.*',
            r'text-embedding-ada-.*',
        ]

    async def prepare(self) -> None:
        """Prepare the OpenAI embedding client."""  
        if not self.api_key:
            raise ValueError('OPENAI_API_KEY environment variable not set')

        if self._client is None:
            # Configure the HTTP client
            config = ClientConfig(
                base_url=self.api_base,
                default_headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                },
                timeout=30.0,
                # Add tracing in debug mode
                tracers=[LoggingTracer()] if logger.isEnabledFor(logging.DEBUG) else [],
            )
            self._client = Client(config)
            await self._client.prepare()

    async def _ensure_client(self) -> Client:
        """Ensure the HTTP client is prepared and return it."""
        await self.prepare()
        if not self._client:
            raise RuntimeError('Failed to initialize OpenAI embedding client')
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts."""
        client = await self._ensure_client()

        # Prepare request payload
        payload = {
            'model': self.model,
            'input': texts,
        }

        # Send request
        response = await client.post('/embeddings', json=payload)
        response.raise_for_status()

        data = response.json()

        # Sort by index to ensure embeddings are returned in the same order as input
        sorted_data = sorted(data['data'], key=lambda x: x['index'])
        embeddings = [item['embedding'] for item in sorted_data]

        return embeddings

    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None

    def __del__(self):
        """Ensure client is closed on garbage collection."""
        if self._client:
            asyncio.create_task(self.close())
