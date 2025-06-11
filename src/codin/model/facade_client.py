"""
Provides a unified ModelFacade for easy interaction with LLM, Embedding,
and potentially Reranker models.
"""
from __future__ import annotations

import logging
import typing as _t

from .base import BaseLLM, BaseEmbedding, BaseReranker
from .config import ModelConfig
from .factory import LLMFactory
# Import module-level functions from registry to use default_model_registry
from .registry import (
    create_embedding as global_create_embedding,
    ModelRegistry # Keep for type hinting if a custom registry is passed
)
# Specific model classes that might be instantiated by default
from .openai_embedding import OpenAIEmbedding

logger = logging.getLogger(__name__)

class ModelFacade:
    """
    A facade providing a simplified interface to interact with underlying
    LLM, Embedding, and Reranker models.
    """

    DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002" # A common default

    def __init__(self,
                 llm: BaseLLM,
                 embedding: _t.Optional[BaseEmbedding] = None,
                 reranker: _t.Optional[BaseReranker] = None):
        """
        Private constructor. Use `create` classmethod for instantiation.
        """
        self._llm: BaseLLM = llm
        self._embedding: _t.Optional[BaseEmbedding] = embedding
        self._reranker: _t.Optional[BaseReranker] = reranker
        logger.info(f"ModelFacade initialized with LLM: {llm.model}")
        if embedding:
            logger.info(f"Embedding client configured: {embedding.model}")
        if reranker:
            logger.info(f"Reranker client configured: {reranker.model}")

    @classmethod
    async def create(cls,
                     provider: str | None = None,
                     model_name: str | None = None,
                     config: ModelConfig | None = None,
                     llm_factory: _t.Optional[LLMFactory] = None,
                     model_registry: _t.Optional[ModelRegistry] = None,
                     embedding_model_name: str | None = None, # Allow explicit embedding model
                     # reranker_model_name: str | None = None, # For future
                    ) -> ModelFacade:
        """
        Asynchronously creates and initializes a ModelFacade instance.

        Args:
            provider: The primary LLM provider name.
            model_name: The primary LLM model name.
            config: A ModelConfig object to be used for all clients. If specific fields
                    (like model_name for embedding) are different, they can be overridden.
            llm_factory: Optional LLMFactory instance for testing/mocking.
            model_registry: Optional ModelRegistry instance for testing/mocking.
            embedding_model_name: Optional specific model name for the embedding client.

        Returns:
            An initialized ModelFacade instance.
        """
        factory = llm_factory or LLMFactory()
        # Use the provided model_registry instance, or default to using module-level functions
        # which operate on default_model_registry.
        # No need to instantiate ModelRegistry() here as that would be an empty registry.

        effective_config = config or ModelConfig()

        # 1. Instantiate the primary LLM
        # LLMFactory.create_llm is now async
        primary_llm = await factory.create_llm(
            model=model_name,
            provider=provider,
            config=effective_config
        )
        logger.debug(f"Primary LLM '{primary_llm.model}' (provider: {primary_llm.config.provider}) instantiated for ModelFacade.")

        # 2. Instantiate Embedding Model (conditionally)
        embedding_client: _t.Optional[BaseEmbedding] = None

        # Determine the provider of the instantiated LLM
        llm_provider = primary_llm.config.provider
        if not llm_provider and provider: # If factory didn't set it but it was passed
            llm_provider = provider.lower()
        elif not llm_provider and model_name: # Try to infer from model_name if factory didn't set it
            try: # This is a bit heuristic, factory._detect_provider is internal
                if "claude" in model_name.lower() or "anthropic" in model_name.lower(): llm_provider = "anthropic"
                elif "gemini" in model_name.lower() or "google" in model_name.lower(): llm_provider = "gemini"
                elif "gpt-" in model_name.lower() or "openai" in model_name.lower(): llm_provider = "openai"
            except Exception:
                pass # Keep llm_provider as None

        # Default to OpenAI embedding if LLM is OpenAI, or if explicitly requested
        # and no other embedding model is specified.
        final_embedding_model_name = embedding_model_name
        if final_embedding_model_name is None and llm_provider == "openai":
            final_embedding_model_name = cls.DEFAULT_OPENAI_EMBEDDING_MODEL
            logger.info(f"Defaulting to OpenAI embedding model: {final_embedding_model_name} for provider {llm_provider}")

        if final_embedding_model_name:
            try:
                # Use a copy of the config, but ensure the model_name is for the embedding model
                embedding_config = effective_config.model_copy(deep=True) if effective_config else ModelConfig()
                embedding_config.model_name = final_embedding_model_name
                # Provider for embedding might be different, or derived from model name.
                # For now, assume create_embedding can handle it or uses a default.
                # If embedding_model_name includes provider (e.g. "openai/text-embedding-ada-002"), registry will find it.

                # ModelRegistry.create_embedding is now async
                # Use the passed model_registry instance or the global module-level function
                if model_registry:
                    embedding_client = await model_registry.create_embedding(
                        model_name=final_embedding_model_name,
                        config=embedding_config
                    )
                else:
                    embedding_client = await global_create_embedding( # Use module-level function
                        model_name=final_embedding_model_name,
                        config=embedding_config
                    )
                logger.debug(f"Embedding client '{embedding_client.model}' instantiated for ModelFacade.")
            except Exception as e:
                logger.warning(f"Could not instantiate embedding model '{final_embedding_model_name}': {e}. Embedding will be unavailable.")
                embedding_client = None

        # 3. Instantiate Reranker Model (placeholder for now)
        reranker_client: _t.Optional[BaseReranker] = None
        # if reranker_model_name:
        #     try:
        #         # ... similar logic for reranker ...
        #     except Exception as e:
        # logger.warning(f"Could not instantiate reranker model ...")
        # reranker_client = None

        return cls(llm=primary_llm, embedding=embedding_client, reranker=reranker_client)

    async def generate(self, prompt: str | list[dict[str, str]], **kwargs) -> str | _t.AsyncIterator[str]:
        """Generates text using the configured LLM."""
        if self._llm is None: # Should not happen if create() was used and succeeded for LLM
            raise RuntimeError("LLM client not configured in ModelFacade.")
        # Assuming kwargs are passed directly to the underlying LLM's generate method
        return await self._llm.generate(prompt=prompt, **kwargs)

    async def generate_with_tools(self, prompt: str | list[dict[str, str]], tools: list[dict], **kwargs) -> dict | _t.AsyncIterator[dict]:
        """Generates text with tool calling capabilities using the configured LLM."""
        if self._llm is None:
            raise RuntimeError("LLM client not configured in ModelFacade.")
        return await self._llm.generate_with_tools(prompt=prompt, tools=tools, **kwargs)

    async def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Embeds texts using the configured embedding model."""
        if self._embedding is None:
            raise RuntimeError("Embedding client not configured or failed to initialize in ModelFacade.")
        return await self._embedding.embed(texts=texts, **kwargs)

    async def rerank(self, query: str, documents: list[str], **kwargs) -> list[tuple[int, float]]:
        """Reranks documents using the configured reranker model."""
        if self._reranker is None:
            raise RuntimeError("Reranker client not configured or failed to initialize in ModelFacade.")
        return await self._reranker.rerank(query=query, documents=documents, **kwargs)

    async def close(self) -> None:
        """Closes all underlying clients."""
        if self._llm and hasattr(self._llm, 'close'):
            await self._llm.close()
            logger.debug("Primary LLM client closed.")
        if self._embedding and hasattr(self._embedding, 'close'):
            await self._embedding.close()
            logger.debug("Embedding client closed.")
        if self._reranker and hasattr(self._reranker, 'close'):
            await self._reranker.close() # pragma: no cover (as reranker is not implemented yet)
            logger.debug("Reranker client closed.")

    async def __aenter__(self) -> ModelFacade:
        # Initialization is now in create(), __aenter__ just returns self
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # Convenience properties if direct access is sometimes needed (use with caution)
    @property
    def llm(self) -> BaseLLM:
        if self._llm is None: raise RuntimeError("LLM client not configured.")
        return self._llm

    @property
    def embedding_model(self) -> BaseEmbedding:
        if self._embedding is None: raise RuntimeError("Embedding client not configured.")
        return self._embedding

    @property
    def reranker_model(self) -> BaseReranker: # pragma: no cover
        if self._reranker is None: raise RuntimeError("Reranker client not configured.")
        return self._reranker
