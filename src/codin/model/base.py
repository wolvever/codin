from __future__ import annotations

import abc
import typing as _t
from enum import Enum

__all__ = [
    "ModelType",
    "BaseModel",
    "BaseLLM",
    "BaseEmbedding",
    "BaseReranker",
]


class ModelType(str, Enum):
    """Type of model."""
    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


class BaseModel(abc.ABC):
    """Base class for all models."""

    model_type: ModelType

    def __init__(self, model: str):
        """Initialize the model and prepare it for use.
        
        Args:
            model: The model name or identifier
        """
        self.model = model
        # Subclasses should perform their preparation steps here

    @classmethod
    @abc.abstractmethod
    def supported_models(cls) -> list[str]:
        """Returns a list of regex patterns for supported model names."""
        pass


class BaseLLM(BaseModel):
    """Base class for LLM implementations."""

    model_type = ModelType.LLM

    @abc.abstractmethod
    async def generate(self, 
                      prompt: str | list[dict[str, str]], 
                      *, 
                      stream: bool = False,
                      temperature: float | None = None,
                      max_tokens: int | None = None,
                      stop_sequences: list[str] | None = None,
                      ) -> _t.AsyncIterator[str] | str:
        """Generate text from the model.
        
        Args:
            prompt: Either a string prompt or a list of messages with 'role' and 'content' keys
            stream: Whether to stream the response
            temperature: Temperature for sampling, higher is more random
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that will stop generation
            
        Returns:
            An async iterator of strings if stream=True, otherwise a complete string
        """
        pass

    @abc.abstractmethod
    async def generate_with_tools(self,
                                prompt: str | list[dict[str, str]],
                                tools: list[dict],
                                *,
                                stream: bool = False,
                                temperature: float | None = None,
                                max_tokens: int | None = None,
                                ) -> dict | _t.AsyncIterator[dict]:
        """Generate text with function/tool calling capabilities.
        
        Args:
            prompt: Either a string prompt or a list of messages
            tools: List of tool definitions in OpenAI format
            stream: Whether to stream the response
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with 'content' and/or 'tool_calls', or async iterator of such dicts
        """
        pass


class BaseEmbedding(BaseModel):
    """Base class for embedding models."""

    model_type = ModelType.EMBEDDING

    @abc.abstractmethod
    async def embed(self, 
                   texts: list[str],
                   ) -> list[list[float]]:
        """Embed texts into vectors.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embedding vectors
        """
        pass


class BaseReranker(BaseModel):
    """Base class for reranker models."""

    model_type = ModelType.RERANKER

    @abc.abstractmethod
    async def rerank(self,
                    query: str,
                    documents: list[str],
                    *,
                    top_k: int | None = None,
                    ) -> list[tuple[int, float]]:
        """Rerank documents given a query.
        
        Args:
            query: Query string
            documents: List of document strings to rerank
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document_index, score) in ranked order
        """
        pass 