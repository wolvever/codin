"""Language model integrations for codin agents.

This module provides LLM integrations for various providers including
OpenAI, Anthropic, Gemini, and other language model services.

Model abstractions and implementations for different LLM providers.
"""

from .anthropic_llm import AnthropicLLM
from .base import BaseEmbedding, BaseLLM, BaseModel, BaseReranker, ModelType
from .config import ModelConfig
from .factory import LLMFactory, create_llm_from_env
from .gemini_llm import GeminiLLM
from .http_utils import ModelAPIError, ContentExtractionError, StreamProcessingError, ModelResponseParsingError
from .litellm_adapter import LiteLLMAdapter
from .openai_compatible_llm import OpenAICompatibleBaseLLM
from .openai_embedding import OpenAIEmbedding
from .openai_llm import OpenAILLM
from .registry import ModelRegistry
from .vllm_client import VLLMClient
from .facade_client import ModelFacade # Added


__all__ = [
    # Base classes
    'ModelType',
    'BaseModel',
    'BaseLLM',
    'OpenAICompatibleBaseLLM', # Added
    'BaseEmbedding',
    'BaseReranker',
    # Config
    'ModelConfig',
    # Registry
    'ModelRegistry',
    # Factory
    'LLMFactory',
    'create_llm_from_env',
    # LLM implementations
    'OpenAILLM',
    'AnthropicLLM',
    'GeminiLLM',
    'LiteLLMAdapter',
    'VLLMClient',
    # Facade
    'ModelFacade', # Added
    # Embedding implementations
    'OpenAIEmbedding',
    # Exceptions
    'ModelAPIError',          # Added
    'ContentExtractionError',
    'StreamProcessingError',
    'ModelResponseParsingError',# Added
]
