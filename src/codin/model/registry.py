"""Model registry for automatic model resolution and instantiation.

This module provides a registry system that allows automatic resolution
of model names to their implementing classes using pattern matching.
It supports LLMs, embedding models, and rerankers.
"""

from __future__ import annotations

import functools
import logging
import re
import typing as _t

from .base import BaseModel, BaseLLM, BaseEmbedding, BaseReranker, ModelType

__all__ = [
    "ModelRegistry",
]

logger = logging.getLogger("codin.model.registry")

# Dict mapping model name regex patterns to model classes
_model_registry: dict[str, tuple[ModelType, type[BaseModel]]] = {}


class ModelRegistry:
    """Registry for models (LLMs, embeddings, rerankers).
    
    Allows registering and instantiating models by name with pattern matching.
    """
    
    @staticmethod
    def register(model_cls: type[BaseModel]) -> type[BaseModel]:
        """Register a model class with the registry.
        
        Can be used as a decorator:
        @ModelRegistry.register
        class MyModel(BaseLLM):
            ...
            
        Args:
            model_cls: The model class to register
            
        Returns:
            The same model class (for use as a decorator)
        """
        for pattern in model_cls.supported_models():
            if pattern in _model_registry:
                existing_type, existing_cls = _model_registry[pattern]
                logger.warning(
                    f"Replacing model for pattern '{pattern}': "
                    f"{existing_cls.__name__} → {model_cls.__name__}"
                )
            
            _model_registry[pattern] = (model_cls.model_type, model_cls)
        
        return model_cls
    
    @staticmethod
    @functools.lru_cache(maxsize=32)
    def resolve_model_class(model_name: str, model_type: ModelType | None = None) -> type[BaseModel]:
        """Resolve a model name to its implementing class.
        
        Args:
            model_name: The name of the model
            model_type: Optional type filter (LLM, embedding, reranker)
            
        Returns:
            The model class
            
        Raises:
            ValueError: If no matching model is found
        """
        candidates = []
        
        for pattern, (registry_type, model_cls) in _model_registry.items():
            if model_type is not None and registry_type != model_type:
                continue
                
            if re.fullmatch(pattern, model_name):
                candidates.append((pattern, model_cls))
        
        if not candidates:
            type_str = f" of type {model_type.value}" if model_type else ""
            raise ValueError(f"No model found for '{model_name}'{type_str}")
        
        # If multiple matches, sort by pattern length (more specific first)
        if len(candidates) > 1:
            candidates.sort(key=lambda x: len(x[0]), reverse=True)
            logger.debug(f"Multiple matches for {model_name}: {candidates}")
        
        return candidates[0][1]
    
    @staticmethod
    def create_llm(model_name: str) -> BaseLLM:
        """Create a new LLM instance.
        
        Args:
            model_name: The name of the LLM model
            
        Returns:
            A BaseLLM instance
        """
        model_cls = ModelRegistry.resolve_model_class(model_name, ModelType.LLM)
        return _t.cast(BaseLLM, model_cls(model=model_name))
    
    @staticmethod
    def create_embedding(model_name: str) -> BaseEmbedding:
        """Create a new embedding model instance.
        
        Args:
            model_name: The name of the embedding model
            
        Returns:
            A BaseEmbedding instance
        """
        model_cls = ModelRegistry.resolve_model_class(model_name, ModelType.EMBEDDING)
        return _t.cast(BaseEmbedding, model_cls(model=model_name))
    
    @staticmethod
    def create_reranker(model_name: str) -> BaseReranker:
        """Create a new reranker model instance.
        
        Args:
            model_name: The name of the reranker model
            
        Returns:
            A BaseReranker instance
        """
        model_cls = ModelRegistry.resolve_model_class(model_name, ModelType.RERANKER)
        return _t.cast(BaseReranker, model_cls(model=model_name))
    
    @staticmethod
    def list_supported_models(model_type: ModelType | None = None) -> list[str]:
        """List all supported model patterns.
        
        Args:
            model_type: Optional filter by model type
            
        Returns:
            List of regex patterns for supported models
        """
        if model_type is None:
            return list(_model_registry.keys())
        
        return [
            pattern for pattern, (registry_type, _) in _model_registry.items()
            if registry_type == model_type
        ] 