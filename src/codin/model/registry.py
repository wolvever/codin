"""Model registry for codin agents.

This module provides a registry system for managing and discovering
language model implementations across different providers.
It supports both a default global registry and instance-based registries.
"""

from __future__ import annotations

import functools
import logging
import re
import typing as _t
import asyncio
import importlib # For dynamic class loading

from .base import BaseEmbedding, BaseLLM, BaseModel, BaseReranker, ModelType
from .config import ModelConfig


__all__ = [
    'ModelRegistry',
    'default_model_registry', # Expose the default instance
    # Expose module-level functions that use the default registry
    'register',
    'resolve_model_class',
    'create_llm',
    'create_embedding',
    'create_reranker',
    'list_supported_models',
    'add_registration_to_default_registry',
]

logger = logging.getLogger('codin.model.registry')

def get_class_from_fqn(fqn: str) -> type[BaseModel]:
    """Imports and returns a class from its fully qualified name."""
    try:
        module_name, class_name = fqn.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if not issubclass(cls, BaseModel):
            raise TypeError(f"Class {fqn} is not a subclass of BaseModel.")
        return cls
    except (ImportError, AttributeError, ValueError) as e:
        logger.error(f"Failed to import class {fqn}: {e}")
        raise ImportError(f"Could not import class {fqn}: {e}") from e


class ModelRegistry:
    """Registry for models (LLMs, embeddings, rerankers).

    Allows registering and instantiating models by name with pattern matching.
    Instances of this class hold their own set of model registrations.
    """
    def __init__(self):
        """Initialize a new model registry instance."""
        self._model_registry: dict[str, tuple[ModelType, type[BaseModel]]] = {}
        logger.debug("Initialized new ModelRegistry instance.")

    def register(self, model_cls: type[BaseModel]) -> type[BaseModel]:
        """Register a model class with this registry instance.

        Can be used as a decorator on a class:
        my_registry = ModelRegistry()
        @my_registry.register
        class MyModel(BaseLLM):
            ...

        Args:
            model_cls: The model class to register.

        Returns:
            The same model class (for use as a decorator).
        """
        if not hasattr(model_cls, 'supported_models') or not callable(model_cls.supported_models):
            raise TypeError(f"Model class {model_cls.__name__} must have a 'supported_models' class method.")
        if not hasattr(model_cls, 'model_type') or not isinstance(model_cls.model_type, ModelType):
            raise TypeError(f"Model class {model_cls.__name__} must have a 'model_type' class attribute of type ModelType.")

        for pattern in model_cls.supported_models():
            if pattern in self._model_registry:
                existing_type, existing_cls = self._model_registry[pattern]
                logger.warning(
                    f"Registry {id(self)}: Replacing model for pattern '{pattern}': {existing_cls.__name__} → {model_cls.__name__}"
                )
            self._model_registry[pattern] = (model_cls.model_type, model_cls)
            logger.debug(f"Registry {id(self)}: Registered model {model_cls.__name__} for pattern '{pattern}'.")
        return model_cls

    def add_registration(self, pattern: str, model_cls_fqn: str, model_type_str: str) -> None:
        """
        Programmatically add a model registration from its fully qualified name.

        Args:
            pattern: The regex pattern for model names.
            model_cls_fqn: Fully qualified name of the model class (e.g., "src.codin.model.openai_llm.OpenAILLM").
            model_type_str: String representation of the model type (e.g., "llm", "embedding").

        Raises:
            ImportError: If the class cannot be imported.
            TypeError: If the imported class is not a BaseModel or model_type_str is invalid.
        """
        logger.info(f"Registry {id(self)}: Attempting to add registration: pattern='{pattern}', fqn='{model_cls_fqn}', type='{model_type_str}'.")
        try:
            model_cls = get_class_from_fqn(model_cls_fqn)
            model_type_enum = ModelType(model_type_str.lower())

            # Basic validation, register method does more detailed checks
            if model_cls.model_type != model_type_enum:
                 logger.warning(
                    f"Registry {id(self)}: Mismatch between provided model_type_str ('{model_type_str}') "
                    f"and class's model_type ('{model_cls.model_type.value}') for {model_cls_fqn}. "
                    f"Using class's model_type: {model_cls.model_type.value}."
                )

            # Use the class's own register method to add itself with all its supported patterns
            # This seems more robust than directly manipulating _model_registry here with just one pattern.
            # However, the original intent might be to register a single pattern to a class.
            # For now, let's register the specific pattern to the class and its declared type.
            # The class's own `supported_models` might include this pattern or others.
            # This ensures the specific pattern is mapped.
            if pattern in self._model_registry:
                 existing_type, existing_cls = self._model_registry[pattern]
                 logger.warning(
                    f"Registry {id(self)}: Replacing model for pattern '{pattern}' via add_registration: {existing_cls.__name__} → {model_cls.__name__}"
                )
            self._model_registry[pattern] = (model_cls.model_type, model_cls) # Use class's model_type
            logger.info(f"Registry {id(self)}: Successfully added registration for pattern '{pattern}' to {model_cls.__name__}.")

        except (ImportError, TypeError, ValueError) as e: # ValueError for invalid ModelType string
            logger.error(f"Registry {id(self)}: Failed to add registration for FQN {model_cls_fqn} with pattern '{pattern}': {e}")
            raise

    @functools.lru_cache(maxsize=32) # Keep cache on instance method
    def resolve_model_class(self, model_name: str, model_type: ModelType | None = None) -> type[BaseModel]:
        """Resolve a model name to its implementing class from this registry instance."""
        candidates = []
        for pattern, (registry_type, model_cls) in self._model_registry.items():
            if model_type is not None and registry_type != model_type:
                continue
            if re.fullmatch(pattern, model_name):
                candidates.append((pattern, model_cls))

        if not candidates:
            type_str = f' of type {model_type.value}' if model_type else ''
            raise ValueError(f"Registry {id(self)}: No model found for '{model_name}'{type_str}")

        if len(candidates) > 1:
            candidates.sort(key=lambda x: len(x[0]), reverse=True)
            logger.debug(f'Registry {id(self)}: Multiple matches for {model_name}: {candidates}')
        return candidates[0][1]

    async def create_llm(self, model_name: str, config: _t.Optional[ModelConfig] = None) -> BaseLLM:
        """Create a new LLM instance from this registry. (Now async)"""
        model_cls = self.resolve_model_class(model_name, ModelType.LLM)
        return _t.cast(BaseLLM, await model_cls(model=model_name, config=config))

    async def create_embedding(self, model_name: str, config: _t.Optional[ModelConfig] = None) -> BaseEmbedding:
        """Create a new embedding model instance from this registry. (Now async)"""
        model_cls = self.resolve_model_class(model_name, ModelType.EMBEDDING)
        return _t.cast(BaseEmbedding, await model_cls(model=model_name, config=config))

    def create_reranker(self, model_name: str, config: _t.Optional[ModelConfig] = None) -> BaseReranker:
        """Create a new reranker model instance from this registry. (Remains synchronous)"""
        model_cls = self.resolve_model_class(model_name, ModelType.RERANKER)
        if hasattr(model_cls, '__init__') and 'config' in model_cls.__init__.__code__.co_varnames:
            return _t.cast(BaseReranker, model_cls(model=model_name, config=config))
        else:
            logger.warning(f"Registry {id(self)}: Reranker {model_cls.__name__} does not accept ModelConfig. Instantiating without it.")
            return _t.cast(BaseReranker, model_cls(model=model_name)) # type: ignore

    def list_supported_models(self, model_type: ModelType | None = None) -> list[str]:
        """List all supported model patterns in this registry instance."""
        if model_type is None:
            return list(self._model_registry.keys())
        return [pattern for pattern, (registry_type, _) in self._model_registry.items() if registry_type == model_type]

# --- Module-level default registry and convenience functions ---
default_model_registry = ModelRegistry()

def register(model_cls: type[BaseModel]) -> type[BaseModel]:
    """Registers a model class with the default global model registry. Intended for decorator use."""
    return default_model_registry.register(model_cls)

def add_registration_to_default_registry(pattern: str, model_cls_fqn: str, model_type_str: str) -> None:
    """Programmatically add a model registration to the default global registry."""
    default_model_registry.add_registration(pattern, model_cls_fqn, model_type_str)

def resolve_model_class(model_name: str, model_type: ModelType | None = None) -> type[BaseModel]:
    """Resolves a model class from the default global model registry."""
    return default_model_registry.resolve_model_class(model_name, model_type)

async def create_llm(model_name: str, config: _t.Optional[ModelConfig] = None) -> BaseLLM:
    """Creates an LLM instance from the default global model registry. (async)"""
    return await default_model_registry.create_llm(model_name, config)

async def create_embedding(model_name: str, config: _t.Optional[ModelConfig] = None) -> BaseEmbedding:
    """Creates an embedding model instance from the default global model registry. (async)"""
    return await default_model_registry.create_embedding(model_name, config)

def create_reranker(model_name: str, config: _t.Optional[ModelConfig] = None) -> BaseReranker:
    """Creates a reranker instance from the default global model registry. (sync)"""
    return default_model_registry.create_reranker(model_name, config)

def list_supported_models(model_type: ModelType | None = None) -> list[str]:
    """Lists supported model patterns from the default global model registry."""
    return default_model_registry.list_supported_models(model_type)

# Example of dynamic registration:
# add_registration_to_default_registry("my-custom-gpt", "src.codin.model.openai_llm.OpenAILLM", "llm")
# my_instance = asyncio.run(create_llm("my-custom-gpt"))
