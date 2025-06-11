"""Model factory for codin agents.

This module provides factory functions for creating language model instances
based on provider names and configuration settings.
"""

import logging
import os
import typing as _t

from .anthropic_llm import AnthropicLLM
from .base import BaseLLM
from .config import ModelConfig  # Import ModelConfig
from .gemini_llm import GeminiLLM
from .litellm_adapter import LiteLLMAdapter
from .openai_llm import OpenAILLM


__all__ = [
    'LLMFactory',
    'create_llm_from_env',
]

logger = logging.getLogger('codin.model.factory')


class LLMFactory:
    """Factory for creating LLM instances based on environment configuration.

    Environment variables:
        LLM_PROVIDER: The LLM provider to use (openai, anthropic, gemini, litellm, auto)
        LLM_MODEL: The model name to use
        LLM_API_KEY: API key for the provider
        LLM_BASE_URL: Base URL for the API (useful for proxy services)

    For auto-detection, the factory will try to determine the provider based on the model name.
    """

    # Provider mapping
    PROVIDER_CLASSES: dict[str, type[BaseLLM]] = {
        'openai': OpenAILLM,
        'anthropic': AnthropicLLM,
        'gemini': GeminiLLM,
        'litellm': LiteLLMAdapter,
    }

    @classmethod
    async def create_llm( # Changed to async
        cls,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        config: _t.Optional[ModelConfig] = None,
    ) -> BaseLLM:
        """Create an LLM instance based on configuration.

        The method prioritizes provided arguments, then values from the `config` object,
        and finally falls back to environment variables.

        Args:
            model: Model name (overrides LLM_MODEL env var and config.model_name)
            provider: Provider name (overrides LLM_PROVIDER env var and config.provider)
            api_key: API key (overrides LLM_API_KEY env var and config.api_key)
            base_url: Base URL (overrides LLM_BASE_URL env var and config.base_url)
            config: A ModelConfig object to source configuration from.

        Returns:
            A BaseLLM instance

        Raises:
            ValueError: If provider is not supported or auto-detection fails
        """
        # Create a base ModelConfig if none is provided
        effective_config = config or ModelConfig()

        # Determine final configuration values: direct args > config object > env vars
        final_provider = provider or \
                         effective_config.provider or \
                         os.environ.get('LLM_PROVIDER', 'auto').lower()

        final_model = model or \
                      effective_config.model_name or \
                      os.environ.get('LLM_MODEL')

        final_api_key = api_key or \
                        effective_config.api_key or \
                        os.environ.get('LLM_API_KEY')

        final_base_url = base_url or \
                         effective_config.base_url or \
                         os.environ.get('LLM_BASE_URL')

        # Other ModelConfig fields can be populated similarly if the factory needs to override them
        # For now, we assume other fields in `effective_config` are used as is.

        if not final_model:
            raise ValueError('Model name must be provided via parameter, ModelConfig, or LLM_MODEL env var')

        # Auto-detect provider if needed
        if final_provider == 'auto':
            final_provider = cls._detect_provider(final_model, final_base_url)
            logger.info(f"Auto-detected provider '{final_provider}' for model '{final_model}'")

        # Validate provider
        if final_provider not in cls.PROVIDER_CLASSES:
            supported_providers = ', '.join(cls.PROVIDER_CLASSES.keys())
            raise ValueError(f"Unsupported provider '{final_provider}'. Supported: {supported_providers}")

        # Prepare the ModelConfig to be passed to the LLM constructor
        # Prioritize specific settings passed to the factory, then existing config, then env.
        instance_config = ModelConfig(
            model_name=final_model, # Model name in config is for reference/consistency
            api_key=final_api_key,
            base_url=final_base_url,
            provider=final_provider,
            # Carry over other settings from the input config if they weren't overridden by direct args
            timeout=effective_config.timeout,
            connect_timeout=effective_config.connect_timeout,
            max_retries=effective_config.max_retries,
            retry_min_wait=effective_config.retry_min_wait,
            retry_max_wait=effective_config.retry_max_wait,
            retry_on_status_codes=effective_config.retry_on_status_codes,
            api_version=effective_config.api_version # Important for Anthropic
        )

        # Create the LLM instance
        llm_class = cls.PROVIDER_CLASSES[final_provider]
        # Pass the constructed config and the final_model directly.
        # The model classes' __init__ are designed to use the direct model first.
        # Model __init__ is now async
        instance = await llm_class(config=instance_config, model=final_model)

        logger.info(f"Created {final_provider} LLM instance for model '{final_model}'")
        if final_base_url: # Log if a non-default base_url was used
            logger.info(f'Using base URL: {final_base_url}')
        if final_api_key: # Log if an API key was explicitly sourced by the factory
             logger.debug(f'Using API key ending with: ...{final_api_key[-4:] if final_api_key else "None"}')

        return instance

    @classmethod
    def _detect_provider(cls, model: str, base_url: str | None = None) -> str:
        """Auto-detect the provider based on model name and base URL.

        Args:
            model: The model name
            base_url: The base URL (if any)

        Returns:
            The detected provider name

        Raises:
            ValueError: If provider cannot be detected
        """
        model_lower = model.lower()

        # If we have a custom base URL, assume it's an OpenAI-compatible proxy
        if base_url and base_url not in ['https://api.openai.com/v1', 'https://api.openai.com']:
            logger.debug(f'Custom base URL detected: {base_url}, using OpenAI provider')
            return 'openai'

        # Model name patterns for different providers
        if any(pattern in model_lower for pattern in ['gpt-', 'o1-', 'text-davinci', 'text-curie']):
            return 'openai'
        if any(pattern in model_lower for pattern in ['claude-', 'anthropic']):
            # If it's a Claude model but we have a custom base URL, use OpenAI (proxy)
            if base_url and base_url not in ['https://api.anthropic.com']:
                return 'openai'
            return 'anthropic'
        if any(pattern in model_lower for pattern in ['gemini-', 'bison', 'gecko']):
            return 'gemini'
        # For unknown models, try OpenAI first (most compatible)
        logger.warning(f"Unknown model '{model}', defaulting to OpenAI provider")
        return 'openai'


async def create_llm_from_env(config: _t.Optional[ModelConfig] = None) -> BaseLLM: # Changed to async
    """Convenience function to create an LLM.

    If a config object is provided, it's used as a base.
    Environment variables can override settings in the config unless specific parameters
    (model, api_key, base_url) are set directly in the config object, which then take precedence
    over environment variables.

    Args:
        config: Optional ModelConfig object.

    Returns:
        A BaseLLM instance.
    """
    # If config is None, LLMFactory.create_llm will create a new ModelConfig()
    # and rely on environment variables for api_key, base_url, model, provider.
    # If config is provided, those values are used as defaults, potentially overridden by env vars
    # if the corresponding fields in the config object are None.
    return await LLMFactory.create_llm(config=config) # Added await
