"""Model factory for codin agents.

This module provides factory functions for creating language model instances
based on provider names and configuration settings.
"""

import logging
import os

from .anthropic_llm import AnthropicLLM
from .base import BaseLLM
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
    def create_llm(
        cls,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> BaseLLM:
        """Create an LLM instance based on configuration.

        Args:
            model: Model name (overrides LLM_MODEL env var)
            provider: Provider name (overrides LLM_PROVIDER env var)
            api_key: API key (overrides LLM_API_KEY env var)
            base_url: Base URL (overrides LLM_BASE_URL env var)

        Returns:
            A BaseLLM instance

        Raises:
            ValueError: If provider is not supported or auto-detection fails
        """
        # Get configuration from environment variables or parameters
        env_provider = os.environ.get('LLM_PROVIDER', 'auto').lower()
        env_model = os.environ.get('LLM_MODEL')
        env_api_key = os.environ.get('LLM_API_KEY')
        env_base_url = os.environ.get('LLM_BASE_URL')

        # Use parameters or fall back to environment variables
        final_provider = provider or env_provider
        final_model = model or env_model
        final_api_key = api_key or env_api_key
        final_base_url = base_url or env_base_url

        if not final_model:
            raise ValueError('Model name must be provided via parameter or LLM_MODEL environment variable')

        # Auto-detect provider if needed
        if final_provider == 'auto':
            final_provider = cls._detect_provider(final_model, final_base_url)
            logger.info(f"Auto-detected provider '{final_provider}' for model '{final_model}'")

        # Validate provider
        if final_provider not in cls.PROVIDER_CLASSES:
            supported = ', '.join(cls.PROVIDER_CLASSES.keys())
            raise ValueError(f"Unsupported provider '{final_provider}'. Supported: {supported}")

        # Set environment variables for the LLM class to use
        original_env = {}
        try:
            # Backup original environment
            env_vars_to_set = {
                'LLM_MODEL': final_model,
                'LLM_PROVIDER': final_provider,
            }
            if final_api_key:
                env_vars_to_set['LLM_API_KEY'] = final_api_key
            if final_base_url:
                env_vars_to_set['LLM_BASE_URL'] = final_base_url

            for key, value in env_vars_to_set.items():
                if key in os.environ:
                    original_env[key] = os.environ[key]
                os.environ[key] = value

            # Create the LLM instance
            llm_class = cls.PROVIDER_CLASSES[final_provider]
            instance = llm_class(model=final_model)

            logger.info(f"Created {final_provider} LLM instance for model '{final_model}'")
            if final_base_url:
                logger.info(f'Using custom base URL: {final_base_url}')

            return instance

        finally:
            # Restore original environment
            for key, value in env_vars_to_set.items():
                if key in original_env:
                    os.environ[key] = original_env[key]
                elif key in os.environ:
                    del os.environ[key]

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


def create_llm_from_env() -> BaseLLM:
    """Convenience function to create an LLM from environment variables only.

    Returns:
        A BaseLLM instance configured from environment variables
    """
    return LLMFactory.create_llm()
