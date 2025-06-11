"""
Client for interacting with vLLM instances that expose an OpenAI-compatible API.
"""
from __future__ import annotations

import logging
import os
import typing as _t

from .openai_compatible_llm import OpenAICompatibleBaseLLM
from .config import ModelConfig
from .registry import register # Changed

logger = logging.getLogger(__name__)

@register # Changed
class VLLMClient(OpenAICompatibleBaseLLM):
    """
    Client for vLLM instances providing an OpenAI-compatible API.

    This client assumes the vLLM server is configured to expose endpoints like
    `/v1/chat/completions`. The 'model' parameter in requests will typically
    refer to the model identifier used by the vLLM server (often the path of
    the loaded model).
    """

    # Default values for VLLM client
    # Model name is often the path to the model on the server, so a generic default might not be useful.
    # Users should typically specify this in config or via environment variable.
    DEFAULT_MODEL = None # Or a placeholder like "default_vllm_model_path"
    DEFAULT_BASE_URL = 'http://localhost:8000/v1' # Common default for local vLLM

    # Environment variable names specific to VLLM configuration, falling back to generic ones.
    # VLLM typically doesn't use an API key for local deployments.
    # The OpenAICompatibleBaseLLM already makes API_KEY optional.
    API_KEY_ENV_VAR = 'VLLM_API_KEY' # If vLLM is configured with one (less common for local)
    BASE_URL_ENV_VAR = 'VLLM_BASE_URL'
    MODEL_ENV_VAR = 'VLLM_MODEL' # Specific model identifier for vLLM server

    async def __init__(self, config: _t.Optional[ModelConfig] = None, model: str | None = None):
        """
        Initialize the VLLMClient.

        Args:
            config: ModelConfig instance. `base_url` is crucial for pointing to the vLLM server.
                    `model_name` should be the identifier the vLLM server uses for the desired model.
                    `api_key` is typically not required for local vLLM.
            model: Optional model name/path to override what's in config or environment variables.
        """
        final_config = config or ModelConfig()

        # Determine model name: direct arg > config.model_name > VLLM_MODEL > LLM_MODEL > self.DEFAULT_MODEL
        chosen_model = model
        if chosen_model is None and final_config.model_name:
            chosen_model = final_config.model_name
        if chosen_model is None:
            chosen_model = os.getenv(self.MODEL_ENV_VAR) or \
                           os.getenv(self.LLM_MODEL_ENV_VAR) # General fallback

        # If no model is specified through any means, and DEFAULT_MODEL is None,
        # OpenAICompatibleBaseLLM might raise an error if its DEFAULT_MODEL is also None,
        # or use its own default. For vLLM, model is often crucial.
        if chosen_model is None and self.DEFAULT_MODEL is not None:
            final_model_name = self.DEFAULT_MODEL
        elif chosen_model is not None:
            final_model_name = chosen_model
        else:
            # Let super().__init__ handle it, it might use its own default or raise error
            # if no model name can be resolved and its DEFAULT_MODEL is also None.
            # Or, raise a more specific error here for vLLM.
            # For vLLM, the model path/name is critical.
            raise ValueError(
                "Model name/path for VLLM must be specified via 'model' argument, "
                "ModelConfig.model_name, or VLLM_MODEL/LLM_MODEL environment variable."
            )

        # Base URL determination specific for VLLMClient before passing to super's init logic
        # The base class's __init__ also does this, but we ensure VLLM_BASE_URL has priority here
        # if it's not already in final_config.base_url
        if final_config.base_url is None: # Only if not already set in the passed config object
            final_config.base_url = os.getenv(self.BASE_URL_ENV_VAR) or \
                                    os.getenv(self.LLM_BASE_URL_ENV_VAR) # General fallback
            if final_config.base_url is None: # If still none, use VLLMClient's default
                 final_config.base_url = self.DEFAULT_BASE_URL

        # API key is optional for vLLM, OpenAICompatibleBaseLLM handles api_key=None
        if final_config.api_key is None: # Only if not already set
            # VLLM typically doesn't use API keys, but check env just in case.
            # If VLLM_API_KEY is set to an empty string, it means "no key".
            # The base class will try LLM_API_KEY and its own API_KEY_ENV_VAR if this is None.
            # To ensure no key is sent if not specified for VLLM:
            api_key_from_env = os.getenv(self.API_KEY_ENV_VAR) # VLLM_API_KEY
            if api_key_from_env is not None: # If VLLM_API_KEY is set (even if empty)
                final_config.api_key = api_key_from_env if api_key_from_env else "" # Use empty string for "no key"
            elif os.getenv(self.LLM_API_KEY_ENV_VAR) is not None: # If generic LLM_API_KEY is set
                 final_config.api_key = os.getenv(self.LLM_API_KEY_ENV_VAR)
            else:
                final_config.api_key = "" # Default to no API key for vLLM if no specific env found

        await super().__init__(config=final_config, model=final_model_name)
        logger.info(f"VLLMClient initialized for model '{self.model}' server at '{self.config.base_url}'")

    @classmethod
    def supported_models(cls) -> list[str]:
        """
        vLLM can serve arbitrary models. The model name used in API calls
        is typically the path or identifier of the model on the vLLM server.
        A generic pattern allows flexibility.
        """
        return [r'.*'] # Matches any model string
