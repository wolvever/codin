"""Base configuration for all Codin components using unified endpoint pattern."""

import os
from abc import ABC, abstractmethod

from pydantic import BaseModel

from .config import EndpointConfig


class ComponentConfig(BaseModel, ABC):
    """Base configuration class for all Codin components."""
    
    endpoint: EndpointConfig
    name: str | None = None
    enabled: bool = True
    
    @classmethod
    @abstractmethod
    def default_endpoint(cls) -> str:
        """Return the default endpoint for this component."""
        pass
    
    @classmethod
    @abstractmethod  
    def env_var_name(cls) -> str:
        """Return the environment variable name for this component."""
        pass
    
    @classmethod
    def from_env(cls, **kwargs) -> "ComponentConfig":
        """Create configuration from environment variables."""
        env_var = cls.env_var_name()
        endpoint_str = os.getenv(env_var, cls.default_endpoint())
        fallback_str = os.getenv(f"{env_var}_FALLBACK")
        
        endpoint_config = EndpointConfig(
            endpoint=endpoint_str,
            fallback_endpoint=fallback_str
        )
        
        return cls(endpoint=endpoint_config, **kwargs)
    
    @classmethod
    def local(cls, path: str, **kwargs) -> "ComponentConfig":
        """Create configuration for local filesystem."""
        endpoint_config = EndpointConfig(endpoint=f"fs://{path}")
        return cls(endpoint=endpoint_config, **kwargs)
    
    @classmethod
    def remote(cls, url: str, **kwargs) -> "ComponentConfig":
        """Create configuration for remote service."""
        endpoint_config = EndpointConfig(endpoint=url)
        return cls(endpoint=endpoint_config, **kwargs)


class MemoryConfig(ComponentConfig):
    """Configuration for Memory service."""
    
    @classmethod
    def default_endpoint(cls) -> str:
        return "fs://./data/memory"
    
    @classmethod
    def env_var_name(cls) -> str:
        return "CODIN_MEMORY_ENDPOINT"


class StorageConfig(ComponentConfig):
    """Configuration for Storage service."""
    
    @classmethod
    def default_endpoint(cls) -> str:
        return "fs://./prompt_templates"
    
    @classmethod
    def env_var_name(cls) -> str:
        return "CODIN_PROMPT_ENDPOINT"


class RegistryConfig(ComponentConfig):
    """Configuration for Registry service."""
    
    @classmethod
    def default_endpoint(cls) -> str:
        return "fs://./tools"
    
    @classmethod
    def env_var_name(cls) -> str:
        return "CODIN_TOOL_ENDPOINT"


class ReplayConfig(ComponentConfig):
    """Configuration for Replay service."""
    
    @classmethod
    def default_endpoint(cls) -> str:
        return "fs://./data/replay"
    
    @classmethod
    def env_var_name(cls) -> str:
        return "CODIN_REPLAY_ENDPOINT"


class ModelConfig(ComponentConfig):
    """Configuration for Model service."""
    
    model_name: str | None = None
    api_key: str | None = None
    provider: str | None = None
    
    @classmethod
    def default_endpoint(cls) -> str:
        return "https://api.openai.com/v1"
    
    @classmethod
    def env_var_name(cls) -> str:
        return "CODIN_MODEL_ENDPOINT"
    
    @classmethod
    def from_env(cls, **kwargs) -> "ModelConfig":
        """Create model configuration from environment variables."""
        config = super().from_env(**kwargs)
        
        # Add model-specific environment variables
        config.model_name = kwargs.get('model_name') or os.getenv('LLM_MODEL')
        config.api_key = kwargs.get('api_key') or os.getenv('LLM_API_KEY')
        config.provider = kwargs.get('provider') or os.getenv('LLM_PROVIDER', 'auto')
        
        return config