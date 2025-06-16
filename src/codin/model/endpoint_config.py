"""Model configuration using unified endpoint system."""

import os
import typing as _t
from dataclasses import dataclass, field

from codin.endpoint import EndpointConfig

from .config import ModelConfig


@dataclass
class ModelEndpointConfig:
    """Model configuration with unified endpoint support."""
    
    # Model identification
    model_name: _t.Optional[str] = None
    provider: _t.Optional[str] = None
    
    # Endpoint configuration
    endpoint_config: _t.Optional[EndpointConfig] = None
    
    # Authentication
    api_key: _t.Optional[str] = None
    api_version: _t.Optional[str] = None
    
    # Request settings
    timeout: _t.Optional[float] = None
    connect_timeout: _t.Optional[float] = None
    max_retries: _t.Optional[int] = None
    retry_min_wait: _t.Optional[float] = None
    retry_max_wait: _t.Optional[float] = None
    retry_on_status_codes: _t.Optional[list[int]] = None
    
    def __post_init__(self):
        """Initialize endpoint configuration if not provided."""
        if self.endpoint_config is None:
            # Try to create from environment or use defaults
            endpoint_str = os.getenv('CODIN_MODEL_ENDPOINT')
            if endpoint_str:
                self.endpoint_config = EndpointConfig(endpoint=endpoint_str)
            else:
                # Create based on provider or use default
                if self.provider == 'anthropic':
                    default_endpoint = "https://api.anthropic.com/v1"
                elif self.provider == 'gemini':
                    default_endpoint = "https://generativelanguage.googleapis.com/v1"
                else:
                    # Default to OpenAI
                    default_endpoint = "https://api.openai.com/v1"
                
                self.endpoint_config = EndpointConfig(endpoint=default_endpoint)
    
    @property
    def base_url(self) -> _t.Optional[str]:
        """Get base URL from endpoint configuration."""
        if self.endpoint_config and self.endpoint_config.is_remote:
            return self.endpoint_config.base_url
        return None
    
    @property
    def is_local_model(self) -> bool:
        """Check if this is a local model endpoint."""
        return self.endpoint_config and self.endpoint_config.is_local
    
    @property  
    def is_remote_model(self) -> bool:
        """Check if this is a remote model endpoint."""
        return self.endpoint_config and self.endpoint_config.is_remote
    
    @property
    def local_model_path(self) -> _t.Optional[str]:
        """Get local model path for filesystem endpoints."""
        if self.is_local_model:
            return self.endpoint_config.local_path
        return None
    
    def to_model_config(self) -> ModelConfig:
        """Convert to legacy ModelConfig for backward compatibility."""
        return ModelConfig(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            api_version=self.api_version,
            timeout=self.timeout,
            connect_timeout=self.connect_timeout,
            max_retries=self.max_retries,
            retry_min_wait=self.retry_min_wait,
            retry_max_wait=self.retry_max_wait,
            retry_on_status_codes=self.retry_on_status_codes,
            provider=self.provider,
        )
    
    @classmethod
    def from_model_config(cls, config: ModelConfig) -> "ModelEndpointConfig":
        """Create from legacy ModelConfig."""
        endpoint_config = None
        if config.base_url:
            endpoint_config = EndpointConfig(endpoint=config.base_url)
        
        return cls(
            model_name=config.model_name,
            provider=config.provider,
            endpoint_config=endpoint_config,
            api_key=config.api_key,
            api_version=config.api_version,
            timeout=config.timeout,
            connect_timeout=config.connect_timeout,
            max_retries=config.max_retries,
            retry_min_wait=config.retry_min_wait,
            retry_max_wait=config.retry_max_wait,
            retry_on_status_codes=config.retry_on_status_codes,
        )
    
    @classmethod
    def from_env(cls, provider: _t.Optional[str] = None) -> "ModelEndpointConfig":
        """Create configuration from environment variables."""
        # Get model configuration from env
        model_name = os.getenv('LLM_MODEL')
        api_key = os.getenv('LLM_API_KEY')
        provider = provider or os.getenv('LLM_PROVIDER', 'auto').lower()
        
        # Create endpoint config from CODIN_MODEL_ENDPOINT or fallback
        endpoint_str = os.getenv('CODIN_MODEL_ENDPOINT')
        if not endpoint_str:
            # Fallback to LLM_BASE_URL for backward compatibility
            base_url = os.getenv('LLM_BASE_URL')
            if base_url:
                endpoint_str = base_url
        
        endpoint_config = None
        if endpoint_str:
            endpoint_config = EndpointConfig(endpoint=endpoint_str)
        
        return cls(
            model_name=model_name,
            provider=provider,
            endpoint_config=endpoint_config,
            api_key=api_key,
        )
    
    @classmethod
    def openai(
        cls, 
        model: str = "gpt-4", 
        api_key: _t.Optional[str] = None,
        base_url: _t.Optional[str] = None
    ) -> "ModelEndpointConfig":
        """Create OpenAI model configuration."""
        endpoint_url = base_url or "https://api.openai.com/v1"
        endpoint_config = EndpointConfig(endpoint=endpoint_url)
        
        return cls(
            model_name=model,
            provider="openai",
            endpoint_config=endpoint_config,
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
        )
    
    @classmethod
    def anthropic(
        cls,
        model: str = "claude-3-sonnet-20240229",
        api_key: _t.Optional[str] = None,
        base_url: _t.Optional[str] = None
    ) -> "ModelEndpointConfig":
        """Create Anthropic model configuration."""
        endpoint_url = base_url or "https://api.anthropic.com/v1"
        endpoint_config = EndpointConfig(endpoint=endpoint_url)
        
        return cls(
            model_name=model,
            provider="anthropic", 
            endpoint_config=endpoint_config,
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY'),
        )
    
    @classmethod
    def local_model(
        cls,
        model_path: str,
        model_name: _t.Optional[str] = None
    ) -> "ModelEndpointConfig":
        """Create local model configuration."""
        endpoint_config = EndpointConfig(endpoint=f"fs://{model_path}")
        
        return cls(
            model_name=model_name or os.path.basename(model_path),
            provider="local",
            endpoint_config=endpoint_config,
        )