"""Unified endpoint configuration for all Codin components."""
import os
from urllib.parse import ParseResult, urlparse

from pydantic import BaseModel, validator


class EndpointConfig(BaseModel):
    """Unified endpoint configuration supporting fs:// and http:// schemes."""
    
    endpoint: str
    fallback_endpoint: str | None = None
    auth: dict | None = None
    timeout: int = 30
    
    @validator('endpoint')
    def validate_endpoint(cls, v):
        """Validate endpoint URL scheme."""
        parsed = urlparse(v)
        if parsed.scheme not in ['fs', 'http', 'https']:
            raise ValueError(f"Unsupported scheme: {parsed.scheme}. Use fs://, http://, or https://")
        return v
    
    @validator('fallback_endpoint')
    def validate_fallback_endpoint(cls, v):
        """Validate fallback endpoint URL scheme."""
        if v is not None:
            parsed = urlparse(v)
            if parsed.scheme not in ['fs', 'http', 'https']:
                raise ValueError(f"Unsupported fallback scheme: {parsed.scheme}")
        return v
    
    @property
    def parsed(self) -> ParseResult:
        """Parse the primary endpoint URL."""
        return urlparse(self.endpoint)
    
    @property
    def parsed_fallback(self) -> ParseResult | None:
        """Parse the fallback endpoint URL."""
        return urlparse(self.fallback_endpoint) if self.fallback_endpoint else None
    
    @property
    def is_local(self) -> bool:
        """Check if primary endpoint is local filesystem."""
        return self.parsed.scheme == 'fs'
    
    @property
    def is_remote(self) -> bool:
        """Check if primary endpoint is remote HTTP."""
        return self.parsed.scheme in ['http', 'https']
    
    @property
    def local_path(self) -> str | None:
        """Get local filesystem path for fs:// endpoints."""
        if self.is_local:
            return self.parsed.path
        return None
    
    @property
    def base_url(self) -> str | None:
        """Get base URL for http:// endpoints."""
        if self.is_remote:
            parsed = self.parsed
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return None

    @classmethod
    def from_env(cls, env_var: str, default: str) -> "EndpointConfig":
        """Create endpoint config from environment variable."""
        endpoint = os.getenv(env_var, default)
        fallback_var = f"{env_var}_FALLBACK"
        fallback = os.getenv(fallback_var)
        
        return cls(
            endpoint=endpoint,
            fallback_endpoint=fallback
        )
    
    @classmethod
    def memory(cls) -> "EndpointConfig":
        """Default memory endpoint configuration."""
        return cls.from_env("CODIN_MEMORY_ENDPOINT", "fs://./data/memory")
    
    @classmethod  
    def prompt(cls) -> "EndpointConfig":
        """Default prompt endpoint configuration."""
        return cls.from_env("CODIN_PROMPT_ENDPOINT", "fs://./prompt_templates")
    
    @classmethod
    def tool(cls) -> "EndpointConfig":
        """Default tool endpoint configuration.""" 
        return cls.from_env("CODIN_TOOL_ENDPOINT", "fs://./tools")
    
    @classmethod
    def model(cls) -> "EndpointConfig":
        """Default model endpoint configuration."""
        return cls.from_env("CODIN_MODEL_ENDPOINT", "http://api.openai.com/v1")
    
    @classmethod
    def replay(cls) -> "EndpointConfig":
        """Default replay endpoint configuration."""
        return cls.from_env("CODIN_REPLAY_ENDPOINT", "fs://./data/replay")