"""Centralized endpoint resolver for consistent endpoint handling."""
import asyncio
import logging
from typing import Optional, Union
from .config import EndpointConfig
from .backends import Backend, LocalBackend, RemoteBackend

logger = logging.getLogger(__name__)


class EndpointResolver:
    """Centralized resolver for endpoint configuration and backend creation."""
    
    _cache: dict[str, Backend] = {}
    
    @classmethod
    def create_backend(cls, config: EndpointConfig) -> Backend:
        """Create backend from endpoint configuration."""
        if config.is_local:
            return LocalBackend(config.local_path)
        elif config.is_remote:
            return RemoteBackend(
                base_url=config.base_url,
                auth=config.auth,
                timeout=config.timeout
            )
        else:
            raise ValueError(f"Unsupported endpoint scheme: {config.parsed.scheme}")
    
    @classmethod
    def get_backend(cls, config: EndpointConfig, use_cache: bool = True) -> Backend:
        """Get backend with optional caching."""
        cache_key = config.endpoint
        
        if use_cache and cache_key in cls._cache:
            return cls._cache[cache_key]
        
        backend = cls.create_backend(config)
        
        if use_cache:
            cls._cache[cache_key] = backend
        
        return backend
    
    @classmethod
    async def get_backend_with_fallback(cls, config: EndpointConfig) -> Backend:
        """Get backend with automatic fallback support."""
        primary_backend = cls.get_backend(config)
        
        # If no fallback configured, return primary
        if not config.fallback_endpoint:
            return primary_backend
        
        # Test primary backend connectivity
        try:
            if config.is_remote:
                # Quick connectivity test for remote backends
                await asyncio.wait_for(
                    primary_backend.exists(""), 
                    timeout=5.0
                )
            return primary_backend
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(
                f"Primary endpoint {config.endpoint} failed ({e}), "
                f"falling back to {config.fallback_endpoint}"
            )
            
            # Create fallback configuration and backend
            fallback_config = EndpointConfig(
                endpoint=config.fallback_endpoint,
                auth=config.auth,
                timeout=config.timeout
            )
            return cls.get_backend(fallback_config)
    
    @classmethod
    async def test_connectivity(cls, config: EndpointConfig) -> bool:
        """Test endpoint connectivity."""
        try:
            backend = cls.get_backend(config, use_cache=False)
            
            if config.is_local:
                # For local endpoints, check if base path exists
                return await backend.exists("")
            else:
                # For remote endpoints, test with timeout
                await asyncio.wait_for(
                    backend.exists(""),
                    timeout=config.timeout
                )
                return True
        except Exception as e:
            logger.debug(f"Connectivity test failed for {config.endpoint}: {e}")
            return False
    
    @classmethod
    def clear_cache(cls):
        """Clear the backend cache."""
        cls._cache.clear()
    
    @classmethod
    async def cleanup(cls):
        """Cleanup all cached backends."""
        for backend in cls._cache.values():
            if hasattr(backend, 'close'):
                await backend.close()
        cls.clear_cache()


class EndpointManager:
    """Convenience manager for component-specific endpoint handling."""
    
    def __init__(self, config: EndpointConfig):
        self.config = config
        self._backend: Optional[Backend] = None
    
    async def get_backend(self) -> Backend:
        """Get backend with fallback support."""
        if self._backend is None:
            self._backend = await EndpointResolver.get_backend_with_fallback(self.config)
        return self._backend
    
    async def read(self, path: str) -> bytes:
        """Read data from endpoint."""
        backend = await self.get_backend()
        return await backend.read(path)
    
    async def write(self, path: str, data: bytes) -> None:
        """Write data to endpoint."""
        backend = await self.get_backend()
        await backend.write(path, data)
    
    async def exists(self, path: str) -> bool:
        """Check if resource exists."""
        backend = await self.get_backend()
        return await backend.exists(path)
    
    async def list(self, path: str = "") -> list[str]:
        """List resources."""
        backend = await self.get_backend()
        return await backend.list(path)
    
    async def delete(self, path: str) -> None:
        """Delete resource."""
        backend = await self.get_backend()
        await backend.delete(path)
    
    async def close(self):
        """Close backend connection."""
        if self._backend and hasattr(self._backend, 'close'):
            await self._backend.close()
        self._backend = None