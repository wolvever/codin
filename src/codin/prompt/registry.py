"""Simplified prompt registry with endpoint-based storage."""

from __future__ import annotations

import os
import typing as _t
from dataclasses import dataclass

from .base import PromptTemplate, PromptVariant
from .storage import StorageBackend, get_storage_backend

__all__ = [
    "PromptRegistry",
]


class PromptRegistry:
    """Simplified prompt registry with endpoint-based storage.
    
    Supports both local filesystem and remote HTTP storage with caching.
    Uses endpoint URLs to determine storage backend:
    - fs://./prompt_templates -> FilesystemStorage  
    - http://host:port/path -> HTTPStorage with caching
    """

    _instance: "PromptRegistry | None" = None
    _endpoint: str = "fs://./prompt_templates"
    _storage: StorageBackend | None = None

    def __init__(self, endpoint: str | None = None):
        """Initialize registry with storage endpoint.
        
        Args:
            endpoint: Storage endpoint URL (defaults to fs://./prompt_templates)
        """
        if endpoint:
            self._endpoint = endpoint
        else:
            # Use environment variable if available
            template_dir = os.getenv("PROMPT_TEMPLATE_DIR", "./prompt_templates")
            self._endpoint = f"fs://{template_dir}"
        
        self._storage = None
        self._in_memory_templates: dict[str, dict[str, PromptTemplate]] = {}
        # For backward compatibility
        self._registry: dict[tuple[str, str], PromptTemplate] = {}
        self.run_mode: str = "local"
        # Enable lazy loading for testing
        self._enable_lazy_loading: bool = True

    @classmethod
    def get_instance(cls, endpoint: str | None = None) -> "PromptRegistry":
        """Get the singleton registry instance."""
        if cls._instance is None or (endpoint and endpoint != cls._endpoint):
            cls._instance = cls(endpoint)
        return cls._instance

    @classmethod 
    def set_endpoint(cls, endpoint: str) -> None:
        """Set the global storage endpoint.
        
        Args:
            endpoint: Storage endpoint URL
                     - fs://./prompt_templates (default)
                     - http://host:port/path
        """
        cls._endpoint = endpoint
        # Reset storage to force re-initialization
        if cls._instance:
            cls._instance._storage = None

    @classmethod
    def get_endpoint(cls) -> str:
        """Get the current storage endpoint."""
        return cls._endpoint

    def _get_storage(self) -> StorageBackend:
        """Get or create storage backend."""
        if self._storage is None:
            self._storage = get_storage_backend(self._endpoint)
        return self._storage

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """Get a template by name and version (synchronous for backward compatibility).
        
        Args:
            name: Template name
            version: Template version (None for latest)
            
        Returns:
            PromptTemplate instance
            
        Raises:
            KeyError: If template not found
        """
        if version is None:
            version = "latest"
        
        # Check in-memory cache first
        if name in self._in_memory_templates:
            if version == "latest":
                # Find the actual latest version
                versions = list(self._in_memory_templates[name].keys())
                if versions:
                    # Sort versions to get the latest one
                    latest_version = sorted(versions)[-1]
                    return self._in_memory_templates[name][latest_version]
            elif version in self._in_memory_templates[name]:
                return self._in_memory_templates[name][version]
        
        # Check old-style registry for backward compatibility
        if version == "latest":
            # Find the latest version in the old registry
            matching_keys = [k for k in self._registry.keys() if k[0] == name]
            if matching_keys:
                # Sort by version and get the latest
                latest_key = sorted(matching_keys, key=lambda x: x[1])[-1]
                return self._registry[latest_key]
        else:
            registry_key = (name, version)
            if registry_key in self._registry:
                return self._registry[registry_key]
        
        # 🔄 LAZY LOADING: Try to load from storage backend
        try:
            import asyncio
            
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an async context, we can't use run_until_complete
                # Instead, we should use the async version, but for sync compatibility
                # we'll create a task and wait for it
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._load_template_async(name, version))
                    template = future.result(timeout=30)  # 30 second timeout
            except RuntimeError:
                # No event loop running, we can create our own
                template = asyncio.run(self._load_template_async(name, version))
            
            if template:
                # Cache in memory
                if name not in self._in_memory_templates:
                    self._in_memory_templates[name] = {}
                self._in_memory_templates[name][version] = template
                
                # Also update old-style registry for backward compatibility
                self._registry[(name, version)] = template
                
                return template
                
        except Exception:
            # If lazy loading fails, continue to raise KeyError
            pass
        
        raise KeyError(f"Template '{name}' version '{version}' not found")

    async def _load_template_async(self, name: str, version: str) -> PromptTemplate | None:
        """Helper method to load template from storage backend."""
        storage = self._get_storage()
        return await storage.load_template(name, version)

    async def get_async(self, name: str, version: str | None = None) -> PromptTemplate:
        """Get a template by name and version (async version).
        
        Args:
            name: Template name
            version: Template version (None for latest)
            
        Returns:
            PromptTemplate instance
            
        Raises:
            KeyError: If template not found
        """
        if version is None:
            version = "latest"
        
        # Check in-memory cache first
        if name in self._in_memory_templates and version in self._in_memory_templates[name]:
            return self._in_memory_templates[name][version]
        
        # Load from storage
        storage = self._get_storage()
        template = await storage.load_template(name, version)
        
        if template is None:
            raise KeyError(f"Template '{name}' version '{version}' not found")
        
        # Cache in memory
        if name not in self._in_memory_templates:
            self._in_memory_templates[name] = {}
        self._in_memory_templates[name][version] = template
        
        return template

    async def get_or_none(self, name: str, version: str | None = None) -> PromptTemplate | None:
        """Get a template by name and version, returning None if not found.
        
        Args:
            name: Template name
            version: Template version (None for latest)
            
        Returns:
            PromptTemplate instance or None if not found
        """
        try:
            return await self.get_async(name, version)
        except KeyError:
            return None

    def register(self, template: PromptTemplate) -> None:
        """Register a template in memory.
        
        Args:
            template: Template to register
        """
        name = template.name
        version = template.version or "latest"
        
        if name not in self._in_memory_templates:
            self._in_memory_templates[name] = {}
        
        self._in_memory_templates[name][version] = template
        
        # Also update old-style registry for backward compatibility
        self._registry[(name, version)] = template

    async def save(self, template: PromptTemplate) -> None:
        """Save a template to storage.
        
        Args:
            template: Template to save
        """
        storage = self._get_storage()
        await storage.save_template(template)
        
        # Update in-memory cache
        self.register(template)

    def list(self, name: str | None = None) -> list[PromptTemplate]:
        """List templates (backward compatible version).
        
        Args:
            name: Optional template name filter
            
        Returns:
            List of PromptTemplate objects
        """
        templates = []
        
        if name:
            # Filter by name
            if name in self._in_memory_templates:
                templates.extend(self._in_memory_templates[name].values())
        else:
            # Return all templates
            for name_dict in self._in_memory_templates.values():
                templates.extend(name_dict.values())
        
        return templates

    async def list_async(self) -> list[tuple[str, str]]:
        """List all available templates (async version).
        
        Returns:
            List of (name, version) tuples
        """
        storage = self._get_storage()
        storage_templates = await storage.list_templates()
        
        # Add in-memory templates
        memory_templates = []
        for name, versions in self._in_memory_templates.items():
            for version in versions:
                memory_templates.append((name, version))
        
        # Combine and deduplicate
        all_templates = list(set(storage_templates + memory_templates))
        return sorted(all_templates)

    def set_run_mode(self, mode: str) -> None:
        """Set the run mode.
        
        Args:
            mode: Run mode ("local" or "remote")
            
        Raises:
            ValueError: If mode is invalid
        """
        if mode not in ("local", "remote"):
            raise ValueError(f"Invalid run mode: {mode}. Must be 'local' or 'remote'")
        self.run_mode = mode

    @classmethod
    def prompt(cls, name: str, *, version: str | None = None):
        """Decorator to register a prompt template function.
        
        Args:
            name: Template name
            version: Template version
            
        Returns:
            Decorator function
        """
        def decorator(func: _t.Callable[[], str]) -> _t.Callable[[], str]:
            text = func()
            template = PromptTemplate(name=name, version=version or "latest", text=text)
            instance = cls.get_instance()
            instance.register(template)
            return func
        
        return decorator

    async def clear_cache(self) -> None:
        """Clear the in-memory template cache."""
        self._in_memory_templates.clear()
        self._registry.clear()

    # Class methods for backward compatibility
    @classmethod
    def register_template(cls, template: PromptTemplate) -> None:
        """Register a template (class method)."""
        instance = cls.get_instance()
        instance.register(template)

    @classmethod
    def get_template(cls, name: str, version: str | None = None) -> PromptTemplate:
        """Get a template (class method)."""
        instance = cls.get_instance()
        return instance.get(name, version)

    @classmethod
    def list_templates(cls) -> list[PromptTemplate]:
        """List all templates (class method)."""
        instance = cls.get_instance()
        return instance.list()

    @classmethod
    def get_mode(cls) -> str:
        """Get the current run mode (class method)."""
        instance = cls.get_instance()
        return instance.run_mode

    @classmethod
    def set_mode(cls, mode: str) -> None:
        """Set the run mode (class method)."""
        instance = cls.get_instance()
        instance.set_run_mode(mode)


# Global convenience functions for backward compatibility
_global_registry: PromptRegistry | None = None


def get_registry(endpoint: str | None = None) -> PromptRegistry:
    """Get the global registry instance."""
    global _global_registry
    
    # Use environment variable if no explicit endpoint
    if endpoint is None:
        endpoint = os.getenv("CODIN_PROMPT_ENDPOINT", "fs://./prompt_templates")
    
    if _global_registry is None or endpoint != PromptRegistry.get_endpoint():
        PromptRegistry.set_endpoint(endpoint)
        _global_registry = PromptRegistry.get_instance(endpoint)
    
    return _global_registry


def set_endpoint(endpoint: str) -> None:
    """Set the global prompt storage endpoint."""
    global _global_registry
    PromptRegistry.set_endpoint(endpoint)
    _global_registry = None  # Force re-creation 