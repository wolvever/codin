"""Storage backends for prompt templates.

Supports both local filesystem and remote HTTP storage with caching.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import tempfile
import time
import typing as _t
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel, ConfigDict

from .base import PromptTemplate, PromptVariant, ModelOptions

__all__ = [
    "StorageBackend",
    "FilesystemStorage", 
    "HTTPStorage",
    "get_storage_backend",
]


class CacheEntry(BaseModel):
    """Cache entry for remote templates."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    template: PromptTemplate
    timestamp: float
    etag: str | None = None


class StorageBackend(ABC):
    """Abstract base class for prompt template storage backends."""
    
    @abstractmethod
    async def load_template(self, name: str, version: str | None = None) -> PromptTemplate | None:
        """Load a template by name and version."""
        pass
    
    @abstractmethod
    async def list_templates(self) -> list[tuple[str, str]]:
        """List all available templates as (name, version) tuples."""
        pass
    
    @abstractmethod
    async def save_template(self, template: PromptTemplate) -> None:
        """Save a template to storage."""
        pass


class FilesystemStorage(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, root_path: str | pathlib.Path = "prompt_templates"):
        """Initialize filesystem storage.
        
        Args:
            root_path: Root directory for prompt templates
        """
        self.root_path = pathlib.Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
    
    def _get_template_path(self, name: str, version: str) -> pathlib.Path:
        """Get the file path for a template."""
        # Use version as filename suffix for multiple versions
        if version == "latest":
            # For latest, try to find the most recent version
            pattern = f"{name}_*.yaml"
            files = list(self.root_path.glob(pattern))
            if files:
                # Sort by modification time, newest first
                files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return files[0]
            else:
                # Fallback to simple name
                return self.root_path / f"{name}.yaml"
        else:
            return self.root_path / f"{name}_{version}.yaml"
    
    async def load_template(self, name: str, version: str | None = None) -> PromptTemplate | None:
        """Load a template from the filesystem."""
        if version is None:
            version = "latest"
        
        template_path = self._get_template_path(name, version)
        
        if not template_path.exists():
            # Try without version suffix
            template_path = self.root_path / f"{name}.yaml"
            if not template_path.exists():
                return None
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            return self._parse_template_data(data, name, version)
            
        except Exception:
            return None
    
    async def list_templates(self) -> list[tuple[str, str]]:
        """List all available templates."""
        templates = []
        
        for yaml_file in self.root_path.glob("*.yaml"):
            # Load the file to get the actual name and version from the YAML content
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                # Get name and version from YAML content, fallback to filename
                name = data.get("name", yaml_file.stem)
                version = data.get("version", "latest")
                
                templates.append((name, version))
            except Exception:
                # Fallback to filename parsing if YAML loading fails
                stem = yaml_file.stem
                if "_" in stem and not stem.count("_") > 2:  # Only split if it looks like name_version
                    name, version = stem.rsplit("_", 1)
                else:
                    name, version = stem, "latest"
                templates.append((name, version))
        
        return templates
    
    async def save_template(self, template: PromptTemplate) -> None:
        """Save a template to the filesystem."""
        template_path = self._get_template_path(template.name, template.version or "latest")
        
        # Convert template to YAML format
        data = self._template_to_dict(template)
        
        with open(template_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def _parse_template_data(self, data: dict, name: str, version: str) -> PromptTemplate:
        """Parse template data from YAML/JSON format."""
        # Create variants from the data
        variants = []
        
        if "variants" in data:
            for variant_data in data["variants"]:
                # Make text optional if messages are provided
                text = variant_data.get("text", "")
                if not text and variant_data.get("messages"):
                    # If no text but messages exist, use a placeholder
                    text = "# Content defined by messages structure"
                elif not text:
                    # Require text if no messages
                    text = variant_data["text"]  # This will raise KeyError if missing
                
                variant = PromptVariant(
                    text=text,
                    conditions=variant_data.get("conditions"),
                    model=variant_data.get("model"),
                    model_provider=variant_data.get("model_provider"),
                    model_options=self._parse_model_options(variant_data.get("model_options")),
                    system_prompt=variant_data.get("system_prompt"),
                    messages=variant_data.get("messages")
                )
                variants.append(variant)
        elif "text" in data:
            # Single variant template
            variant = PromptVariant(
                text=data["text"],
                conditions=data.get("conditions"),
                model=data.get("model"),
                model_provider=data.get("model_provider"),
                model_options=self._parse_model_options(data.get("model_options")),
                system_prompt=data.get("system_prompt"),
                messages=data.get("messages")
            )
            variants.append(variant)
        
        return PromptTemplate(
            name=data.get("name", name),
            version=data.get("version", version),
            variants=variants,
            metadata=data.get("metadata")
        )
    
    def _parse_model_options(self, options_data: dict | None) -> ModelOptions | None:
        """Parse model options from dict."""
        if not options_data:
            return None
        
        return ModelOptions(**options_data)
    
    def _template_to_dict(self, template: PromptTemplate) -> dict:
        """Convert template to dictionary format."""
        data = {
            "name": template.name,
            "version": template.version,
            "variants": []
        }
        
        if template.metadata:
            data["metadata"] = template.metadata
        
        for variant in template.variants or []:
            variant_data = {
                "text": variant.text,
                "conditions": variant.conditions,
                "model_provider": variant.model_provider,
                "model": variant.model,
                "model_options": variant.model_options.to_dict() if variant.model_options else None,
                "system_prompt": variant.system_prompt,
                "messages": variant.messages
            }
            
            # Remove None values
            variant_data = {k: v for k, v in variant_data.items() if v is not None}
            data["variants"].append(variant_data)
        
        return data


class HTTPStorage(StorageBackend):
    """Remote HTTP storage backend with local caching."""
    
    def __init__(self, base_url: str, cache_dir: str | pathlib.Path | None = None, cache_ttl: int = 3600):
        """Initialize HTTP storage.
        
        Args:
            base_url: Base URL for the remote prompt registry
            cache_dir: Local cache directory (None for temp dir)
            cache_ttl: Cache TTL in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.cache_ttl = cache_ttl
        
        if cache_dir is None:
            self.cache_dir = pathlib.Path(tempfile.gettempdir()) / "codin_prompt_cache"
        else:
            self.cache_dir = pathlib.Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, CacheEntry] = {}
        self._load_cache()
    
    def _get_cache_key(self, name: str, version: str) -> str:
        """Get cache key for a template."""
        return f"{name}:{version}"
    
    def _get_cache_path(self, cache_key: str) -> pathlib.Path:
        """Get cache file path."""
        safe_key = cache_key.replace(":", "_").replace("/", "_")
        return self.cache_dir / f"{safe_key}.pkl"
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                    if isinstance(entry, CacheEntry):
                        # Extract cache key from filename
                        cache_key = cache_file.stem.replace("_", ":")
                        self._cache[cache_key] = entry
            except Exception:
                # Remove corrupted cache files
                cache_file.unlink(missing_ok=True)
    
    def _save_cache_entry(self, cache_key: str, entry: CacheEntry) -> None:
        """Save cache entry to disk."""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(entry, f)
            self._cache[cache_key] = entry
        except Exception:
            pass  # Ignore cache save errors
    
    def _is_cache_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - entry.timestamp < self.cache_ttl
    
    async def load_template(self, name: str, version: str | None = None) -> PromptTemplate | None:
        """Load a template from remote HTTP endpoint."""
        if version is None:
            version = "latest"
        
        cache_key = self._get_cache_key(name, version)
        
        # Check cache first
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if self._is_cache_valid(entry):
                return entry.template
        
        # Fetch from remote
        try:
            url = f"{self.base_url}/templates/{name}"
            params = {"version": version} if version != "latest" else {}
            
            headers = {}
            # Add If-None-Match header if we have cached etag
            if cache_key in self._cache and self._cache[cache_key].etag:
                headers["If-None-Match"] = self._cache[cache_key].etag
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers, timeout=30.0)
                
                if response.status_code == 304:  # Not Modified
                    # Update cache timestamp
                    if cache_key in self._cache:
                        self._cache[cache_key].timestamp = time.time()
                        return self._cache[cache_key].template
                
                if response.status_code == 404:
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                # Parse template
                fs_storage = FilesystemStorage()  # Use FS parser
                template = fs_storage._parse_template_data(data, name, version)
                
                # Cache the result
                entry = CacheEntry(
                    template=template,
                    timestamp=time.time(),
                    etag=response.headers.get("ETag")
                )
                self._save_cache_entry(cache_key, entry)
                
                return template
                
        except Exception:
            # Fall back to cache if available
            if cache_key in self._cache:
                return self._cache[cache_key].template
            return None
    
    async def list_templates(self) -> list[tuple[str, str]]:
        """List all available templates from remote endpoint."""
        try:
            url = f"{self.base_url}/templates"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                
                templates = []
                for item in data.get("templates", []):
                    name = item.get("name")
                    versions = item.get("versions", ["latest"])
                    for version in versions:
                        templates.append((name, version))
                
                return templates
                
        except Exception:
            # Fall back to cached templates
            return list(set((entry.template.name, entry.template.version) 
                           for entry in self._cache.values()))
    
    async def save_template(self, template: PromptTemplate) -> None:
        """Save a template to remote endpoint."""
        try:
            url = f"{self.base_url}/templates/{template.name}"
            
            # Convert template to dict
            fs_storage = FilesystemStorage()  # Use FS serializer
            data = fs_storage._template_to_dict(template)
            
            async with httpx.AsyncClient() as client:
                response = await client.put(url, json=data, timeout=30.0)
                response.raise_for_status()
                
                # Update cache
                cache_key = self._get_cache_key(template.name, template.version or "latest")
                entry = CacheEntry(
                    template=template,
                    timestamp=time.time(),
                    etag=response.headers.get("ETag")
                )
                self._save_cache_entry(cache_key, entry)
                
        except Exception as e:
            raise RuntimeError(f"Failed to save template to remote storage: {e}")


def get_storage_backend(endpoint: str) -> StorageBackend:
    """Create storage backend based on endpoint URL.
    
    Args:
        endpoint: Storage endpoint URL
                 - fs://./path/to/templates -> FilesystemStorage
                 - http://host:port/path -> HTTPStorage
                 - https://host:port/path -> HTTPStorage
    
    Returns:
        Appropriate storage backend instance
    """
    parsed = urlparse(endpoint)
    
    if parsed.scheme == "fs":
        # Filesystem storage - handle special case of fs://./path
        if parsed.netloc == "." and parsed.path:
            # This is fs://./path format
            path = "." + parsed.path  # Convert /prompt_templates to ./prompt_templates
        elif parsed.netloc and not parsed.path:
            # This is fs://path format (no leading /)
            path = parsed.netloc
        else:
            # This is regular fs:///absolute/path format
            path = parsed.path
        
        return FilesystemStorage(path)
    
    elif parsed.scheme in ("http", "https"):
        # HTTP storage
        base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return HTTPStorage(base_url)
    
    else:
        raise ValueError(f"Unsupported storage endpoint scheme: {parsed.scheme}") 