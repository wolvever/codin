"""Prompt storage backends for codin agents.

This module provides storage backends for prompt templates including
file system and memory-based storage implementations.

Supports both local filesystem and remote HTTP storage with caching.
"""

from __future__ import annotations

import pathlib
import pickle
import tempfile
import time
import logging # Added
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import httpx # Still needed for Response type hint if not fully encapsulated
import yaml
from pydantic import BaseModel, ConfigDict

from .base import ModelOptions, PromptTemplate, PromptVariant
# Import Client and ClientConfig from the correct location
from ...client import Client, ClientConfig # Assuming relative path from src/codin/prompt to src/codin/client

logger = logging.getLogger(__name__) # Added

__all__ = [
    'FilesystemStorage',
    'HTTPStorage',
    'StorageBackend',
    'get_storage_backend',
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
    @abstractmethod
    async def list_templates(self) -> list[tuple[str, str]]:
        """List all available templates as (name, version) tuples."""
    @abstractmethod
    async def save_template(self, template: PromptTemplate) -> None:
        """Save a template to storage."""

    async def close(self) -> None: # Default close for backends that don't need it
        """Close any open resources. Default is a no-op."""
        pass

    async def __aenter__(self) -> StorageBackend:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class FilesystemStorage(StorageBackend):
    """Local filesystem storage backend."""
    def __init__(self, root_path: str | pathlib.Path = 'prompt_templates'):
        self.root_path = pathlib.Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _get_template_path(self, name: str, version: str) -> pathlib.Path:
        if version == 'latest':
            pattern = f'{name}_*.yaml'
            files = list(self.root_path.glob(pattern))
            if files:
                files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return files[0]
            return self.root_path / f'{name}.yaml'
        return self.root_path / f'{name}_{version}.yaml'

    async def load_template(self, name: str, version: str | None = None) -> PromptTemplate | None:
        if version is None: version = 'latest'
        template_path = self._get_template_path(name, version)
        if not template_path.exists():
            template_path = self.root_path / f'{name}.yaml'
            if not template_path.exists(): return None
        try:
            with open(template_path, encoding='utf-8') as f: data = yaml.safe_load(f)
            return self._parse_template_data(data, name, version)
        except Exception: return None

    async def list_templates(self) -> list[tuple[str, str]]:
        templates = []
        for yaml_file in self.root_path.glob('*.yaml'):
            try:
                with open(yaml_file, encoding='utf-8') as f: data = yaml.safe_load(f)
                name = data.get('name', yaml_file.stem)
                version = data.get('version', 'latest')
                templates.append((name, version))
            except Exception:
                stem = yaml_file.stem
                name, version = stem.rsplit('_', 1) if '_' in stem and not stem.count('_') > 2 else (stem, 'latest')
                templates.append((name, version))
        return templates

    async def save_template(self, template: PromptTemplate) -> None:
        template_path = self._get_template_path(template.name, template.version or 'latest')
        data = self._template_to_dict(template)
        with open(template_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def _parse_template_data(self, data: dict, name: str, version: str) -> PromptTemplate:
        variants = []
        if 'variants' in data:
            for variant_data in data['variants']:
                text = variant_data.get('text', '')
                if not text and variant_data.get('messages'): text = '# Content defined by messages structure'
                elif not text: text = variant_data['text']
                variants.append(PromptVariant(
                    text=text, conditions=variant_data.get('conditions'), model=variant_data.get('model'),
                    model_provider=variant_data.get('model_provider'),
                    model_options=self._parse_model_options(variant_data.get('model_options')),
                    system_prompt=variant_data.get('system_prompt'), messages=variant_data.get('messages')))
        elif 'text' in data:
            variants.append(PromptVariant(
                text=data['text'], conditions=data.get('conditions'), model=data.get('model'),
                model_provider=data.get('model_provider'),
                model_options=self._parse_model_options(data.get('model_options')),
                system_prompt=data.get('system_prompt'), messages=data.get('messages')))
        return PromptTemplate(name=data.get('name', name), version=data.get('version', version),
                              variants=variants, metadata=data.get('metadata'))

    def _parse_model_options(self, options_data: dict | None) -> ModelOptions | None:
        return ModelOptions(**options_data) if options_data else None

    def _template_to_dict(self, template: PromptTemplate) -> dict:
        data = {'name': template.name, 'version': template.version, 'variants': []}
        if template.metadata: data['metadata'] = template.metadata
        for variant in template.variants or []:
            variant_data = {k: v for k, v in variant.__dict__.items() if v is not None}
            if variant.model_options: variant_data['model_options'] = variant.model_options.to_dict()
            data['variants'].append(variant_data)
        return data


class HTTPStorage(StorageBackend):
    """Remote HTTP storage backend with local caching, using codin.client.Client."""

    def __init__(self,
                 base_url: str,
                 cache_dir: str | pathlib.Path | None = None,
                 cache_ttl: int = 3600,
                 client: Client | None = None,
                 client_config: ClientConfig | None = None
                 ):
        self.base_url = base_url.rstrip('/')
        self.cache_ttl = cache_ttl
        self._client_provided = client is not None

        if client:
            self._http_client: Client = client
        elif client_config:
            self._http_client = Client(config=client_config)
        else:
            # Use a minimal default config for the internally created client.
            # The base_url for the client itself isn't strictly necessary here since
            # we construct full URLs for requests.
            self._http_client = Client(config=ClientConfig(base_url=""))

        if cache_dir is None:
            self.cache_dir = pathlib.Path(tempfile.gettempdir()) / 'codin_prompt_cache'
        else:
            self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, CacheEntry] = {}
        self._load_cache()
        logger.info(f"HTTPStorage initialized for base_url: {self.base_url}. Client provided: {self._client_provided}")

    async def close(self) -> None:
        """Closes the internal HTTP client if it was created by this instance."""
        if not self._client_provided and self._http_client:
            logger.debug(f"Closing internally managed HTTP client for {self.base_url}")
            await self._http_client.close()
        else:
            logger.debug(f"HTTP client for {self.base_url} was externally provided, not closing.")

    def _get_cache_key(self, name: str, version: str) -> str:
        return f'{name}:{version}'

    def _get_cache_path(self, cache_key: str) -> pathlib.Path:
        safe_key = cache_key.replace(':', '_').replace('/', '_')
        return self.cache_dir / f'{safe_key}.pkl'

    def _load_cache(self) -> None:
        for cache_file in self.cache_dir.glob('*.pkl'):
            try:
                with open(cache_file, 'rb') as f: entry = pickle.load(f)
                if isinstance(entry, CacheEntry):
                    cache_key = cache_file.stem.replace('_', ':', 1) # Corrected potential over-replacement
                    self._cache[cache_key] = entry
            except Exception: cache_file.unlink(missing_ok=True)

    def _save_cache_entry(self, cache_key: str, entry: CacheEntry) -> None:
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f: pickle.dump(entry, f)
            self._cache[cache_key] = entry
        except Exception as e: logger.warning(f"Failed to save cache entry {cache_key}: {e}")

    def _is_cache_valid(self, entry: CacheEntry) -> bool:
        return time.time() - entry.timestamp < self.cache_ttl

    async def load_template(self, name: str, version: str | None = None) -> PromptTemplate | None:
        version = version or 'latest'
        cache_key = self._get_cache_key(name, version)
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if self._is_cache_valid(entry): return entry.template

        try:
            url = f'{self.base_url}/templates/{name}'
            params = {'version': version} if version != 'latest' else {}
            headers = {}
            if cache_key in self._cache and self._cache[cache_key].etag:
                headers['If-None-Match'] = self._cache[cache_key].etag

            logger.debug(f"HTTPStorage: Loading template '{name}' (version: {version}) from {url}")
            response = await self._http_client.get(url, params=params, headers=headers) # Uses codin.client.Client

            if response.status_code == 304:
                if cache_key in self._cache:
                    self._cache[cache_key].timestamp = time.time()
                    logger.debug(f"HTTPStorage: Template '{name}' not modified, using cache.")
                    return self._cache[cache_key].template
            if response.status_code == 404: return None
            response.raise_for_status() # Handled by codin.client.Client's retry/error logic too
            data = response.json()

            # TODO: Use FilesystemStorage._parse_template_data - this is a bit of a hack
            # This indicates a need for a shared parser or moving parsing into PromptTemplate itself
            fs_storage = FilesystemStorage()
            template = fs_storage._parse_template_data(data, name, version)

            entry = CacheEntry(template=template, timestamp=time.time(), etag=response.headers.get('ETag'))
            self._save_cache_entry(cache_key, entry)
            return template
        except Exception as e:
            logger.warning(f"HTTPStorage: Failed to load template '{name}' from remote: {e}. Falling back to cache if available.")
            if cache_key in self._cache: return self._cache[cache_key].template
            return None

    async def list_templates(self) -> list[tuple[str, str]]:
        try:
            url = f'{self.base_url}/templates'
            logger.debug(f"HTTPStorage: Listing templates from {url}")
            response = await self._http_client.get(url)
            response.raise_for_status()
            data = response.json()
            templates = []
            for item in data.get('templates', []):
                name, versions = item.get('name'), item.get('versions', ['latest'])
                for version in versions: templates.append((name, version))
            return templates
        except Exception as e:
            logger.warning(f"HTTPStorage: Failed to list templates from remote: {e}. Falling back to cache.")
            return list(set((entry.template.name, entry.template.version or 'latest') for entry in self._cache.values()))

    async def save_template(self, template: PromptTemplate) -> None:
        try:
            url = f'{self.base_url}/templates/{template.name}'
            fs_storage = FilesystemStorage()
            data = fs_storage._template_to_dict(template)

            logger.debug(f"HTTPStorage: Saving template '{template.name}' to {url}")
            response = await self._http_client.put(url, json=data) # Uses codin.client.Client
            response.raise_for_status()

            cache_key = self._get_cache_key(template.name, template.version or 'latest')
            entry = CacheEntry(template=template, timestamp=time.time(), etag=response.headers.get('ETag'))
            self._save_cache_entry(cache_key, entry)
        except Exception as e:
            logger.error(f"HTTPStorage: Failed to save template '{template.name}' to remote: {e}")
            raise RuntimeError(f'Failed to save template to remote storage: {e}') from e


def get_storage_backend(endpoint: str) -> StorageBackend:
    parsed = urlparse(endpoint)
    if parsed.scheme == 'fs':
        path = '.' + parsed.path if parsed.netloc == '.' and parsed.path else (parsed.netloc if parsed.netloc and not parsed.path else parsed.path)
        return FilesystemStorage(path)
    if parsed.scheme in ('http', 'https'):
        base_url = f'{parsed.scheme}://{parsed.netloc}{parsed.path}'
        # HTTPStorage will create its own default Client instance here
        return HTTPStorage(base_url)
    raise ValueError(f'Unsupported storage endpoint scheme: {parsed.scheme}')
