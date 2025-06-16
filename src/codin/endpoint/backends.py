"""Backend interfaces for unified endpoint handling."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import aiohttp
import aiofiles


class Backend(ABC):
    """Abstract backend interface for endpoint handling."""
    
    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if resource exists at path."""
        pass
    
    @abstractmethod
    async def read(self, path: str) -> bytes:
        """Read resource from path."""
        pass
    
    @abstractmethod
    async def write(self, path: str, data: bytes) -> None:
        """Write data to path."""
        pass
    
    @abstractmethod
    async def list(self, path: str = "") -> list[str]:
        """List resources at path."""
        pass
    
    @abstractmethod
    async def delete(self, path: str) -> None:
        """Delete resource at path."""
        pass


class LocalBackend(Backend):
    """Local filesystem backend."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve relative path to absolute path within base_path."""
        if path.startswith('/'):
            path = path[1:]
        return self.base_path / path
    
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        return self._resolve_path(path).exists()
    
    async def read(self, path: str) -> bytes:
        """Read file contents."""
        file_path = self._resolve_path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    
    async def write(self, path: str, data: bytes) -> None:
        """Write data to file."""
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)
    
    async def list(self, path: str = "") -> list[str]:
        """List files in directory."""
        dir_path = self._resolve_path(path)
        if not dir_path.exists():
            return []
        
        if dir_path.is_file():
            return [path]
        
        files = []
        for item in dir_path.iterdir():
            relative_path = str(item.relative_to(self.base_path))
            files.append(relative_path)
        
        return sorted(files)
    
    async def delete(self, path: str) -> None:
        """Delete file or directory."""
        file_path = self._resolve_path(path)
        if file_path.exists():
            if file_path.is_file():
                file_path.unlink()
            else:
                import shutil
                shutil.rmtree(file_path)


class RemoteBackend(Backend):
    """Remote HTTP backend."""
    
    def __init__(self, base_url: str, auth: Optional[Dict[str, Any]] = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.auth = auth or {}
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            headers = {}
            if 'api_key' in self.auth:
                headers['Authorization'] = f"Bearer {self.auth['api_key']}"
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        
        return self._session
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        path = path.lstrip('/')
        return f"{self.base_url}/{path}"
    
    async def exists(self, path: str) -> bool:
        """Check if resource exists via HEAD request."""
        session = await self._get_session()
        url = self._build_url(path)
        
        try:
            async with session.head(url) as response:
                return response.status == 200
        except aiohttp.ClientError:
            return False
    
    async def read(self, path: str) -> bytes:
        """Read resource via GET request."""
        session = await self._get_session()
        url = self._build_url(path)
        
        async with session.get(url) as response:
            if response.status == 404:
                raise FileNotFoundError(f"Resource not found: {path}")
            response.raise_for_status()
            return await response.read()
    
    async def write(self, path: str, data: bytes) -> None:
        """Write resource via PUT request."""
        session = await self._get_session()
        url = self._build_url(path)
        
        async with session.put(url, data=data) as response:
            response.raise_for_status()
    
    async def list(self, path: str = "") -> list[str]:
        """List resources via GET request to directory endpoint."""
        session = await self._get_session()
        url = self._build_url(path)
        
        # Assume the remote service provides a directory listing endpoint
        list_url = f"{url}?list=true" if not url.endswith('/') else f"{url}?list=true"
        
        async with session.get(list_url) as response:
            if response.status == 404:
                return []
            response.raise_for_status()
            result = await response.json()
            return result.get('files', [])
    
    async def delete(self, path: str) -> None:
        """Delete resource via DELETE request."""
        session = await self._get_session()
        url = self._build_url(path)
        
        async with session.delete(url) as response:
            if response.status != 404:  # Ignore not found
                response.raise_for_status()