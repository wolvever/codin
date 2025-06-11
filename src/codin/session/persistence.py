import abc
import typing as _t
import os
import json
import aiofiles
import aiofiles.os # For async os operations like path.exists, remove
import httpx

if _t.TYPE_CHECKING:
    from .base import Session # Forward reference for type hinting

class SessionPersistor(abc.ABC):
    """Abstract base class for session persistence mechanisms."""

    @abc.abstractmethod
    async def load_session(self, session_id: str) -> _t.Optional['Session']:
        """Load a session from the persistence store.

        Args:
            session_id: The ID of the session to load.

        Returns:
            The loaded Session object, or None if not found.
        """
        pass

    @abc.abstractmethod
    async def save_session(self, session: 'Session') -> None:
        """Save a session to the persistence store.

        Args:
            session: The Session object to save.
        """
        pass

    @abc.abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete a session from the persistence store.

        Args:
            session_id: The ID of the session to delete.
        """
        pass

class LocalFilePersistor(SessionPersistor):
    """Persists sessions as JSON files on the local filesystem."""

    def __init__(self, base_path: str):
        self.base_path = base_path
        # Ensure base_path exists, create if not.
        # This should be synchronous as it's part of setup.
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)

    def _get_session_filepath(self, session_id: str) -> str:
        """Helper to get the full file path for a given session_id."""
        return os.path.join(self.base_path, f"{session_id}.json")

    async def load_session(self, session_id: str) -> _t.Optional['Session']:
        """Load a session from a JSON file."""
        filepath = self._get_session_filepath(session_id)
        try:
            if not await aiofiles.os.path.exists(filepath): # type: ignore
                return None
            async with aiofiles.open(filepath, mode='r', encoding='utf-8') as f:
                data = await f.read()
            session_data = json.loads(data)

            from .base import Session
            return Session(**session_data)
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None

    async def save_session(self, session: 'Session') -> None:
        """Save a session to a JSON file."""
        filepath = self._get_session_filepath(session.session_id)
        session_json = session.model_dump_json(indent=2)
        try:
            async with aiofiles.open(filepath, mode='w', encoding='utf-8') as f:
                await f.write(session_json)
        except Exception as e:
            print(f"Error saving session {session.session_id}: {e}")

    async def delete_session(self, session_id: str) -> None:
        """Delete a session file."""
        filepath = self._get_session_filepath(session_id)
        try:
            if await aiofiles.os.path.exists(filepath): # type: ignore
                await aiofiles.os.remove(filepath) # type: ignore
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")


class HttpPersistor(SessionPersistor):
    """Persists sessions via HTTP calls to a remote service."""

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip('/')
        self._client = client if client else httpx.AsyncClient()
        self._owns_client = client is None # Flag to indicate if this instance owns the client

    def _get_session_url(self, session_id: str) -> str:
        return f"{self.base_url}/{session_id}"

    async def load_session(self, session_id: str) -> _t.Optional['Session']:
        """Load a session from the remote HTTP service."""
        url = self._get_session_url(session_id)
        try:
            response = await self._client.get(url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            session_data = response.json()

            from .base import Session
            return Session(**session_data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404: # Log only if not a 404, as 404 is "normal" not found
                 print(f"HTTP error loading session {session_id}: {e.response.status_code} - {e}")
            return None
        except Exception as e:
            print(f"Error loading session {session_id} via HTTP: {e}")
            return None

    async def save_session(self, session: 'Session') -> None:
        """Save a session to the remote HTTP service (using PUT)."""
        url = self._get_session_url(session.session_id)
        session_dict = session.model_dump()
        try:
            response = await self._client.put(url, json=session_dict)
            response.raise_for_status()
        except Exception as e:
            print(f"Error saving session {session.session_id} via HTTP: {e}")

    async def delete_session(self, session_id: str) -> None:
        """Delete a session from the remote HTTP service."""
        url = self._get_session_url(session_id)
        try:
            response = await self._client.delete(url)
            if response.status_code == 404:
                return
            response.raise_for_status()
        except Exception as e:
            print(f"Error deleting session {session_id} via HTTP: {e}")

    async def close(self) -> None:
        """Close the underlying HTTP client if it was created by this instance."""
        if self._owns_client: # Only close if this instance created it
            await self._client.aclose()
