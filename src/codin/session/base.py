"""Session management components for codin agents.

This module provides session management components including session state tracking
(ID, context, basic metrics), and session lifecycle management.
"""

import asyncio
import typing as _t
from contextlib import asynccontextmanager
from datetime import datetime
import os # Added
from urllib.parse import urlparse # Added

from pydantic import BaseModel, ConfigDict, Field

# Removed ReplayService and other unused imports from here, will be handled by __init__.py if needed
# from codin.replay.base import ReplayService
# from ..agent.types import State # Not directly used in this file after refactor
# from ..memory.base import MemMemoryService, MemoryService # Not directly used in this file after refactor

# Added for persistence
from .persistence import SessionPersistor, LocalFilePersistor, HttpPersistor


__all__ = [
    # ReplayService might be re-exported from __init__ if still needed there
    'Session',
    'SessionManager',
]


# =============================================================================
# Data-oriented Session classes (merged from agent/session.py)
# =============================================================================


class Session(BaseModel):
    """Data-oriented session that holds session identifiers, creation timestamp,
    a generic context dictionary, and high-level metrics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)

    # Execution metrics
    metrics: dict[str, _t.Any] = Field(default_factory=dict)
    context: dict[str, _t.Any] = Field(default_factory=dict)

    def __post_init__(self):
        """Initialize default metrics."""
        if not self.metrics:
            self.metrics = {
                'start_time': self.created_at.timestamp(),
                'total_tool_calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'cost': 0.0,
                'last_activity': self.created_at.timestamp(),
            }
        else:
            self.metrics.setdefault('start_time', self.created_at.timestamp())
            self.metrics['last_activity'] = datetime.now().timestamp()


    def get_metrics_summary(self) -> dict[str, _t.Any]:
        """Get a summary of session metrics."""
        current_time = datetime.now().timestamp()
        start_time = self.metrics.get('start_time', self.created_at.timestamp())
        elapsed = current_time - start_time

        return {
            'session_id': self.session_id,
            'elapsed_seconds': elapsed,
            'total_tool_calls': self.metrics.get('total_tool_calls', 0),
            'input_tokens': self.metrics.get('input_tokens', 0),
            'output_tokens': self.metrics.get('output_tokens', 0),
            'cost': self.metrics.get('cost', 0.0),
            'last_activity': self.metrics.get('last_activity', self.created_at.timestamp()),
        }


class SessionManager:
    """Manages active sessions with optional persistence and cleanup."""

    def __init__(self, endpoint: _t.Optional[str] = None):
        """Initialize the SessionManager.

        Args:
            endpoint: An optional URI or local path for session persistence.
                If None, sessions are in-memory only.
                Supported schemes:
                - Local file system:
                    - URI: `file:///path/to/sessions_directory`
                    - Absolute path: `/abs/path/to/sessions_directory`
                    - Relative path: `rel/path/to/sessions_directory`
                - HTTP service:
                    - URI: `http://host/api/sessions`
                    - URI: `https://host/api/sessions`
                Sessions are stored as JSON files in the specified directory for
                file persistence, or sent to the HTTP endpoint.
        """
        self._sessions: dict[str, Session] = {}
        self._persistor: _t.Optional[SessionPersistor] = None
        self._cleanup_lock = asyncio.Lock()

        if endpoint:
            parsed_url = urlparse(endpoint)
            if parsed_url.scheme in ('http', 'https'):
                self._persistor = HttpPersistor(endpoint)
            elif parsed_url.scheme == 'file':
                # Adjust for file URI format (e.g., file:///path or file:C:/path on Windows)
                path = os.path.abspath(os.path.join(parsed_url.netloc, parsed_url.path))
                if os.name == 'nt': # Windows path adjustments
                    # Remove leading '/' if path starts like /C:/
                    if path.startswith('/') and len(path) > 2 and path[2] == ':':
                        path = path[1:]
                    # Handle drive letter if already correct from urlparse.netloc
                    elif len(parsed_url.netloc) > 1 and parsed_url.netloc[1] == ':':
                         path = parsed_url.netloc + parsed_url.path
                self._persistor = LocalFilePersistor(path)
            elif not parsed_url.scheme and endpoint: # No scheme, assume local path
                self._persistor = LocalFilePersistor(endpoint)
            # else: self._persistor remains None for unrecognized schemes or if endpoint is just a name

    async def get_or_create_session(self, session_id: str) -> Session:
        """Get existing session from memory, load from persistor, or create a new one."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        if self._persistor:
            try:
                # print(f"Attempting to load session {session_id} from persistor.") # Debug
                loaded_session = await self._persistor.load_session(session_id)
                if loaded_session:
                    # print(f"Loaded session {session_id} successfully.") # Debug
                    self._sessions[session_id] = loaded_session
                    return loaded_session
            except Exception as e:
                # Consider using a proper logger in a real application
                print(f"Error loading session {session_id} from persistence: {e}") # Placeholder for logging

        # print(f"Creating new session {session_id}.") # Debug
        new_session = Session(session_id=session_id)
        self._sessions[session_id] = new_session
        # If persistor exists, save the newly created session immediately
        # This behavior might be desirable to ensure the session exists in persistent store
        # as soon as it's created, even if the context manager exits prematurely.
        # However, the original spec saves only on successful context exit.
        # For now, adhering to saving only on successful context exit or full cleanup.
        return new_session

    def get_session(self, session_id: str) -> Session | None:
        """Get existing session by ID from memory cache."""
        return self._sessions.get(session_id)

    @asynccontextmanager
    async def session(
        self, session_id: str
    ) -> _t.AsyncGenerator[Session, None]:
        """Context manager to manage a session's lifecycle.
        Retrieves from memory/persistor or creates a session.
        Saves on successful context exit if a persistor is configured.
        Always removes from in-memory cache on exit.
        """
        session_obj = await self.get_or_create_session(session_id)
        succeeded = False
        try:
            yield session_obj
            succeeded = True # Mark as succeeded if yield completes without error
        finally:
            if session_id in self._sessions: # Check if session still in memory (it should be)
                if self._persistor and succeeded: # Only save if context block was successful
                    try:
                        # print(f"Saving session {session_obj.session_id} on context exit.") # Debug
                        await self._persistor.save_session(session_obj)
                    except Exception as e:
                        print(f"Error saving session {session_obj.session_id} on context exit: {e}")
            # Always remove from in-memory cache after context manager usage.
            # If not persisted, it's lost. If persisted, it's loaded next time.
            if session_id in self._sessions:
                del self._sessions[session_id]


    async def close_session(self, session_id: str) -> None:
        """Removes a session from the in-memory cache.
        Note: Persistence (saving or deleting from store) is handled by
        the session context manager or the main cleanup method.
        This method primarily ensures the session is no longer active in memory.
        """
        if session_id in self._sessions:
            # print(f"Closing session {session_id} from memory (explicit call).") # Debug
            del self._sessions[session_id]
        # Optionally, one might want to also delete from persistor here,
        # but current design has explicit delete on persistor or cleanup saving.
        # For now, this just removes from memory.

    async def cleanup_inactive_sessions(self, max_age_seconds: float = 3600) -> int:
        """Cleanup sessions that haven't been active for specified time.
        This involves saving them to persistence (if configured) and removing from memory.
        """
        async with self._cleanup_lock:
            current_time = datetime.now().timestamp()
            inactive_session_ids = []

            # Iterate over a copy of items for safe removal if needed during iteration elsewhere
            for session_id, session_obj in list(self._sessions.items()):
                last_activity = session_obj.metrics.get('last_activity', session_obj.created_at.timestamp())
                if current_time - last_activity > max_age_seconds:
                    inactive_session_ids.append(session_id)

            for session_id in inactive_session_ids:
                if session_id in self._sessions: # Check if still present
                    session_to_cleanup = self._sessions[session_id]
                    if self._persistor:
                        try:
                            # print(f"Saving inactive session {session_id} due to inactivity.") # Debug
                            await self._persistor.save_session(session_to_cleanup)
                        except Exception as e:
                            print(f"Error saving inactive session {session_id}: {e}")
                    # Remove from memory after attempting to save
                    del self._sessions[session_id]
            return len(inactive_session_ids)

    def get_active_sessions(self) -> dict[str, dict[str, _t.Any]]:
        """Get summary of all active sessions currently in memory."""
        return {session_id: session.get_metrics_summary() for session_id, session in self._sessions.items()}

    async def cleanup(self) -> None:
        """Saves all currently active in-memory sessions to the persistor (if configured),
        closes the persistor if it owns its client, and clears the in-memory session cache.
        """
        if self._persistor:
            # print("SessionManager cleanup: saving all active in-memory sessions.") # Debug
            for session_id, session_obj in list(self._sessions.items()): # Iterate over a copy
                try:
                    # print(f"Saving session {session_id} during main cleanup.") # Debug
                    await self._persistor.save_session(session_obj)
                except Exception as e:
                    print(f"Error saving session {session_id} during main cleanup: {e}")

            if hasattr(self._persistor, 'close') and callable(getattr(self._persistor, 'close')):
                try:
                    # print("Closing persistor during main cleanup.") # Debug
                    await self._persistor.close() # type: ignore
                except Exception as e:
                    print(f"Error closing persistor during main cleanup: {e}")

        # print("Clearing all in-memory sessions during main cleanup.") # Debug
        self._sessions.clear()
