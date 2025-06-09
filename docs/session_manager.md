# SessionManager Helper

`SessionManager` provides robust management for agent execution sessions, including in-memory caching and optional persistence.

## Asynchronous Context Manager

It exposes an asynchronous context manager to simplify working with sessions. Use `async with session_manager.session(session_id)` to retrieve (from memory, or from a persistor if configured) or create a new session. The session is available within the context block and is automatically managed:
- If a persistor is configured, the session will be saved upon successful exit of the context block.
- The session is always removed from the in-memory cache upon exiting the context block.

```python
from codin.session import SessionManager, Session # Assuming Session might be type hinted or used

# In-memory only session manager
manager = SessionManager()

async def process_with_session(session_id: str):
    async with manager.session(session_id) as current_session:
        # current_session is a Session object
        # Run agent logic with this session
        current_session.context["my_data"] = "some value"
        print(f"Processing session: {current_session.session_id}, data: {current_session.context}")
    # current_session is now removed from in-memory cache.
    # If manager had a persistor, current_session would have been saved.

# Example usage:
# await process_with_session("demo_session_1")
```

## Session Persistence

`SessionManager` can persist sessions by providing an `endpoint` URI or local path during its instantiation. This allows sessions to be reloaded across different manager instances or server restarts.

### Supported Endpoint Types:

1.  **Local File System**:
    *   Sessions are stored as individual JSON files in the specified directory.
    *   **URI format**: `file:///path/to/your/sessions_directory`
        *   Example: `SessionManager(endpoint="file:///app/data/agent_sessions")`
    *   **Direct path**: Relative or absolute paths are also supported.
        *   Example (relative): `SessionManager(endpoint="data/agent_sessions")`
        *   Example (absolute): `SessionManager(endpoint="/var/codin/agent_sessions")`

2.  **HTTP Service**:
    *   Sessions are persisted by making HTTP requests to a remote service.
    *   `SessionManager` will issue:
        *   `GET {endpoint}/{session_id}` to load a session.
        *   `PUT {endpoint}/{session_id}` with JSON body to save a session.
        *   `DELETE {endpoint}/{session_id}` to delete a session.
    *   **URI format**: `http://your-service.com/api/sessions` or `https://your-service.com/api/sessions`
        *   Example: `SessionManager(endpoint="https://api.example.com/prod/sessions")`
    *   This uses the `httpx` library internally. If using this feature, ensure `httpx` is included in your project's dependencies.

### Examples:

```python
from codin.session import SessionManager

# 1. In-memory sessions (default behavior)
in_memory_manager = SessionManager()
# Sessions managed by in_memory_manager are not persisted.

# 2. Local file persistence
# Sessions will be stored in 'project_root/my_sessions_data/'
# (directory created if it doesn't exist)
file_persistor_manager = SessionManager(endpoint="my_sessions_data")

# Example using a file URI (useful for unambiguous paths)
# manager_file_uri = SessionManager(endpoint="file:///var/app/data/sessions")

# 3. HTTP persistence
# Assumes an HTTP service is running at https://api.example.com/sessions
# http_persistor_manager = SessionManager(endpoint="https://api.example.com/sessions")

# --- Usage with a configured manager ---
# async def use_persistent_session(manager: SessionManager, session_id: str):
#     async with manager.session(session_id) as session:
#         session.context["interaction_count"] = session.context.get("interaction_count", 0) + 1
#         print(f"Session {session_id} updated. Count: {session.context['interaction_count']}")
#     # Session is saved here if manager has a persistor and context exited cleanly.

# Example with file persistor:
# await use_persistent_session(file_persistor_manager, "user123_chat")
# await use_persistent_session(file_persistor_manager, "user123_chat") # Loads previous state
```

When a persistor is configured:
- `SessionManager.session(session_id)` will first try to load the session from the persistor if not found in memory.
- If the session is modified and the context block (`async with ...`) completes without errors, the session is saved via the persistor.
- `SessionManager.cleanup()` will attempt to save all currently active in-memory sessions before clearing the cache and closing the persistor (if applicable, e.g., an HTTP client).
- `SessionManager.cleanup_inactive_sessions()` will also save sessions to the persistor before removing them from memory.
```
