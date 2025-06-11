"""
The `codin.replay` module provides a system for recording message exchanges
(e.g., between a client and an agent) for later analysis, debugging, or replay.
It's designed to be flexible, allowing different backends for storing replay data.

Core Components:
---------------

`BaseReplay`:
  - An abstract base class defining the interface for all replay backends.
  - Key method: `async def record_message_exchange(self, client_message: Any, agent_message: Any, session_id: str, **kwargs) -> None`
  - Provides a helper `_serialize_message(message: Any)` for converting messages
    (e.g., Pydantic models) into JSON-serializable formats. Subclasses can override this.

Available Backends:
------------------

`FileReplay(BaseReplay)`:
  - Purpose: Saves message exchanges to a session-specific JSONL (JSON Lines) file.
  - Instantiation:
    ```python
    from pathlib import Path
    from codin.replay import FileReplay

    replay_backend = FileReplay(
        session_id="my_unique_session_id",
        base_dir=Path("/custom/replay/logs/path") # Optional
    )
    ```
  - `session_id`: A unique identifier for the session. This is crucial for organizing logs.
  - `base_dir`: The directory where session log files will be stored.
    Defaults to `~/.codin/sessions/`. The directory will be created if it doesn't exist.
    Each session will have its own `.jsonl` file within this directory, named using the
    session ID and a timestamp (e.g., `replay-my_unique_session_id-YYYY-MM-DDTHH-MM-SS.jsonl`).
  - Each line in the file is a JSON object representing either a "session_start" meta event
    or a "message_exchange" event (or a custom `exchange_type` like "client_request").

`HttpReplay(BaseReplay)` (Conceptual):
  - Purpose: Sends message exchanges to a specified HTTP/S endpoint.
  - This is currently a conceptual design and would require a running HTTP server
    configured to receive the replay data.
  - Instantiation:
    ```python
    from codin.replay import HttpReplay

    replay_backend = HttpReplay(
        session_id="my_unique_session_id",
        endpoint_url="https://your-replay-server.com/api/v1/record"
        # client_session: Optional aiohttp.ClientSession can also be passed
    )
    ```
  - `session_id`: Unique identifier for the session.
  - `endpoint_url`: The URL where the replay data (as JSON) will be POSTed.

Common Usage:
-------------

1. Initialization:
   Instantiate your chosen replay backend, providing at least a `session_id`.
   (Typically done via the replay factory, see Integration Example).

   ```python
   # Example with FileReplay, often created by a factory:
   # replay = FileReplay(session_id="run_123")
   ```

2. Recording Messages:
   Use the `record_message_exchange` method to log a pair of messages.
   The `exchange_type` kwarg can be used to specify the type of event.

   ```python
   # client_msg = {"text": "Hello agent", "user_id": "user_x"}
   # agent_msg = {"text": "Hello user_x, how can I help?", "confidence": 0.9}
   # session_id_for_log = "run_123" # Provided by AgentRunner

   # await replay_instance.record_message_exchange(
   #     client_message=client_msg,
   #     agent_message=agent_msg,
   #     session_id=session_id_for_log,
   #     exchange_type='custom_event_type', # Optional, defaults to 'message_exchange'
   #     turn_number=1,
   #     environment="test"
   # )
   ```
   - The `session_id` is managed and passed by `AgentRunner`.
   - `_serialize_message` is used internally.

3. Cleanup:
   Call `cleanup()` on the replay instance when a session ends.
   (Typically handled by `LocalDispatcher`, see Integration Example).

   ```python
   # await replay_instance.cleanup()
   ```

Integration Example (using Replay Factory):
-------------------------------------------
The replay system is typically integrated starting from the API layer, which provides
a factory for creating replay instances. This factory is then used by `LocalDispatcher`
to create and manage the lifecycle of the replay backend for a specific run.

1. API Layer (e.g., `src/codin/api/app.py`):
   - When a request is received (e.g., at the `/v1/submit` endpoint), the API layer
     can define a factory function (e.g., a lambda) for creating a specific
     replay backend instance (like `FileReplay`).
   - This factory is added to the `a2a_request` dictionary.
     ```python
     # In an API endpoint function (e.g., in src/codin/api/app.py)
     # from ..replay import FileReplay # Make sure FileReplay or other backends are imported
     # req: SubmitRequest (FastAPI request model)

     # Define a factory that takes a runner_id string and returns a BaseReplay instance
     req.a2a_request['replay_factory'] = lambda runner_id_str: FileReplay(session_id=runner_id_str)

     # This modified req.a2a_request is then passed to dispatcher.submit()
     # runner_id = await dispatcher.submit(req.a2a_request)
     ```

2. `LocalDispatcher.submit`:
   - Retrieves the `replay_factory` from the `a2a_request`.
   - After generating a unique `runner_id` for the request, it calls this factory
     with the `runner_id` to obtain a `replay_instance`. If no factory is provided
     or it's invalid, `replay_instance` will be `None`.
     ```python
     # In LocalDispatcher.submit(self, a2a_request: dict)
     # runner_id = f'run_{request_id[:8]}' # Generated runner_id

     replay_factory = a2a_request.get('replay_factory')
     replay_instance = None
     if replay_factory and callable(replay_factory):
         replay_instance = replay_factory(runner_id)

     # This replay_instance (which can be None) is then passed to _handle_request
     # task = asyncio.create_task(
     #    self._handle_request(dispatch_request, result, stream_queue, replay_instance)
     # )
     ```

3. `LocalDispatcher._handle_request`:
   - This method receives the `replay_instance` (which could be `None`) as an argument.
     ```python
     # async def _handle_request(self, ..., replay_instance: Optional[BaseReplay] = None):
     ```
   - It then passes this `replay_instance` to the `AgentRunner`'s constructor.
     ```python
     # In LocalDispatcher._handle_request, when creating AgentRunners:
     # runner_group.add_runner(
     #    AgentRunner(agent, replay_backend=replay_instance)
     # )
     ```
   - Crucially, `_handle_request` ensures that `replay_instance.cleanup()` is called
     in its `finally` block, managing the lifecycle of the replay backend for the
     duration of the request handling.
     ```python
     # In LocalDispatcher._handle_request's finally block:
     # finally:
     #     if replay_instance:
     #         await replay_instance.cleanup()
     ```

4. `AgentRunner` Usage:
   - `AgentRunner` accepts the `replay_backend` in its constructor and uses it to
     record the initial client request and subsequent agent messages, as detailed
     in `AgentRunner`'s documentation or the `AgentRunner` section of "Testing Suggestions".
     (e.g., calling `self.replay_backend.record_message_exchange(...)`).

This factory-based approach decouples `LocalDispatcher` from concrete replay
implementations, allowing the calling layer (e.g., API) to decide if and how
replays are recorded for a given request.

This module aims to provide a straightforward yet extensible way to add
replay capabilities to different parts of the codin system.

Testing Suggestions:
--------------------

Unit tests are crucial for ensuring the reliability of the replay system.
Consider the following areas for testing:

`FileReplay` (and other `BaseReplay` implementations):
  - Initialization:
    - Test with and without `base_dir` (check default path creation).
    - Test directory creation if it doesn't exist.
  - Message Recording:
    - Record a single message exchange and verify the content of the `.jsonl` file.
      Check timestamp format, `session_id`, client/agent messages, `exchange_type`, and any kwargs.
    - Verify the "session_start" metadata is the first line.
    - Test recording with custom `exchange_type` (e.g., 'client_request').
  - File Handling:
    - Correct file naming convention (session_id, timestamp).
    - `cleanup()` method: ensure file is closed, queue is processed, and task is awaited.
    - Behavior when multiple messages are recorded (appended to the same file).
  - Edge Cases:
    - Empty messages, messages with special characters.

`HttpReplay` (Conceptual Tests - Adapt if/when implemented):
  - Payload Construction, HTTP Interaction (mocking), Error Handling, Session Management.
    (Details as previously listed).

`BaseReplay._serialize_message`:
  - Test with various data types: Pydantic models, standard types (`str`, `int`, `None`, etc.),
    custom objects (fallback to `str()`), nested structures.
    (Details as previously listed).

API Layer (e.g., `src/codin/api/app.py`):
  - Test that the API endpoint (e.g., `/v1/submit`) correctly constructs and adds the
    `replay_factory` to the `a2a_request` dictionary passed to `dispatcher.submit`.
  - Ensure the factory, when called, produces the intended `BaseReplay` instance (e.g., `FileReplay`
    correctly configured with a `session_id`).

`LocalDispatcher` Integration:
  - Test `submit` method:
    - Verify it correctly retrieves `replay_factory` from `a2a_request`.
    - If a valid factory is provided, ensure it's called with the correct `runner_id`.
    - If no factory or an invalid one is provided, ensure `replay_instance` becomes `None`.
    - Confirm the resulting `replay_instance` (or `None`) is passed to `_handle_request`.
  - Test `_handle_request` method (lifecycle management):
    - Ensure it correctly passes the received `replay_instance` to `AgentRunner`.
    - Confirm `replay_instance.cleanup()` is called in the `finally` block if an instance was provided,
      even if errors occur in the `try` block.

`AgentRunner` Integration:
  - Test that `AgentRunner` correctly calls `replay_backend.record_message_exchange`:
    - Once with `agent_message=None` and `exchange_type='client_request'` before agent processing.
    - For each agent output during processing, with the correct `client_message` and `agent_message`.
  - Verify that `session_id` is correctly passed.
  - Test behavior when `replay_backend` is `None` (should not attempt to record or fail).
"""

from .base import BaseReplay
from .file import FileReplay
from .http import HttpReplay

__all__ = [
    "BaseReplay",
    "FileReplay",
    "HttpReplay",
]
