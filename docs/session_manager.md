# SessionManager Helper

`SessionManager` now exposes a small asynchronous context manager to simplify
working with short lived sessions. Use `async with session_manager.session(id)`
to create or retrieve a session and ensure it is cleaned up on exit.

```python
from codin.session import SessionManager

manager = SessionManager()
async with manager.session("demo") as session:
    # run agent logic with this session
    ...
# session is automatically closed
```

