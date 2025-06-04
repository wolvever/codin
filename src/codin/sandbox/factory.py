"""Sandbox factory for creating sandbox instances.

This module provides a factory function to create sandbox instances by backend name.
"""

from __future__ import annotations

import typing as _t

from .base import Sandbox
from .local import LocalSandbox
from .e2b import E2BSandbox
from .daytona import DaytonaSandbox
from .codex import CodexSandbox

__all__ = ["create_sandbox"]

# Registry of available sandbox backends
_SANDBOXES: _t.Dict[str, _t.Type[Sandbox]] = {
    "local": LocalSandbox,
    "e2b": E2BSandbox,
    "daytona": DaytonaSandbox,
    "codex": CodexSandbox,
}


async def create_sandbox(backend: str = "local", **kwargs) -> Sandbox:
    """Instantiate and initialize a sandbox by backend keyword.

    Args:
        backend: The sandbox backend to use ("local", "e2b", "daytona", "codex")
        **kwargs: Additional arguments to pass to the sandbox constructor

    Returns:
        An initialized sandbox instance

    Raises:
        ValueError: If the backend is not recognized
        RuntimeError: If the sandbox fails to initialize

    Example:
        >>> sandbox = await create_sandbox("local", workdir="/tmp/sandbox")
        >>> result = await sandbox.run_cmd("echo hello")
        >>> print(result.stdout)
        hello
        >>> await sandbox.down()
    """
    try:
        cls = _SANDBOXES[backend]
    except KeyError as e:
        raise ValueError(f"Unknown sandbox backend: {backend!r}. Available: {list(_SANDBOXES.keys())}") from e
    
    sandbox = cls(**kwargs)
    await sandbox.up()  # Initialize the sandbox
    return sandbox 