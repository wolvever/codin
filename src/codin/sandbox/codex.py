"""Codex sandbox implementation.

This module provides a CodexSandbox stub that is not yet implemented.
"""

from __future__ import annotations

import typing as _t
from pathlib import Path

from .base import Sandbox, ExecResult

__all__ = ["CodexSandbox"]


class CodexSandbox(Sandbox):
    """Codex CLI sandbox wrapper (TODO)."""

    def __init__(self, **_kwargs):
        super().__init__()
        raise NotImplementedError(
            "CodexSandbox is not yet implemented. Contributions welcome!"
        )

    async def _up(self) -> None:
        """Set up the Codex sandbox."""
        pass

    async def _down(self) -> None:
        """Clean up the Codex sandbox."""
        pass

    async def run_cmd(
        self,
        cmd: _t.Union[str, _t.Iterable[str]],
        *,
        cwd: _t.Optional[str] = None,
        timeout: _t.Optional[float] = None,
        env: _t.Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Execute a command in the Codex sandbox."""
        raise NotImplementedError("CodexSandbox is not yet implemented")

    async def run_code(
        self,
        code: _t.Optional[str] = None,
        *,
        file_path: _t.Optional[_t.Union[str, Path]] = None,
        language: str = "python",
        dependencies: _t.Optional[_t.List[str]] = None,
        timeout: _t.Optional[float] = None,
        env: _t.Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Execute code in the Codex sandbox."""
        raise NotImplementedError("CodexSandbox is not yet implemented")

    async def list_files(self, path: str = ".") -> _t.List[str]:
        """List files in the Codex sandbox."""
        raise NotImplementedError("CodexSandbox is not yet implemented")

    async def read_file(self, path: str) -> str:
        """Read a file from the Codex sandbox."""
        raise NotImplementedError("CodexSandbox is not yet implemented")

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the Codex sandbox."""
        raise NotImplementedError("CodexSandbox is not yet implemented") 