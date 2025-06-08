"""Codex sandbox implementation for codin agents.

This module provides a sandbox implementation that integrates with
the Codex CLI for secure code execution and file operations.
"""

import platform
import shutil
import typing as _t
from pathlib import Path
import logging

from .local import LocalSandbox

from .base import ExecResult, ShellEnvironmentPolicy

logger = logging.getLogger(__name__)

__all__ = ['CodexSandbox']


class CodexSandbox(LocalSandbox):
    """Sandbox that routes commands through the Codex CLI sandbox helpers."""

    def __init__(self, workdir: str | None = None, *, codex_cmd: str = 'codex', env_policy: ShellEnvironmentPolicy | None = None, **_kwargs):
        super().__init__(workdir=workdir, env_policy=env_policy)
        self._codex_cmd = codex_cmd
        self._codex_available = shutil.which(codex_cmd) is not None
        self._cmd_prefix = ['debug', 'landlock'] if platform.system().lower() == 'linux' else ['debug', 'seatbelt']
        if not self._codex_available:
            logger.warning('Codex CLI not found – falling back to LocalSandbox behaviour')

    async def _up(self) -> None:
        """Set up the Codex sandbox."""

    async def _down(self) -> None:
        """Clean up the Codex sandbox."""

    async def run_cmd(
        self,
        cmd: str | _t.Iterable[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute a command in the Codex sandbox."""
        if not self._codex_available:
            # Fallback to LocalSandbox behaviour
            return await super().run_cmd(cmd, cwd=cwd, timeout=timeout, env=env)

        cmd_list = cmd if isinstance(cmd, list) or isinstance(cmd, tuple) else [cmd]
        full_cmd = [self._codex_cmd, *self._cmd_prefix, '--', *cmd_list]
        return await super().run_cmd(full_cmd, cwd=cwd, timeout=timeout, env=env)

    async def run_code(
        self,
        code: str | None = None,
        *,
        file_path: str | Path | None = None,
        language: str = 'python',
        dependencies: list[str] | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute code in the Codex sandbox."""
        return await super().run_code(
            code,
            file_path=file_path,
            language=language,
            dependencies=dependencies,
            timeout=timeout,
            env=env,
        )

    async def list_files(self, path: str = '.') -> list[str]:
        """List files in the Codex sandbox."""
        return await super().list_files(path)

    async def read_file(self, path: str) -> str:
        """Read a file from the Codex sandbox."""
        return await super().read_file(path)

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the Codex sandbox."""
        await super().write_file(path, content)
