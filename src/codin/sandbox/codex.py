"""Codex sandbox implementation for codin agents.

This module provides a sandbox implementation that integrates with
the Codex CLI for secure code execution and file operations.
"""

import logging
import platform
import shutil
import typing as _t
from pathlib import Path

from .base import ExecResult, ShellEnvironmentPolicy # Added ExecResult
from .local import LocalSandbox

logger = logging.getLogger(__name__)

__all__ = ['CodexSandbox']


class CodexSandbox(LocalSandbox):
    """Sandbox that routes commands through the Codex CLI sandbox helpers."""

    def __init__(
        self,
        workdir: str | None = None,
        *,
        codex_cmd: str = "codex",
        env_policy: ShellEnvironmentPolicy | None = None,
        **_kwargs: _t.Any,
    ) -> None:
        super().__init__(workdir=workdir, env_policy=env_policy)
        self._codex_cmd = codex_cmd
        self._codex_available = shutil.which(codex_cmd) is not None
        self._cmd_prefix = (
            ["debug", "landlock"]
            if platform.system().lower() == "linux"
            else ["debug", "seatbelt"]
        )
        if not self._codex_available:
            logger.warning(
                "Codex CLI not found – falling back to LocalSandbox behaviour"
            )

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
        # If the command is already a codex command, don't prefix it again.
        if cmd_list and cmd_list[0] == self._codex_cmd:
            full_cmd = cmd_list
        else:
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
        # This method from LocalSandbox already handles code execution appropriately.
        # If codex_available, its run_cmd will be used by underlying helpers,
        # effectively sandboxing the execution of interpreters (python, node, etc.).
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
        # LocalSandbox.list_files uses Python's os module, which is fine
        # as it operates on the mapped working directory.
        return await super().list_files(path)

    async def read_file(self, path: str) -> str:
        """Read a file from the Codex sandbox."""
        # LocalSandbox.read_file uses Python's open(), operating on mapped workdir.
        return await super().read_file(path)

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the Codex sandbox."""
        # LocalSandbox.write_file uses Python's open(), operating on mapped workdir.
        await super().write_file(path, content)

    async def list_available_binaries(self) -> list[str]:
        # This is a common way to list commands, but might need adjustment
        # based on the specific shell and available tools in the Codex environment.
        # It tries 'compgen -c' (bash), then 'ls' on common bin dirs.
        commands_to_try = [
            "bash -c 'compgen -c'", # Bash specific, lists all commands
            "sh -c 'ls /bin /usr/bin /usr/local/bin 2>/dev/null | sort -u'", # More generic
        ]
        for cmd_str in commands_to_try:
            try:
                # Use self.run_cmd to ensure commands are run within the codex sandbox if available
                # self.run_cmd itself prepends the codex command and prefix.
                # We pass the command string directly to run_cmd, which expects str | Iterable[str].
                # LocalSandbox's _prepare_command will handle shell execution.
                result = await self.run_cmd(cmd_str)

                if result.exit_code == 0 and result.stdout:
                    return sorted(list(set(result.stdout.strip().split('\n'))))
            except Exception:
                # Try the next command if one fails
                pass
        return [] # Return empty if no command succeeded
