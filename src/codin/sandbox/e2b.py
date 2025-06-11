"""E2B sandbox implementation for codin agents.

This module provides a sandbox implementation that integrates with
E2B cloud sandboxes for secure remote code execution.
"""

import asyncio
import shlex
import tempfile
import typing as _t
import zipfile
from pathlib import Path
import os # For path manipulation

from .base import ExecResult, Sandbox, ShellEnvironmentPolicy
from .common_exec import CommonCodeExecutionMixin # Added import

__all__ = ['E2BSandbox']


class E2BSandbox(Sandbox, CommonCodeExecutionMixin): # Inherit from mixin
    """Adapter around E2B cloud sandbox SDK."""

    def __init__(self, *, env_policy: ShellEnvironmentPolicy | None = None, **kwargs):
        super().__init__(env_policy=env_policy)
        try:
            from e2b import Sandbox as _E2B_SDK
            from e2b.sandbox.main import ProcessMessage # For type hint
        except ImportError as e:  # pragma: no cover
            raise RuntimeError('e2b package not available – install with `pip install e2b`.') from e

        self._E2BProcessMessage: _t.Type[ProcessMessage] = ProcessMessage # Store for type hinting

        self._sandbox = _E2B_SDK(**kwargs) # e2b.Sandbox instance

    # _get_language_executor, _get_main_file_for_language, _install_dependencies
    # are now inherited from CommonCodeExecutionMixin.
    # The mixin's defaults (e.g. python3) align with E2B's typical environment.

    # Lifecycle ------------------------------------------------------------
    async def _up(self) -> None:
        """Set up the E2B sandbox."""
        # E2B sandbox is typically started on instantiation or first command.
        # If specific up command is needed, it would be here.
        # For now, assuming it's managed by the E2B SDK lifecycle.
        pass

    async def _down(self) -> None:
        """Clean up the E2B sandbox."""
        loop = asyncio.get_event_loop()
        # E2B SDK's close() is the method to terminate/clean up the sandbox.
        await loop.run_in_executor(None, self._sandbox.close)


    # Exec -----------------------------------------------------------------
    async def run_cmd(
        self,
        cmd: str | _t.Iterable[str],
        *,
        cwd: str | None = None, # E2B SDK expects absolute paths for cwd
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute a command in the E2B sandbox."""
        loop = asyncio.get_event_loop()

        cmd_str = cmd if isinstance(cmd, str) else ' '.join(shlex.quote(c) for c in cmd)

        # E2B's cwd defaults to /home/user. If providing, ensure it's absolute.
        effective_cwd = cwd if cwd else "/home/user" # Default or specified cwd

        def _exec_sync():
            # E2B SDK's process.start() and awaiting output.
            proc = self._sandbox.process.start(
                cmd_str,
                cwd=effective_cwd,
                env_vars=self._prepare_env(env), # from Sandbox base
                timeout_s=timeout,
            )
            proc.wait() # Wait for the process to complete

            stdout = "".join(msg.line for msg in proc.output.stdout_messages)
            stderr = "".join(msg.line for msg in proc.output.stderr_messages)


            return ExecResult(stdout, stderr, proc.exit_code if proc.exit_code is not None else -1)

        try:
            return await loop.run_in_executor(None, _exec_sync)
        except Exception as e: # Catch exceptions from E2B SDK calls
             return ExecResult(stdout="", stderr=f"E2B command execution failed: {e}", exit_code=127)


    async def run_code(
        self,
        code: str | None = None,
        *,
        file_path: str | Path | None = None, # Local path to file/dir/zip
        language: str = 'python',
        dependencies: list[str] | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute code in the E2B sandbox."""
        if not code and not file_path:
            return ExecResult('', 'Either code or file_path must be provided', 1)
        if code and file_path:
            return ExecResult('', 'Cannot provide both code and file_path', 1)

        # 1. Install dependencies using the inherited _install_dependencies
        if dependencies:
            # This will call self.run_cmd, using E2B's execution. Cwd will be /home/user by default.
            dep_result = await self._install_dependencies(dependencies, language)
            if dep_result.exit_code != 0:
                return ExecResult(
                    dep_result.stdout, f'Failed to install dependencies: {dep_result.stderr}', dep_result.exit_code
                )

        # 2. Handle direct code execution
        if code:
            # _common_run_code_logic will use self.write_file to write to a path like /tmp/code_HASH.ext
            # E2B's self.write_file handles absolute paths correctly.
            return await self._common_run_code_logic(
                code=code, file_path=None, language=language,
                dependencies=None, # Already handled
                timeout=timeout, env=env
            )

        # 3. Handle file path execution (uploading local files/zips/dirs to E2B)
        if file_path:
            local_source_path = Path(file_path) # Path on the system running this agent
            if not local_source_path.is_absolute(): # Ensure path is absolute for clarity
                 local_source_path = Path(os.getcwd()) / local_source_path

            if not local_source_path.exists():
                return ExecResult('', f'Local file/directory not found: {local_source_path}', 1)

            # Define a base directory in E2B for staging. /home/user is common.
            # Create a unique subdirectory within /home/user for this execution.
            remote_base_dir = "/home/user"
            remote_staging_dir_name = f"codin_run_{os.urandom(4).hex()}"
            # Full absolute path for the staging directory in E2B
            e2b_staging_abs_path = str(Path(remote_base_dir) / remote_staging_dir_name)

            # E2B's filesystem commands create parent directories if they don't exist.
            # So, no explicit mkdir needed for e2b_staging_abs_path before writing files into it.

            staged_files_relative_to_remote_dir: list[str] = []

            try:
                if local_source_path.is_file():
                    if local_source_path.suffix == '.zip':
                        with tempfile.TemporaryDirectory() as local_temp_extract_dir:
                            local_extract_path = Path(local_temp_extract_dir)
                            with zipfile.ZipFile(local_source_path, 'r') as zip_ref:
                                zip_ref.extractall(local_extract_path)

                            for item in local_extract_path.rglob('*'):
                                if item.is_file():
                                    relative_path_in_zip = item.relative_to(local_extract_path)
                                    # E2B paths should be absolute or relative to /home/user
                                    e2b_target_abs_path = Path(e2b_staging_abs_path) / relative_path_in_zip
                                    await self.write_file(str(e2b_target_abs_path), item.read_text(encoding='utf-8'))
                                    staged_files_relative_to_remote_dir.append(str(relative_path_in_zip))
                    else: # Single file
                        e2b_target_abs_path = Path(e2b_staging_abs_path) / local_source_path.name
                        await self.write_file(str(e2b_target_abs_path), local_source_path.read_text(encoding='utf-8'))
                        staged_files_relative_to_remote_dir.append(local_source_path.name)

                elif local_source_path.is_dir():
                    for item in local_source_path.rglob('*'):
                        if item.is_file():
                            relative_path_in_dir = item.relative_to(local_source_path)
                            e2b_target_abs_path = Path(e2b_staging_abs_path) / relative_path_in_dir
                            await self.write_file(str(e2b_target_abs_path), item.read_text(encoding='utf-8'))
                            staged_files_relative_to_remote_dir.append(str(relative_path_in_dir))
                else:
                    return ExecResult('', f'Unsupported local path type: {local_source_path}', 1)

                if not staged_files_relative_to_remote_dir:
                     return ExecResult('', f'No files found to execute for path: {local_source_path}', 1)

                main_file_in_remote_dir = self._get_main_file_for_language(language, staged_files_relative_to_remote_dir)
                if not main_file_in_remote_dir:
                    return ExecResult('', f'No main file found for lang {language} in uploaded content from {local_source_path.name}', 1)

                # This is the absolute path to the main executable file within E2B.
                final_exec_path_in_e2b = str(Path(e2b_staging_abs_path) / main_file_in_remote_dir)

                # _common_run_code_logic will derive cwd from final_exec_path_in_e2b's parent.
                # This derived cwd will be an absolute path in E2B.
                return await self._common_run_code_logic(
                    code=None,
                    file_path=final_exec_path_in_e2b, # Absolute path in E2B
                    language=language,
                    dependencies=None, # Already handled
                    timeout=timeout,
                    env=env
                )
            except Exception as e: # Catch errors during file prep/upload
                return ExecResult('', f"Error preparing/uploading files to E2B for execution: {e}", 1)

        return ExecResult('', 'Internal error in E2B run_code dispatch', 1)


    # Filesystem -----------------------------------------------------------
    async def list_files(self, path: str = '.') -> list[str]:
        """List files in the E2B sandbox. Path is absolute or relative to /home/user."""
        loop = asyncio.get_event_loop()
        # E2B SDK's list method path is typically relative to /home/user or an absolute path.
        # If path is '.', it lists /home/user.
        # For consistency, ensure path is treated as absolute or relative to a known root like /home/user.
        # The mixin doesn't impose a structure, so this method adapts.
        # If path starts with '/', assume absolute. Otherwise, relative to /home/user.
        query_path = path
        if not path.startswith('/'):
            query_path = str(Path("/home/user") / path) # Default to /home/user if relative

        def _list_files_sync():
            try:
                # E2B SDK returns FileInfo objects
                return [fi.path for fi in self._sandbox.filesystem.list(query_path)]
            except Exception as e: # Catch E2B specific errors if any
                # print(f"E2B list_files error for path '{query_path}': {e}")
                return [] # Return empty list on error, or re-raise

        return await loop.run_in_executor(None, _list_files_sync)

    async def read_file(self, path: str) -> str:
        """Read a file from the E2B sandbox. Path is absolute or relative to /home/user."""
        loop = asyncio.get_event_loop()

        query_path = path
        if not path.startswith('/'): # Assuming relative paths are to /home/user
            query_path = str(Path("/home/user") / path)

        async def _read_file_async_wrapper(): # E2B SDK's read is sync
            try:
                return self._sandbox.filesystem.read(query_path)
            except Exception as e: # Catch E2B specific errors (e.g., file not found)
                # Map to FileNotFoundError for consistency if possible
                # This requires knowledge of E2B's specific exception types
                raise FileNotFoundError(f"File not found or unreadable in E2B at '{query_path}': {e}") from e

        try:
            return await loop.run_in_executor(None, _read_file_async_wrapper)
        except FileNotFoundError: # Re-raise to ensure it's from this async context
            raise


    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the E2B sandbox. Path is absolute or relative to /home/user."""
        loop = asyncio.get_event_loop()

        # E2B SDK's write takes an absolute path.
        # If path for `_common_run_code_logic` (e.g. /tmp/code_...) is absolute, it's used as is.
        # If it's relative, it needs to be based on a known E2B root.
        # The mixin's _common_run_code_logic uses `temp_file_name = f"temp_code_{hash(code) % 10000}{ext}"`
        # for temp scripts. This needs to be an absolute path for E2B.

        effective_path = path
        if not path.startswith('/'):
            # If _common_run_code_logic passes "temp_code.js", make it "/tmp/temp_code.js"
            if "temp_code_" in path and Path(path).parent == Path("."): # Heuristic for mixin's temp files
                 effective_path = str(Path("/tmp") / path)
            else: # Otherwise, assume relative to /home/user for general writes
                 effective_path = str(Path("/home/user") / path)

        async def _write_file_async_wrapper():
            try:
                # E2B SDK's write_bytes might be more robust if content can be binary
                # For string content, `write` should be fine.
                # It creates parent directories if they don't exist.
                self._sandbox.filesystem.write(effective_path, content)
            except Exception as e:
                raise IOError(f"Failed to write file to E2B at '{effective_path}': {e}") from e

        try:
            await loop.run_in_executor(None, _write_file_async_wrapper)
        except IOError: # Re-raise
            raise

    async def list_available_binaries(self) -> list[str]:
        # Similar approach to CodexSandbox, adjust as needed for E2B's environment
        commands_to_try = [
            "bash -c 'compgen -c'",
            "sh -c 'ls /bin /usr/bin /usr/local/bin /sbin /usr/sbin 2>/dev/null | sort -u'",
        ]
        for cmd_str in commands_to_try:
            try:
                # Using self.run_cmd which is already implemented for E2B
                result = await self.run_cmd(cmd_str)

                if result.exit_code == 0 and result.stdout:
                    return sorted(list(set(result.stdout.strip().split('\n'))))
            except Exception:
                # Try the next command if one fails
                pass
        return []
