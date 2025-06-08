"""Local sandbox implementation for codin agents.

This module provides a sandbox implementation that executes code locally
using subprocess for development and testing purposes.

**Warning** – this is not isolated! Only use for trusted inputs or when
combined with additional OS-level sandboxing (e.g. containers).
"""

import asyncio
import os
import platform
import subprocess
import tempfile
import typing as _t
import zipfile
from pathlib import Path
import shutil # Added for copytree and copy2

from .base import ExecResult, Sandbox, ShellEnvironmentPolicy
from .common_exec import CommonCodeExecutionMixin # Added import

__all__ = ['LocalSandbox']


class LocalSandbox(Sandbox, CommonCodeExecutionMixin): # Inherit from mixin
    """Execute commands directly on the host using subprocess.

    **Warning** – this is not isolated! Only use for trusted inputs or when
    combined with additional OS-level sandboxing (e.g. containers).

    Enhanced to support both Windows PowerShell and Unix bash automatically.
    """

    def __init__(self, workdir: str | None = None, *, env_policy: ShellEnvironmentPolicy | None = None):
        super().__init__(env_policy=env_policy)
        self._workdir = workdir or os.getcwd()
        self.__is_windows = platform.system().lower() == 'windows' # Renamed for clarity
        self._shell_executable = self._detect_shell()

    @property
    def _is_windows(self) -> bool: # Property for mixin access
        return self.__is_windows

    def _detect_shell(self) -> str:
        """Detect the best shell to use for this platform."""
        if self.__is_windows:
            return 'powershell.exe'
        return '/bin/bash'

    def _prepare_command(self, cmd: str | _t.Iterable[str]) -> tuple[str | list[str], bool]:
        """Prepare command for execution, handling cross-platform differences."""
        if isinstance(cmd, str):
            if self.__is_windows:
                return ['powershell.exe', '-NoProfile', '-Command', cmd], False
            return cmd, True # For bash, use shell=True
        return list(cmd), False

    # _get_main_file_for_language is inherited from CommonCodeExecutionMixin
    # _install_dependencies is inherited from CommonCodeExecutionMixin

    def _get_language_executor(self, language: str) -> list[str]:
        """Get the command to execute code for a given language."""
        lang_lower = language.lower()

        if lang_lower in ('python', 'py'):
            return ['python']  # LocalSandbox specific: uses 'python'
        if lang_lower in ('javascript', 'js', 'node'):
            return ['node']
        if lang_lower in ('bash', 'sh'):
            # Use PowerShell for bash commands on Windows, consistent with original LocalSandbox
            return ['powershell.exe', '-Command'] if self.__is_windows else ['/bin/bash']
        if lang_lower in ('powershell', 'ps1'):
            return ['powershell.exe', '-Command']
        if lang_lower == 'go':
            return ['go', 'run']
        if lang_lower == 'rust':
            return ['cargo', 'run', '--']
        if lang_lower in ('java',):
            return ['java']
        if lang_lower in ('c', 'cpp', 'c++'):
            # Compilation is handled within run_code for LocalSandbox
            raise NotImplementedError(f'Language {language} requires compilation handled by run_code.')
        # Fallback to mixin's definition or its ValueError for unsupported languages
        return super()._get_language_executor(language)

    # Lifecycle ------------------------------------------------------------
    async def _up(self) -> None:
        """Set up the local sandbox."""
        # Ensure working directory exists
        os.makedirs(self._workdir, exist_ok=True)

    async def _down(self) -> None:
        """Clean up the local sandbox."""

    # Exec -----------------------------------------------------------------
    async def run_cmd(
        self,
        cmd: str | _t.Iterable[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute a command in the sandbox."""
        args, shell = self._prepare_command(cmd)

        # Use run in executor to prevent blocking the event loop
        loop = asyncio.get_event_loop()

        def _run_subprocess():
            # Set up environment with UTF-8 encoding for Windows
            subprocess_env = self._prepare_env(env)
            if self.__is_windows: # Use renamed internal variable
                # Force UTF-8 encoding on Windows
                subprocess_env['PYTHONIOENCODING'] = 'utf-8'

            effective_cwd = cwd or self._workdir # Ensure cwd is used

            # For Windows PowerShell commands, we need special handling
            if self.__is_windows and isinstance(args, list) and args[0] == 'powershell.exe':
                proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=effective_cwd,
                    env=subprocess_env,
                    shell=False,  # Don't use shell=True with PowerShell args
                    encoding='utf-8',
                    errors='replace',
                )
            else:
                proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=effective_cwd,
                    env=subprocess_env,
                    shell=shell,
                    encoding='utf-8',
                    errors='replace',
                )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                return ExecResult(stdout=stdout, stderr=stderr, exit_code=proc.returncode)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                return ExecResult(stdout=stdout, stderr=stderr, exit_code=-1)

        # Run the blocking operation in a thread pool
        return await loop.run_in_executor(None, _run_subprocess)

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
        """Execute code in the sandbox."""
        if not code and not file_path:
            return ExecResult(stdout='', stderr='Either code or file_path must be provided', exit_code=1)
        if code and file_path:
            return ExecResult(stdout='', stderr='Cannot provide both code and file_path', exit_code=1)

        # 1. Install dependencies using the inherited _install_dependencies
        if dependencies:
            # _install_dependencies calls self.run_cmd, which uses self._workdir as default cwd.
            dep_result = await self._install_dependencies(dependencies, language)
            if dep_result.exit_code != 0:
                return ExecResult(
                    stdout=dep_result.stdout,
                    stderr=f'Failed to install dependencies: {dep_result.stderr}',
                    exit_code=dep_result.exit_code,
                )

        # 2. Handle direct code execution
        if code:
            lang_lower = language.lower()
            if lang_lower in ('c', 'cpp', 'c++'):
                ext = '.c' if lang_lower == 'c' else '.cpp'
                # Create temp dir within self._workdir for C/C++ compilation
                with tempfile.TemporaryDirectory(dir=self._workdir) as tmpdir_str:
                    tmpdir = Path(tmpdir_str)
                    src_file = tmpdir / f'code{ext}'
                    src_file.write_text(code, encoding='utf-8')
                    exe_name = 'a.out' if not self.__is_windows else 'a.exe'
                    exe_file = tmpdir / exe_name
                    compiler = 'gcc' if lang_lower == 'c' else 'g++'
                    compile_cmd = [compiler, str(src_file), '-o', str(exe_file)]
                    if lang_lower in ('cpp', 'c++'): compile_cmd.extend(['-std=c++17'])

                    compile_result = await self.run_cmd(compile_cmd, cwd=str(tmpdir), timeout=timeout, env=env)
                    if compile_result.exit_code != 0:
                        return ExecResult(stdout=compile_result.stdout,
                                          stderr=f'Compilation failed: {compile_result.stderr}',
                                          exit_code=compile_result.exit_code)
                    return await self.run_cmd([str(exe_file)], cwd=str(tmpdir), timeout=timeout, env=env)

            # For other languages, delegate to common logic.
            # self.write_file (called by mixin if needed for temp script) will place files in self._workdir.
            return await self._common_run_code_logic(code=code, file_path=None, language=language,
                                                     dependencies=None, # Already handled
                                                     timeout=timeout, env=env)

        # 3. Handle file path execution
        if file_path:
            source_path = Path(file_path) # User-provided path, could be relative or absolute

            # Before checking existence, if source_path is relative, resolve it against CWD or a sensible default.
            # Assuming source_path if relative, is relative to current os.getcwd() or similar context.
            # For robustness, absolute paths are preferred for file_path argument if ambiguity exists.
            if not source_path.is_absolute():
                 source_path = Path(os.getcwd()) / source_path # Or some other defined base for relative paths

            if not source_path.exists():
                return ExecResult(stdout='', stderr=f'File or directory not found: {source_path}', exit_code=1)

            # Create a temporary directory within self._workdir to stage and run files
            with tempfile.TemporaryDirectory(dir=self._workdir) as temp_exec_dir_str:
                temp_exec_dir = Path(temp_exec_dir_str)

                target_exec_file_abs: Path # Absolute path to the file to run, inside temp_exec_dir
                execution_cwd_abs: Path    # Absolute CWD for the command, inside temp_exec_dir

                if source_path.is_file():
                    if source_path.suffix == '.zip':
                        try:
                            with zipfile.ZipFile(source_path, 'r') as zipf:
                                zipf.extractall(temp_exec_dir)
                            # List files relative to temp_exec_dir for main file detection
                            extracted_files = [str(p.relative_to(temp_exec_dir)) for p in temp_exec_dir.rglob('*') if p.is_file()]
                            main_file_rel = self._get_main_file_for_language(language, extracted_files)
                            if not main_file_rel:
                                return ExecResult(stdout='', stderr=f'No main file found for {language} in zip {source_path.name}', exit_code=1)
                            target_exec_file_abs = temp_exec_dir / main_file_rel
                            execution_cwd_abs = target_exec_file_abs.parent
                        except zipfile.BadZipFile:
                            return ExecResult(stdout='', stderr=f'Invalid zip file: {source_path}', exit_code=1)
                    else: # Single regular file
                        # Copy the file into temp_exec_dir
                        target_exec_file_abs = temp_exec_dir / source_path.name
                        shutil.copy2(source_path, target_exec_file_abs) # Use shutil.copy2 to preserve metadata
                        execution_cwd_abs = temp_exec_dir
                elif source_path.is_dir():
                    # Copy contents of source_path directory into temp_exec_dir
                    for item in source_path.iterdir(): # Iterate over items in source_path
                        dest_path = temp_exec_dir / item.name
                        if item.is_dir():
                            shutil.copytree(item, dest_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, dest_path) # Copy file

                    # List files relative to the new root (temp_exec_dir) for main file detection
                    copied_files = [str(p.relative_to(temp_exec_dir)) for p in temp_exec_dir.rglob('*') if p.is_file()]
                    main_file_rel = self._get_main_file_for_language(language, copied_files)
                    if not main_file_rel:
                        return ExecResult(stdout='', stderr=f'No main file found for {language} in dir {source_path.name}', exit_code=1)
                    target_exec_file_abs = temp_exec_dir / main_file_rel
                    execution_cwd_abs = target_exec_file_abs.parent
                else:
                    return ExecResult(stdout='', stderr=f'Unsupported path type: {source_path}', exit_code=1)

                # Now, target_exec_file_abs and execution_cwd_abs are set.
                # These are absolute paths within the temp_exec_dir (which is itself in self._workdir).

                lang_lower = language.lower()
                if lang_lower in ('c', 'cpp', 'c++'):
                    exe_name = 'a.out' if not self.__is_windows else 'a.exe'
                    # Place executable in the execution_cwd_abs (e.g., alongside main source file)
                    exe_file = execution_cwd_abs / exe_name
                    compiler = 'gcc' if lang_lower == 'c' else 'g++'
                    compile_cmd = [compiler, str(target_exec_file_abs), '-o', str(exe_file)]
                    if lang_lower in ('cpp', 'c++'): compile_cmd.extend(['-std=c++17'])

                    compile_result = await self.run_cmd(compile_cmd, cwd=str(execution_cwd_abs), timeout=timeout, env=env)
                    if compile_result.exit_code != 0:
                        return ExecResult(stdout=compile_result.stdout,
                                          stderr=f'Compilation failed: {compile_result.stderr}',
                                          exit_code=compile_result.exit_code)
                    return await self.run_cmd([str(exe_file)], cwd=str(execution_cwd_abs), timeout=timeout, env=env)

                # For other languages, call _common_run_code_logic.
                # Pass the absolute path (target_exec_file_abs) within the temp sandbox dir.
                # The _common_run_code_logic's run_cmd will use its parent as cwd if not otherwise specified.
                # However, our _common_run_code_logic sets cwd based on file_path.parent.
                # So, it will use execution_cwd_abs correctly.
                return await self._common_run_code_logic(
                    code=None,
                    file_path=str(target_exec_file_abs), # This is an absolute path for run_cmd
                    language=language,
                    dependencies=None, # Already handled
                    timeout=timeout,
                    env=env
                )

        # Fallback if neither code nor file_path is provided (should be caught by initial checks)
        return ExecResult(stdout='', stderr='Internal error in run_code logic: No path or code provided.', exit_code=1)


    # Filesystem -----------------------------------------------------------
    def _abs(self, path: str) -> str:
        """Get absolute path within the sandbox working directory, ensuring it's safe."""
        resolved_workdir = Path(self._workdir).resolve()

        # Treat 'path' as relative to 'resolved_workdir'
        # Path.joinpath can handle if 'path' is already absolute, but behavior might be unexpected.
        # Normalizing 'path' first helps simplify ".." etc.
        normalized_path_segment = Path(path).normalize()

        # If normalized_path_segment is absolute, it might try to escape.
        # This check is tricky. For now, we assume 'path' is intended to be relative.
        # If 'path' could be absolute and outside 'resolved_workdir', it should be an error.
        if normalized_path_segment.is_absolute():
            # This means 'path' is an absolute path. We must check if it's within workdir.
            # This scenario is less common if 'path' is always meant to be relative to sandbox root.
            abs_path_candidate = normalized_path_segment.resolve()
        else:
            abs_path_candidate = (resolved_workdir / normalized_path_segment).resolve()

        # Final check: is the resolved path within the working directory?
        # Using os.path.commonprefix for a string-based check after resolving,
        # or Path.is_relative_to for Python 3.9+
        try: # Python 3.9+
            if not abs_path_candidate.is_relative_to(resolved_workdir):
                raise ValueError(f"Attempted to access path '{abs_path_candidate}' outside of working directory '{resolved_workdir}'.")
        except AttributeError: # Fallback for Python < 3.9
            if os.path.commonprefix([str(resolved_workdir), str(abs_path_candidate)]) != str(resolved_workdir):
                raise ValueError(f"Attempted to access path '{abs_path_candidate}' outside of working directory '{resolved_workdir}'.")

        return str(abs_path_candidate)


    async def list_files(self, path: str = '.') -> list[str]:
        """List files in a directory relative to self._workdir.
        Paths returned are relative to the 'path' argument itself.
        e.g., list_files('sub') for a file 'sub/file.txt' returns ['file.txt']
        """
        loop = asyncio.get_event_loop()
        try:
            # 'path' is relative to self._workdir. _abs resolves it safely.
            base_abs_path_str = self._abs(path)
        except ValueError: # Path outside workdir or invalid
            return [] # Or raise an error, matching original more closely might mean no error here

        base_abs_path = Path(base_abs_path_str)

        def _list_files_sync():
            if not base_abs_path.is_dir():
                # Original os.walk on a non-dir path wouldn't error but yield nothing.
                return []

            found_files: list[str] = []
            for item in base_abs_path.rglob('*'): # Recursive globbing
                if item.is_file():
                    # Make path relative to base_abs_path (the original 'path' argument)
                    relative_file_path = str(item.relative_to(base_abs_path))
                    found_files.append(relative_file_path)
            return found_files

        return await loop.run_in_executor(None, _list_files_sync)

    async def read_file(self, path: str) -> str:
        """Read a file from the filesystem relative to self._workdir."""
        loop = asyncio.get_event_loop()
        try:
            abs_file_path_str = self._abs(path) # path is relative to self._workdir
        except ValueError as e: # Path was outside workdir or invalid
             # Raise FileNotFoundError to be consistent with file system errors
             raise FileNotFoundError(f"Cannot read file: Invalid path '{path}'. {e}") from e

        abs_file_path = Path(abs_file_path_str)

        def _read_file_sync():
            if not abs_file_path.is_file():
                # Use the original relative path in the error for user clarity
                raise FileNotFoundError(f"File not found at '{path}' (resolved to '{abs_file_path}')")
            with open(abs_file_path, 'r', encoding='utf-8') as f:
                return f.read()

        try:
            return await loop.run_in_executor(None, _read_file_sync)
        except FileNotFoundError: # Re-raise to ensure it's from async context if raised in thread
            raise


    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file relative to self._workdir.
        'path' can be a simple filename like 'temp_code.js' or a relative path like 'subdir/file.txt'.
        It will be created inside self._workdir.
        """
        loop = asyncio.get_event_loop()
        try:
            # 'path' is relative to self._workdir. _abs resolves it safely.
            abs_file_path_str = self._abs(path)
        except ValueError as e: # Path was outside workdir or invalid
            # Raise PermissionError as writing outside sandbox is a permission issue.
            raise PermissionError(f"Cannot write file: Invalid path '{path}'. {e}") from e

        abs_file_path = Path(abs_file_path_str)

        def _write_file_sync():
            # Ensure parent directory of abs_file_path exists.
            abs_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(abs_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        await loop.run_in_executor(None, _write_file_sync)
