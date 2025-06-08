import abc
import os
import tempfile
import typing as _t
import zipfile
from pathlib import Path

from .base import ExecResult


class CommonCodeExecutionMixin(abc.ABC):
    """
    Mixin class for common code execution logic in sandboxes.

    Assumes the class it's mixed with implements:
    - async def run_cmd(self, cmd, ...) -> ExecResult
    - async def write_file(self, path, content) -> None
    - async def list_files(self, path) -> list[str]
    - _is_windows attribute (optional, defaults to False for wider compatibility)
    """

    @abc.abstractmethod
    async def run_cmd(
        self,
        cmd: str | _t.Iterable[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Abstract method for running a command."""
        pass

    @abc.abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Abstract method for writing a file to the sandbox."""
        pass

    @abc.abstractmethod
    async def list_files(self, path: str = '.') -> list[str]:
        """Abstract method for listing files in a directory within the sandbox."""
        pass

    # Helper to check for _is_windows, defaulting to False if not present
    @property
    def _is_windows(self) -> bool:
        return getattr(self, '__is_windows', False)

    def _get_language_executor(self, language: str) -> list[str]:
        """Get the command to execute code for a given language."""
        language = language.lower()

        if language in ('python', 'py'):
            return ['python3']  # Defaulting to python3
        if language in ('javascript', 'js', 'node'):
            return ['node']
        if language in ('bash', 'sh'):
            # Handled by _common_run_code_logic or specific sandbox for Windows shell
            return ['/bin/bash']
        if language == 'go':
            return ['go', 'run']
        if language == 'rust':
            return ['cargo', 'run', '--']
        if language in ('java',):
            return ['java']
        # C/C++ typically require compilation, which is complex and varies.
        # LocalSandbox has specific logic for this.
        # For a common mixin, it's safer to indicate not directly supported here.
        if language in ('c', 'cpp', 'c++'):
            raise NotImplementedError(
                f'Language {language} requires compilation - '
                f'specific sandbox should handle this.'
            )
        if language in ('powershell', 'ps1'):
            if self._is_windows:
                return ['powershell.exe', '-Command']
            else:
                # PowerShell can be installed on Linux/macOS too (pwsh)
                # but for simplicity, we'll raise error if not on windows.
                raise ValueError(
                    'Powershell execution is typically for Windows. '
                    'Ensure your environment supports it or override this method.'
                )
        raise ValueError(f'Unsupported language: {language}')

    def _get_main_file_for_language(self, language: str, files: list[str]) -> str | None:
        """Determine the main file to execute for a given language from a list of files."""
        language = language.lower()

        main_patterns = {
            'python': ['main.py', 'app.py', '__main__.py', 'run.py'],
            'javascript': ['main.js', 'index.js', 'app.js', 'server.js'],
            'node': ['main.js', 'index.js', 'app.js', 'server.js'],
            'go': ['main.go'],
            'rust': ['main.rs', 'src/main.rs'], # common for cargo projects
            'java': ['Main.java', 'App.java'],
            'bash': ['main.sh', 'run.sh', 'start.sh'],
            'powershell': ['main.ps1', 'run.ps1', 'start.ps1'],
            'c': ['main.c'],
            'cpp': ['main.cpp', 'main.cc', 'main.cxx'],
            'c++': ['main.cpp', 'main.cc', 'main.cxx'],
        }

        patterns = main_patterns.get(language, [])

        for pattern in patterns:
            if pattern in files:
                return pattern
            # Check for files in subdirectories e.g. src/main.py
            if language == 'python': # specific check for python common patterns
                 if f"src/{pattern}" in files:
                     return f"src/{pattern}"


        extensions = {
            'python': ['.py'],
            'javascript': ['.js'],
            'node': ['.js'],
            'go': ['.go'],
            'rust': ['.rs'],
            'java': ['.java'],
            'bash': ['.sh'],
            'powershell': ['.ps1'],
            'c': ['.c'],
            'cpp': ['.cpp', '.cc', '.cxx'],
            'c++': ['.cpp', '.cc', '.cxx'],
        }

        if language in extensions:
            for file_path in files:
                if any(file_path.endswith(ext) for ext in extensions[language]):
                    return file_path
        return None

    async def _install_dependencies(self, dependencies: list[str], language: str) -> ExecResult:
        """Install dependencies for the given language."""
        if not dependencies:
            return ExecResult(stdout='', stderr='', exit_code=0)

        language = language.lower()
        cmd: list[str] = []

        if language in ('python', 'py'):
            cmd = ['pip', 'install'] + dependencies
        elif language in ('javascript', 'js', 'node'):
            cmd = ['npm', 'install'] + dependencies
        elif language == 'go':
            cmd = ['go', 'get'] + dependencies
        elif language == 'rust':
            # For Rust, dependencies are managed by Cargo.toml,
            # `cargo build` or `cargo run` usually handles them.
            # Explicit install command is not typical for deps list here.
            return ExecResult(
                stdout='Rust dependencies should be specified in Cargo.toml and are built by cargo commands.',
                stderr='',
                exit_code=0
            )
        else:
            return ExecResult(
                stdout='',
                stderr=f'Dependency installation not supported for {language} via this method.',
                exit_code=1
            )

        if not cmd: # Should not happen if logic is correct
             return ExecResult(stdout='', stderr='Internal error: no command for dependency installation.', exit_code=1)

        return await self.run_cmd(cmd)

    async def _common_run_code_logic(
        self,
        code: str | None = None,
        file_path: str | Path | None = None,
        language: str = 'python',
        dependencies: list[str] | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """
        Common logic for run_code method.
        Concrete sandbox classes will provide implementations for file operations
        and command execution.
        """
        if not code and not file_path:
            return ExecResult(stdout='', stderr='Either code or file_path must be provided', exit_code=1)

        if code and file_path:
            return ExecResult(stdout='', stderr='Cannot provide both code and file_path', exit_code=1)

        # 1. Install dependencies
        if dependencies:
            dep_result = await self._install_dependencies(dependencies, language)
            if dep_result.exit_code != 0:
                return ExecResult(
                    stdout=dep_result.stdout,
                    stderr=f'Failed to install dependencies: {dep_result.stderr}',
                    exit_code=dep_result.exit_code,
                )

        # 2. Handle direct code execution
        if code:
            try:
                # Special handling for C/C++ (compilation) should be in specific sandbox
                if language.lower() in ('c', 'cpp', 'c++'):
                    # The mixin itself won't compile. It expects the concrete class to override
                    # run_code or handle this before calling _common_run_code_logic,
                    # or to have a specific _execute_compiled_code method.
                    # For now, this path will rely on language executor not supporting it.
                    pass

                executor = self._get_language_executor(language)
                cmd_list: _t.List[str] = []

                # How to pass code to the executor varies by language
                if language.lower() in ('python', 'py'):
                    cmd_list = executor + ['-c', code]
                elif language.lower() in ('bash', 'sh'):
                    if self._is_windows: # This condition might need refinement based on actual shell used
                        # Assuming PowerShell is the target for shell commands on Windows
                        # Or this part should be overridden by LocalSandbox entirely
                        cmd_list = ['powershell.exe', '-Command', code]
                    else:
                        cmd_list = executor + ['-c', code]
                elif language.lower() in ('powershell', 'ps1'):
                    if not self._is_windows:
                         raise ValueError("Powershell is primarily a Windows shell.")
                    cmd_list = ['powershell.exe', '-Command', code]
                else:
                    # For many languages (JS, Go, Rust, Java), code is written to a temp file
                    # The specific sandbox's write_file will handle where this temp file goes.
                    # E.g. /tmp in E2B/Daytona, or a local temp dir in LocalSandbox.

                    # Define extensions for temp files
                    ext_map = {
                        'javascript': '.js', 'js': '.js', 'node': '.js',
                        'go': '.go', 'rust': '.rs', 'java': '.java',
                        # Other languages can be added here if they run from files
                    }
                    ext = ext_map.get(language.lower())
                    if not ext:
                        return ExecResult(stdout='', stderr=f"Direct code execution for {language} by writing to temp file is not fully configured in mixin.", exit_code=1)

                    # Using a simple temp file name. Concrete implementation might need more robust naming.
                    # The path for `write_file` is relative to sandbox's root.
                    # LocalSandbox might make this an OS temp file.
                    # E2B/Daytona will write to their /tmp or similar.
                    temp_file_name = f"temp_code_{hash(code) % 10000}{ext}"

                    # In LocalSandbox, this temp file should be created in its _workdir or OS temp.
                    # In remote sandboxes, this will be a path like /tmp/temp_code_xxxx.js
                    await self.write_file(temp_file_name, code)

                    # The command will then be `executor /path/to/temp_file_name`
                    # The `cwd` for run_cmd might need to be set to the directory of temp_file_name
                    # if the language or code expects relative paths.
                    cmd_list = executor + [temp_file_name]

                if not cmd_list:
                    return ExecResult(stdout='', stderr=f"Could not determine command for language {language}", exit_code=1)

                return await self.run_cmd(cmd_list, timeout=timeout, env=env)

            except (ValueError, NotImplementedError) as e:
                return ExecResult(stdout='', stderr=str(e), exit_code=1)

        # 3. Handle file path execution (placeholder - details depend on abstract file ops)
        if file_path:
            # This part is more complex to generalize due to differences in:
            # - How files/zips/dirs are "uploaded" or made available.
            #   LocalSandbox: copies, uses temp dirs.
            #   Daytona/E2B: use self.write_file for each file, or specific upload tools.
            # - Where the code is executed (e.g., in a temp dir, or relative to uploaded path).
            #
            # A truly common version would need abstract methods like:
            # - async def _prepare_execution_environment_from_path(self, file_path, language) -> (exec_file_path, cwd)
            #
            # For this iteration, we'll assume the concrete class's run_code
            # will handle the path preparation and then call a more focused execution helper,
            # or this part of _common_run_code_logic will be simpler.

            # Simplified approach: Assume file_path is directly usable or prepared by caller
            # This part will likely be overridden or supplemented by the concrete classes.

            file_path_obj = Path(file_path) # Ensure it's a Path object

            # The following is a very simplified placeholder.
            # Real implementation needs to handle:
            # - Zip extraction: list files within zip, find main, "upload" them.
            # - Directory "upload": list local files, "upload" them, find main.
            # - Single file "upload".
            # - Determining CWD for execution.

            # This basic version assumes file_path is a single file ready to be executed
            # in the sandbox environment, and _get_main_file_for_language is not needed here
            # as the file_path IS the main file.
            # More complex scenarios (zip/dir) need to be handled by the overriding class
            # or by expanding this method with more abstract file operations.

            try:
                if language.lower() in ('c', 'cpp', 'c++'):
                    # Similar to direct code, compilation is sandbox-specific
                    raise NotImplementedError("C/C++ execution from file_path requires sandbox-specific compilation logic.")

                executor = self._get_language_executor(language)
                # Assuming file_path is a path string relative to sandbox root
                # and is the direct file to execute.
                # Cwd might need to be the parent directory of file_path_obj.
                # This needs to be provided by the concrete sandbox.
                exec_target_path_str = str(file_path) # Needs to be string for executor list

                # Determine cwd:
                # For LocalSandbox, it would be file_path_obj.parent if it's an absolute path from OS perspective
                # or relative to its workdir.
                # For remote sandboxes, it's the parent of the remote path.
                # This needs to be handled carefully by the caller or an overridden method.
                # Let's assume for now that run_cmd in the concrete class handles cwd resolution if path is relative.
                run_cwd = str(Path(exec_target_path_str).parent)
                if run_cwd == '.': # If it's a top-level file.
                    run_cwd = None


                cmd = executor + [exec_target_path_str]
                return await self.run_cmd(cmd, cwd=run_cwd, timeout=timeout, env=env)
            except (ValueError, NotImplementedError) as e:
                return ExecResult(stdout='', stderr=str(e), exit_code=1)

        return ExecResult(stdout='', stderr='Reached end of _common_run_code_logic without action', exit_code=1)

    # Concrete classes should override run_code and call _common_run_code_logic
    # appropriately, potentially after handling specific setup (like C/C++ compilation
    # or complex file_path preparation).
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
        # This is a basic implementation.
        # Concrete classes might need to override this to handle:
        # - C/C++ compilation (LocalSandbox)
        # - Complex file_path preparations (extracting zips, preparing directories)
        #   before calling _common_run_code_logic.

        # For example, LocalSandbox's C/C++ handling:
        if language.lower() in ('c', 'cpp', 'c++') and code:
             # LocalSandbox has a specific way to compile and run C/C++ code.
             # This logic should ideally stay in LocalSandbox or be made very generic.
             # For now, we'll indicate it's not handled by this generic run_code.
             # A better approach would be for LocalSandbox.run_code to NOT call
             # super()._common_run_code_logic for C/C++ if it wants to handle it fully.
            return ExecResult(stdout='', stderr=f"Generic mixin run_code does not handle C/C++ compilation. Override in concrete class.", exit_code=1)

        # If file_path is a complex type (zip/dir), the concrete class should prepare it,
        # then call _common_run_code_logic with a simplified 'executable_file_path' and 'execution_cwd'.
        # The current _common_run_code_logic's file_path handling is too basic for zips/dirs.

        # Example: if file_path is a zip or dir, the concrete class should:
        # 1. "Upload" / copy files to a staging area in the sandbox.
        # 2. Determine the main_file_to_execute within that staging area.
        # 3. Determine the correct cwd for execution (usually the root of the staging area).
        # 4. Call:
        #    await self._common_run_code_logic(
        #        code=None,  # Code is not used here
        #        file_path=main_file_to_execute, # This is now a simple path in the sandbox
        #        language=language,
        #        dependencies=dependencies, # Already handled by _common_run_code_logic
        #        timeout=timeout,
        #        env=env,
        #        # Potentially pass execution_cwd to _common_run_code_logic or ensure run_cmd uses it
        #    )
        # This means the file_path argument to _common_run_code_logic should be the *final* path to execute.

        return await self._common_run_code_logic(
            code=code,
            file_path=file_path, # This needs to be the path *within* the sandbox if prepared by caller
            language=language,
            dependencies=dependencies,
            timeout=timeout,
            env=env
        )
