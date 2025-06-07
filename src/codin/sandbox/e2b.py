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

from .base import ExecResult, Sandbox, ShellEnvironmentPolicy


__all__ = ['E2BSandbox']


class E2BSandbox(Sandbox):
    """Adapter around E2B cloud sandbox SDK."""

    def __init__(self, *, env_policy: ShellEnvironmentPolicy | None = None, **kwargs):
        super().__init__(env_policy=env_policy)
        try:
            from e2b import Sandbox as _E2B
        except ImportError as e:  # pragma: no cover
            raise RuntimeError('e2b package not available – install with `pip install e2b`.') from e

        self._sandbox = _E2B(**kwargs)

    def _get_language_executor(self, language: str) -> list[str]:
        """Get the command to execute code for a given language."""
        language = language.lower()

        if language in ('python', 'py'):
            return ['python3']
        if language in ('javascript', 'js', 'node'):
            return ['node']
        if language in ('bash', 'sh'):
            return ['/bin/bash']
        if language == 'go':
            return ['go', 'run']
        if language == 'rust':
            return ['cargo', 'run', '--']
        if language in ('java',):
            return ['java']
        if language in ('c', 'cpp', 'c++'):
            raise NotImplementedError(f'Language {language} requires compilation - not yet supported')
        raise ValueError(f'Unsupported language: {language}')

    def _get_main_file_for_language(self, language: str, files: list[str]) -> str | None:
        """Determine the main file to execute for a given language."""
        language = language.lower()

        # Common main file patterns by language
        main_patterns = {
            'python': ['main.py', 'app.py', '__main__.py', 'run.py'],
            'javascript': ['main.js', 'index.js', 'app.js', 'server.js'],
            'node': ['main.js', 'index.js', 'app.js', 'server.js'],
            'go': ['main.go'],
            'rust': ['main.rs', 'src/main.rs'],
            'java': ['Main.java', 'App.java'],
            'bash': ['main.sh', 'run.sh', 'start.sh'],
        }

        patterns = main_patterns.get(language, [])

        # First, look for exact matches
        for pattern in patterns:
            if pattern in files:
                return pattern

        # Then look for files with the right extension
        extensions = {
            'python': ['.py'],
            'javascript': ['.js'],
            'node': ['.js'],
            'go': ['.go'],
            'rust': ['.rs'],
            'java': ['.java'],
            'bash': ['.sh'],
        }

        if language in extensions:
            for file in files:
                if any(file.endswith(ext) for ext in extensions[language]):
                    return file

        return None

    async def _install_dependencies(self, dependencies: list[str], language: str) -> ExecResult:
        """Install dependencies for the given language."""
        if not dependencies:
            return ExecResult('', '', 0)

        language = language.lower()

        if language in ('python', 'py'):
            # Use pip to install Python dependencies
            cmd = ['pip', 'install'] + dependencies
        elif language in ('javascript', 'js', 'node'):
            # Use npm to install Node.js dependencies
            cmd = ['npm', 'install'] + dependencies
        elif language == 'go':
            # Use go get for Go dependencies
            cmd = ['go', 'get'] + dependencies
        elif language == 'rust':
            # For Rust, dependencies should be in Cargo.toml
            return ExecResult('', 'Rust dependencies should be specified in Cargo.toml', 1)
        else:
            return ExecResult('', f'Dependency installation not supported for {language}', 1)

        return await self.run_cmd(cmd)

    # Lifecycle ------------------------------------------------------------
    async def _up(self) -> None:
        """Set up the E2B sandbox."""
        # E2B sandbox is created in __init__, so nothing to do

    async def _down(self) -> None:
        """Clean up the E2B sandbox."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sandbox.kill)

    # Exec -----------------------------------------------------------------
    async def run_cmd(
        self,
        cmd: str | _t.Iterable[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute a command in the E2B sandbox."""
        loop = asyncio.get_event_loop()

        # E2B expects a single shell string
        cmd_str = cmd if isinstance(cmd, str) else ' '.join(shlex.quote(c) for c in cmd)

        def _exec():
            result = self._sandbox.commands.run(
                cmd_str,
                cwd=cwd,
                envs=self._prepare_env(env),
                timeout=timeout,
            )
            return ExecResult(result.stdout, result.stderr, result.exit_code)

        return await loop.run_in_executor(None, _exec)

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
        """Execute code in the E2B sandbox."""
        if not code and not file_path:
            return ExecResult('', 'Either code or file_path must be provided', 1)

        if code and file_path:
            return ExecResult('', 'Cannot provide both code and file_path', 1)

        # Install dependencies first if provided
        if dependencies:
            dep_result = await self._install_dependencies(dependencies, language)
            if dep_result.exit_code != 0:
                return ExecResult(
                    dep_result.stdout, f'Failed to install dependencies: {dep_result.stderr}', dep_result.exit_code
                )

        # Handle direct code execution
        if code:
            try:
                executor = self._get_language_executor(language)
                if language.lower() in ('python', 'py'):
                    cmd = executor + ['-c', code]
                elif language.lower() in ('bash', 'sh'):
                    cmd = ['/bin/bash', '-c', code]
                else:
                    # For other languages, write to temp file in sandbox
                    ext_map = {
                        'javascript': '.js',
                        'js': '.js',
                        'node': '.js',
                        'go': '.go',
                        'rust': '.rs',
                        'java': '.java',
                    }
                    ext = ext_map.get(language.lower(), '.txt')
                    temp_filename = f'/tmp/code_{hash(code) % 10000}{ext}'

                    # Write code to file in sandbox
                    await self.write_file(temp_filename, code)

                    cmd = executor + [temp_filename]

                return await self.run_cmd(cmd, timeout=timeout, env=env)

            except (ValueError, NotImplementedError) as e:
                return ExecResult('', str(e), 1)

        # Handle file path execution
        if file_path:
            file_path = Path(file_path)

            if not file_path.exists():
                return ExecResult('', f'File not found: {file_path}', 1)

            # Upload file(s) to sandbox
            if file_path.is_file():
                if file_path.suffix == '.zip':
                    # Extract zip file locally first, then upload
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        try:
                            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                zip_ref.extractall(temp_path)

                            # Upload all extracted files
                            for root, dirs, files in temp_path.rglob('*'):
                                if root.is_file():
                                    rel_path = root.relative_to(temp_path)
                                    content = root.read_text(encoding='utf-8')
                                    await self.write_file(str(rel_path), content)

                            # Find main file
                            all_files = [str(p.relative_to(temp_path)) for p in temp_path.rglob('*') if p.is_file()]
                            main_file = self._get_main_file_for_language(language, all_files)
                            if not main_file:
                                return ExecResult('', f'No main file found for language {language}', 1)

                            exec_path = main_file
                        except zipfile.BadZipFile:
                            return ExecResult('', f'Invalid zip file: {file_path}', 1)
                else:
                    # Upload single file
                    content = file_path.read_text(encoding='utf-8')
                    exec_path = file_path.name
                    await self.write_file(exec_path, content)

            elif file_path.is_dir():
                # Upload entire directory
                for file in file_path.rglob('*'):
                    if file.is_file():
                        rel_path = file.relative_to(file_path)
                        content = file.read_text(encoding='utf-8')
                        await self.write_file(str(rel_path), content)

                # Find main file
                all_files = [str(p.relative_to(file_path)) for p in file_path.rglob('*') if p.is_file()]
                main_file = self._get_main_file_for_language(language, all_files)
                if not main_file:
                    return ExecResult('', f'No main file found for language {language}', 1)

                exec_path = main_file

            else:
                return ExecResult('', f'Invalid file path: {file_path}', 1)

            # Execute the file
            try:
                executor = self._get_language_executor(language)
                cmd = executor + [exec_path]
                return await self.run_cmd(cmd, timeout=timeout, env=env)
            except (ValueError, NotImplementedError) as e:
                return ExecResult('', str(e), 1)

    # Filesystem -----------------------------------------------------------
    async def list_files(self, path: str = '.') -> list[str]:
        """List files in the E2B sandbox."""
        loop = asyncio.get_event_loop()

        def _list_files():
            return [f.path for f in self._sandbox.files.list(path)]

        return await loop.run_in_executor(None, _list_files)

    async def read_file(self, path: str) -> str:
        """Read a file from the E2B sandbox."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._sandbox.files.read(path))

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the E2B sandbox."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._sandbox.files.write(path, content))
