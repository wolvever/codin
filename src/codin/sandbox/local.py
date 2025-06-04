"""Local sandbox implementation using subprocess.

This module provides a LocalSandbox that executes commands directly on the host
using subprocess. It supports both Windows PowerShell and Unix bash automatically.

**Warning** – this is not isolated! Only use for trusted inputs or when
combined with additional OS-level sandboxing (e.g. containers).
"""

from __future__ import annotations

import asyncio
import os
import platform
import shlex
import subprocess
import tempfile
import zipfile
import typing as _t
from pathlib import Path

from .base import Sandbox, ExecResult

__all__ = ["LocalSandbox"]


class LocalSandbox(Sandbox):
    """Execute commands directly on the host using subprocess.

    **Warning** – this is not isolated! Only use for trusted inputs or when
    combined with additional OS-level sandboxing (e.g. containers).
    
    Enhanced to support both Windows PowerShell and Unix bash automatically.
    """

    def __init__(self, workdir: _t.Optional[str] = None):
        super().__init__()
        self._workdir = workdir or os.getcwd()
        self._is_windows = platform.system().lower() == "windows"
        self._shell_executable = self._detect_shell()

    def _detect_shell(self) -> str:
        """Detect the best shell to use for this platform."""
        if self._is_windows:
            # On Windows, prefer PowerShell, fallback to cmd
            return "powershell.exe"
        else:
            # On Unix-like systems, use bash
            return "/bin/bash"

    def _prepare_command(self, cmd: _t.Union[str, _t.Iterable[str]]) -> tuple[_t.Union[str, _t.List[str]], bool]:
        """Prepare command for execution, handling cross-platform differences."""
        if isinstance(cmd, str):
            if self._is_windows:
                # For PowerShell, wrap the command properly
                return ["powershell.exe", "-NoProfile", "-Command", cmd], False
            else:
                # For bash, use shell=True
                return cmd, True
        else:
            # For list commands, just return as-is
            return list(cmd), False

    def _get_language_executor(self, language: str) -> _t.List[str]:
        """Get the command to execute code for a given language."""
        language = language.lower()
        
        if language in ("python", "py"):
            return ["python"]
        elif language in ("javascript", "js", "node"):
            return ["node"]
        elif language in ("bash", "sh"):
            return ["/bin/bash"] if not self._is_windows else ["powershell.exe", "-Command"]
        elif language in ("powershell", "ps1"):
            return ["powershell.exe", "-Command"]
        elif language == "go":
            return ["go", "run"]
        elif language == "rust":
            return ["cargo", "run", "--"]
        elif language in ("java",):
            return ["java"]
        elif language in ("c", "cpp", "c++"):
            # For C/C++, we'd need to compile first - this is a simplified approach
            raise NotImplementedError(f"Language {language} requires compilation - not yet supported")
        else:
            raise ValueError(f"Unsupported language: {language}")

    def _get_main_file_for_language(self, language: str, files: _t.List[str]) -> _t.Optional[str]:
        """Determine the main file to execute for a given language."""
        language = language.lower()
        
        # Common main file patterns by language
        main_patterns = {
            "python": ["main.py", "app.py", "__main__.py", "run.py"],
            "javascript": ["main.js", "index.js", "app.js", "server.js"],
            "node": ["main.js", "index.js", "app.js", "server.js"],
            "go": ["main.go"],
            "rust": ["main.rs", "src/main.rs"],
            "java": ["Main.java", "App.java"],
            "bash": ["main.sh", "run.sh", "start.sh"],
            "powershell": ["main.ps1", "run.ps1", "start.ps1"],
        }
        
        patterns = main_patterns.get(language, [])
        
        # First, look for exact matches
        for pattern in patterns:
            if pattern in files:
                return pattern
        
        # Then look for files with the right extension
        extensions = {
            "python": [".py"],
            "javascript": [".js"],
            "node": [".js"],
            "go": [".go"],
            "rust": [".rs"],
            "java": [".java"],
            "bash": [".sh"],
            "powershell": [".ps1"],
        }
        
        if language in extensions:
            for file in files:
                if any(file.endswith(ext) for ext in extensions[language]):
                    return file
        
        return None

    async def _install_dependencies(self, dependencies: _t.List[str], language: str) -> ExecResult:
        """Install dependencies for the given language."""
        if not dependencies:
            return ExecResult("", "", 0)
        
        language = language.lower()
        
        if language in ("python", "py"):
            # Use pip to install Python dependencies
            cmd = ["pip", "install"] + dependencies
        elif language in ("javascript", "js", "node"):
            # Use npm to install Node.js dependencies
            cmd = ["npm", "install"] + dependencies
        elif language == "go":
            # Use go get for Go dependencies
            cmd = ["go", "get"] + dependencies
        elif language == "rust":
            # For Rust, dependencies should be in Cargo.toml
            return ExecResult("", "Rust dependencies should be specified in Cargo.toml", 1)
        else:
            return ExecResult("", f"Dependency installation not supported for {language}", 1)
        
        return await self.run_cmd(cmd)

    # Lifecycle ------------------------------------------------------------
    async def _up(self) -> None:
        """Set up the local sandbox."""
        # Ensure working directory exists
        os.makedirs(self._workdir, exist_ok=True)

    async def _down(self) -> None:
        """Clean up the local sandbox."""
        pass

    # Exec -----------------------------------------------------------------
    async def run_cmd(
        self,
        cmd: _t.Union[str, _t.Iterable[str]],
        *,
        cwd: _t.Optional[str] = None,
        timeout: _t.Optional[float] = None,
        env: _t.Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Execute a command in the sandbox."""
        args, shell = self._prepare_command(cmd)

        # Use run in executor to prevent blocking the event loop
        loop = asyncio.get_event_loop()
        
        def _run_subprocess():
            # Set up environment with UTF-8 encoding for Windows
            subprocess_env = {**os.environ, **(env or {})}
            if self._is_windows:
                # Force UTF-8 encoding on Windows
                subprocess_env["PYTHONIOENCODING"] = "utf-8"
            
            # For Windows PowerShell commands, we need special handling
            if self._is_windows and isinstance(args, list) and args[0] == "powershell.exe":
                proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd or self._workdir,
                    env=subprocess_env,
                    shell=False,  # Don't use shell=True with PowerShell args
                    encoding='utf-8',  # Explicitly set encoding
                    errors='replace',  # Handle encoding errors gracefully
                )
            else:
                proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd or self._workdir,
                    env=subprocess_env,
                    shell=shell,
                    encoding='utf-8',  # Explicitly set encoding
                    errors='replace',  # Handle encoding errors gracefully
                )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                return ExecResult(stdout, stderr, exit_code=proc.returncode)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                return ExecResult(stdout, stderr, exit_code=-1)
        
        # Run the blocking operation in a thread pool
        return await loop.run_in_executor(None, _run_subprocess)

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
        """Execute code in the sandbox."""
        if not code and not file_path:
            return ExecResult("", "Either code or file_path must be provided", 1)
        
        if code and file_path:
            return ExecResult("", "Cannot provide both code and file_path", 1)
        
        # Install dependencies first if provided
        if dependencies:
            dep_result = await self._install_dependencies(dependencies, language)
            if dep_result.exit_code != 0:
                return ExecResult(
                    dep_result.stdout,
                    f"Failed to install dependencies: {dep_result.stderr}",
                    dep_result.exit_code
                )
        
        # Handle direct code execution
        if code:
            try:
                executor = self._get_language_executor(language)
                if language.lower() in ("python", "py"):
                    cmd = executor + ["-c", code]
                elif language.lower() in ("javascript", "js", "node"):
                    # For Node.js, we need to write to a temp file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                        f.write(code)
                        temp_file = f.name
                    try:
                        cmd = executor + [temp_file]
                        return await self.run_cmd(cmd, timeout=timeout, env=env)
                    finally:
                        os.unlink(temp_file)
                elif language.lower() in ("bash", "sh"):
                    if self._is_windows:
                        cmd = ["powershell.exe", "-Command", code]
                    else:
                        cmd = ["/bin/bash", "-c", code]
                elif language.lower() in ("powershell", "ps1"):
                    cmd = ["powershell.exe", "-Command", code]
                else:
                    # For other languages, write to temp file
                    ext_map = {
                        "go": ".go",
                        "rust": ".rs",
                        "java": ".java",
                    }
                    ext = ext_map.get(language.lower(), ".txt")
                    with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
                        f.write(code)
                        temp_file = f.name
                    try:
                        cmd = executor + [temp_file]
                        return await self.run_cmd(cmd, timeout=timeout, env=env)
                    finally:
                        os.unlink(temp_file)
                
                return await self.run_cmd(cmd, timeout=timeout, env=env)
            
            except (ValueError, NotImplementedError) as e:
                return ExecResult("", str(e), 1)
        
        # Handle file path execution
        if file_path:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return ExecResult("", f"File not found: {file_path}", 1)
            
            # Create a temporary directory for execution
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                if file_path.is_file():
                    if file_path.suffix == '.zip':
                        # Extract zip file
                        try:
                            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                zip_ref.extractall(temp_path)
                            
                            # Find main file
                            all_files = []
                            for root, dirs, files in os.walk(temp_path):
                                for file in files:
                                    rel_path = os.path.relpath(os.path.join(root, file), temp_path)
                                    all_files.append(rel_path)
                            
                            main_file = self._get_main_file_for_language(language, all_files)
                            if not main_file:
                                return ExecResult("", f"No main file found for language {language}", 1)
                            
                            exec_path = temp_path / main_file
                        except zipfile.BadZipFile:
                            return ExecResult("", f"Invalid zip file: {file_path}", 1)
                    else:
                        # Copy single file
                        exec_path = temp_path / file_path.name
                        exec_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
                
                elif file_path.is_dir():
                    # Copy entire directory
                    import shutil
                    shutil.copytree(file_path, temp_path / file_path.name)
                    
                    # Find main file
                    all_files = []
                    search_path = temp_path / file_path.name
                    for root, dirs, files in os.walk(search_path):
                        for file in files:
                            rel_path = os.path.relpath(os.path.join(root, file), search_path)
                            all_files.append(rel_path)
                    
                    main_file = self._get_main_file_for_language(language, all_files)
                    if not main_file:
                        return ExecResult("", f"No main file found for language {language}", 1)
                    
                    exec_path = search_path / main_file
                
                else:
                    return ExecResult("", f"Invalid file path: {file_path}", 1)
                
                # Execute the file
                try:
                    executor = self._get_language_executor(language)
                    cmd = executor + [str(exec_path)]
                    return await self.run_cmd(cmd, cwd=str(exec_path.parent), timeout=timeout, env=env)
                except (ValueError, NotImplementedError) as e:
                    return ExecResult("", str(e), 1)

    # Filesystem -----------------------------------------------------------
    def _abs(self, path: str) -> str:
        """Get absolute path within the sandbox working directory."""
        return os.path.join(self._workdir, path)

    async def list_files(self, path: str = ".") -> _t.List[str]:
        """List files in a directory."""
        loop = asyncio.get_event_loop()
        
        def _list_files():
            root = self._abs(path)
            files: _t.List[str] = []
            for dirpath, _dirnames, filenames in os.walk(root):
                for f in filenames:
                    files.append(os.path.relpath(os.path.join(dirpath, f), root))
            return files
        
        return await loop.run_in_executor(None, _list_files)

    async def read_file(self, path: str) -> str:
        """Read a file from the filesystem."""
        loop = asyncio.get_event_loop()
        
        def _read_file():
            with open(self._abs(path), "r", encoding="utf-8") as f:
                return f.read()
        
        return await loop.run_in_executor(None, _read_file)

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file."""
        loop = asyncio.get_event_loop()
        
        def _write_file():
            abs_path = self._abs(path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        await loop.run_in_executor(None, _write_file) 