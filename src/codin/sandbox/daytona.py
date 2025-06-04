"""Daytona sandbox implementation.

This module provides a DaytonaSandbox that uses the Daytona Runner API.
"""

from __future__ import annotations

import asyncio
import base64
import shlex
import tempfile
import zipfile
import typing as _t
from pathlib import Path

from .base import Sandbox, ExecResult

__all__ = ["DaytonaSandbox"]


class DaytonaSandbox(Sandbox):
    """Daytona Runner API adapter.

    You must supply a pre-existing *workspace_id* and *api_key* (personal access token) 
    or have them available in the environment variables `DAYTONA_WORKSPACE_ID` and 
    `DAYTONA_API_KEY`.
    """

    BASE_URL = "https://runner.api.daytona.io"  # TODO: make configurable

    def __init__(self, workspace_id: _t.Optional[str] = None, api_key: _t.Optional[str] = None):
        super().__init__()
        import os
        import requests

        self._workspace_id = workspace_id or os.environ.get("DAYTONA_WORKSPACE_ID")
        self._api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if not self._workspace_id or not self._api_key:
            raise ValueError("workspace_id and api_key must be provided for DaytonaSandbox")

        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self._api_key}"

    def _get_language_executor(self, language: str) -> _t.List[str]:
        """Get the command to execute code for a given language."""
        language = language.lower()
        
        if language in ("python", "py"):
            return ["python3"]
        elif language in ("javascript", "js", "node"):
            return ["node"]
        elif language in ("bash", "sh"):
            return ["/bin/bash"]
        elif language == "go":
            return ["go", "run"]
        elif language == "rust":
            return ["cargo", "run", "--"]
        elif language in ("java",):
            return ["java"]
        elif language in ("c", "cpp", "c++"):
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

    # Internal helpers -----------------------------------------------------
    def _url(self, path: str) -> str:
        """Build API URL for the given path."""
        return f"{self.BASE_URL}/workspaces/{self._workspace_id}{path}"

    async def _post(self, path: str, **json):
        """Make a POST request to the Daytona API."""
        import requests
        loop = asyncio.get_event_loop()
        
        def _post():
            r = self._session.post(self._url(path), json=json or None, timeout=60)
            if r.status_code >= 400:
                raise requests.HTTPError(f"Daytona API error {r.status_code}: {r.text}")
            return r.json() if r.text else {}
        
        return await loop.run_in_executor(None, _post)

    async def _get(self, path: str):
        """Make a GET request to the Daytona API."""
        import requests
        loop = asyncio.get_event_loop()
        
        def _get():
            r = self._session.get(self._url(path), timeout=60)
            if r.status_code >= 400:
                raise requests.HTTPError(f"Daytona API error {r.status_code}: {r.text}")
            return r.json() if r.text else {}
        
        return await loop.run_in_executor(None, _get)

    # Lifecycle ------------------------------------------------------------
    async def _up(self) -> None:
        """Set up the Daytona sandbox."""
        # Check if workspace is already running
        info = await self._get("")
        if info.get("state") != "started":
            # Start the workspace if needed
            await self._post("/start")

    async def _down(self) -> None:
        """Clean up the Daytona sandbox."""
        try:
            await self._post("/destroy")
        except Exception:
            # Ignore errors during cleanup
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
        """Execute a command in the Daytona sandbox."""
        cmd_str = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
        payload: _t.Dict[str, _t.Any] = {"cmd": cmd_str}
        if cwd:
            payload["cwd"] = cwd
        if env:
            payload["envs"] = env
        if timeout:
            payload["timeout"] = int(timeout)
        
        try:
            data = await self._post("/exec", **payload)
            return ExecResult(data.get("stdout", ""), data.get("stderr", ""), data.get("exit_code", -1))
        except Exception as e:
            return ExecResult("", str(e), 1)

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
        """Execute code in the Daytona sandbox."""
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
                elif language.lower() in ("bash", "sh"):
                    cmd = ["/bin/bash", "-c", code]
                else:
                    # For other languages, write to temp file in sandbox
                    ext_map = {
                        "javascript": ".js",
                        "js": ".js",
                        "node": ".js",
                        "go": ".go",
                        "rust": ".rs",
                        "java": ".java",
                    }
                    ext = ext_map.get(language.lower(), ".txt")
                    temp_filename = f"/tmp/code_{hash(code) % 10000}{ext}"
                    
                    # Write code to file in sandbox
                    await self.write_file(temp_filename, code)
                    
                    cmd = executor + [temp_filename]
                
                return await self.run_cmd(cmd, timeout=timeout, env=env)
            
            except (ValueError, NotImplementedError) as e:
                return ExecResult("", str(e), 1)
        
        # Handle file path execution
        if file_path:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return ExecResult("", f"File not found: {file_path}", 1)
            
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
                            for root, dirs, files in temp_path.rglob("*"):
                                if root.is_file():
                                    rel_path = root.relative_to(temp_path)
                                    content = root.read_text(encoding='utf-8')
                                    await self.write_file(str(rel_path), content)
                            
                            # Find main file
                            all_files = [str(p.relative_to(temp_path)) for p in temp_path.rglob("*") if p.is_file()]
                            main_file = self._get_main_file_for_language(language, all_files)
                            if not main_file:
                                return ExecResult("", f"No main file found for language {language}", 1)
                            
                            exec_path = main_file
                        except zipfile.BadZipFile:
                            return ExecResult("", f"Invalid zip file: {file_path}", 1)
                else:
                    # Upload single file
                    content = file_path.read_text(encoding='utf-8')
                    exec_path = file_path.name
                    await self.write_file(exec_path, content)
            
            elif file_path.is_dir():
                # Upload entire directory
                for file in file_path.rglob("*"):
                    if file.is_file():
                        rel_path = file.relative_to(file_path)
                        content = file.read_text(encoding='utf-8')
                        await self.write_file(str(rel_path), content)
                
                # Find main file
                all_files = [str(p.relative_to(file_path)) for p in file_path.rglob("*") if p.is_file()]
                main_file = self._get_main_file_for_language(language, all_files)
                if not main_file:
                    return ExecResult("", f"No main file found for language {language}", 1)
                
                exec_path = main_file
            
            else:
                return ExecResult("", f"Invalid file path: {file_path}", 1)
            
            # Execute the file
            try:
                executor = self._get_language_executor(language)
                cmd = executor + [exec_path]
                return await self.run_cmd(cmd, timeout=timeout, env=env)
            except (ValueError, NotImplementedError) as e:
                return ExecResult("", str(e), 1)

    # Filesystem -----------------------------------------------------------
    async def list_files(self, path: str = ".") -> _t.List[str]:
        """List files in the Daytona sandbox."""
        try:
            data = await self._get(f"/fs/list?path={path}")
            return data.get("files", [])
        except Exception:
            return []

    async def read_file(self, path: str) -> str:
        """Read a file from the Daytona sandbox."""
        import requests
        loop = asyncio.get_event_loop()
        
        def _read_file():
            r = self._session.get(self._url(f"/fs/read?path={path}"), timeout=60)
            if r.status_code >= 400:
                raise requests.HTTPError(f"Daytona API error {r.status_code}: {r.text}")
            return base64.b64decode(r.text).decode()
        
        return await loop.run_in_executor(None, _read_file)

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the Daytona sandbox."""
        await self._post("/fs/write", path=path, content=content) 