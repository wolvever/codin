"""Daytona sandbox implementation for codin agents.

This module provides a sandbox implementation that integrates with
Daytona Runner API for secure remote code execution.
"""

import asyncio
import base64
import shlex
import tempfile
import typing as _t
import zipfile
from pathlib import Path
import os # For path manipulation

from .base import ExecResult, Sandbox, ShellEnvironmentPolicy
from .common_exec import CommonCodeExecutionMixin

__all__ = ['DaytonaSandbox']


class DaytonaSandbox(Sandbox, CommonCodeExecutionMixin):
    """Daytona Runner API adapter.

    You must supply a pre-existing *workspace_id* and *api_key* (personal access token)
    or have them available in the environment variables `DAYTONA_WORKSPACE_ID` and
    `DAYTONA_API_KEY`.
    """

    BASE_URL = 'https://runner.api.daytona.io'  # TODO: make configurable

    def __init__(
        self,
        workspace_id: str | None = None,
        api_key: str | None = None,
        *,
        env_policy: ShellEnvironmentPolicy | None = None,
    ):
        super().__init__(env_policy=env_policy)
        # import os # Moved to top-level
        import requests

        self._workspace_id = workspace_id or os.environ.get('DAYTONA_WORKSPACE_ID')
        self._api_key = api_key or os.environ.get('DAYTONA_API_KEY')
        if not self._workspace_id or not self._api_key:
            raise ValueError('workspace_id and api_key must be provided for DaytonaSandbox')

        self._session = requests.Session()
        self._session.headers['Authorization'] = f'Bearer {self._api_key}'

    # _get_language_executor, _get_main_file_for_language, _install_dependencies
    # are now inherited from CommonCodeExecutionMixin.
    # The mixin's defaults (e.g. python3) align with Daytona's original implementation.

    # Internal helpers -----------------------------------------------------
    def _url(self, path: str) -> str:
        """Build API URL for the given path."""
        return f'{self.BASE_URL}/workspaces/{self._workspace_id}{path}'

    async def _post(self, path: str, **json):
        """Make a POST request to the Daytona API."""
        import requests

        loop = asyncio.get_event_loop()

        def _post_sync():
            r = self._session.post(self._url(path), json=json or None, timeout=60)
            if r.status_code >= 400:
                # Log the error or include more details
                error_message = f'Daytona API error {r.status_code}: {r.text} for URL {self._url(path)} with payload {json}'
                raise requests.HTTPError(error_message, response=r)
            return r.json() if r.text else {}

        return await loop.run_in_executor(None, _post_sync)

    async def _get(self, path: str):
        """Make a GET request to the Daytona API."""
        import requests # Keep import local if only used here and _post

        loop = asyncio.get_event_loop()

        def _get_sync():
            r = self._session.get(self._url(path), timeout=60)
            if r.status_code >= 400:
                error_message = f'Daytona API error {r.status_code}: {r.text} for URL {self._url(path)}'
                raise requests.HTTPError(error_message, response=r)
            return r.json() if r.text else {}

        return await loop.run_in_executor(None, _get_sync)

    # Lifecycle ------------------------------------------------------------
    async def _up(self) -> None:
        """Set up the Daytona sandbox."""
        try:
            info = await self._get('')
            if info.get('state') != 'started':
                await self._post('/start')
        except Exception as e:
            # Better to propagate or log critical setup errors
            # For now, keeping original behavior of potentially ignoring if _get fails
            # but if _post('/start') fails, it will propagate.
            print(f"Error during DaytonaSandbox _up: {e}") # Or logger.error
            # raise # Optionally re-raise to make setup failures explicit


    async def _down(self) -> None:
        """Clean up the Daytona sandbox."""
        try:
            await self._post('/destroy')
        except Exception:
            pass # Original behavior: ignore errors during cleanup

    # Exec -----------------------------------------------------------------
    async def run_cmd(
        self,
        cmd: str | _t.Iterable[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute a command in the Daytona sandbox."""
        cmd_str = cmd if isinstance(cmd, str) else ' '.join(shlex.quote(c) for c in cmd)
        payload: dict[str, _t.Any] = {'cmd': cmd_str}
        if cwd:
            # Daytona API expects cwd to be an absolute path in the workspace or relative to workspace root.
            # Paths like "." are fine.
            payload['cwd'] = cwd
        env_map = self._prepare_env(env) # From Sandbox base class
        if env_map:
            payload['envs'] = env_map
        if timeout:
            payload['timeout'] = int(timeout)

        try:
            data = await self._post('/exec', **payload)
            return ExecResult(data.get('stdout', ''), data.get('stderr', ''), data.get('exit_code', -1))
        except Exception as e:
            # Provide more context in error if possible
            return ExecResult('', f'Failed to execute command in Daytona: {e}', 1)

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
        """Execute code in the Daytona sandbox."""
        if not code and not file_path:
            return ExecResult('', 'Either code or file_path must be provided', 1)
        if code and file_path:
            return ExecResult('', 'Cannot provide both code and file_path', 1)

        # 1. Install dependencies using the inherited _install_dependencies
        if dependencies:
            dep_result = await self._install_dependencies(dependencies, language) # Calls self.run_cmd
            if dep_result.exit_code != 0:
                return ExecResult(
                    dep_result.stdout, f'Failed to install dependencies: {dep_result.stderr}', dep_result.exit_code
                )

        # 2. Handle direct code execution
        if code:
            # _common_run_code_logic will handle writing to a temp file in Daytona via self.write_file
            # (e.g. /tmp/code_HASH.ext) for languages not supporting -c.
            return await self._common_run_code_logic(
                code=code, file_path=None, language=language,
                dependencies=None, # Already handled
                timeout=timeout, env=env
            )

        # 3. Handle file path execution (uploading local files/zips/dirs to Daytona)
        if file_path:
            local_source_path = Path(file_path) # Path on the system running this agent
            if not local_source_path.exists():
                return ExecResult('', f'Local file/directory not found: {local_source_path}', 1)

            # Create a unique directory name in Daytona for staging files
            # Using workspace root as base for these staged files.
            # Example: "codin_run_12345/"
            remote_staging_dir_name = f"codin_run_{os.urandom(4).hex()}"

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
                                    # Daytona paths should use forward slashes
                                    daytona_target_path = Path(remote_staging_dir_name) / relative_path_in_zip
                                    await self.write_file(str(daytona_target_path).replace(os.sep, '/'), item.read_text(encoding='utf-8'))
                                    staged_files_relative_to_remote_dir.append(str(relative_path_in_zip).replace(os.sep, '/'))
                    else: # Single file
                        daytona_target_path = Path(remote_staging_dir_name) / local_source_path.name
                        await self.write_file(str(daytona_target_path).replace(os.sep, '/'), local_source_path.read_text(encoding='utf-8'))
                        staged_files_relative_to_remote_dir.append(local_source_path.name)

                elif local_source_path.is_dir():
                    for item in local_source_path.rglob('*'):
                        if item.is_file():
                            relative_path_in_dir = item.relative_to(local_source_path)
                            daytona_target_path = Path(remote_staging_dir_name) / relative_path_in_dir
                            await self.write_file(str(daytona_target_path).replace(os.sep, '/'), item.read_text(encoding='utf-8'))
                            staged_files_relative_to_remote_dir.append(str(relative_path_in_dir).replace(os.sep, '/'))
                else:
                    return ExecResult('', f'Unsupported local path type: {local_source_path}', 1)

                if not staged_files_relative_to_remote_dir:
                     return ExecResult('', f'No files found to execute for path: {local_source_path}', 1)

                # Determine the main file to execute within the remote_staging_dir_name
                main_file_in_remote_dir = self._get_main_file_for_language(language, staged_files_relative_to_remote_dir)
                if not main_file_in_remote_dir:
                    return ExecResult('', f'No main file found for language {language} in uploaded content from {local_source_path.name}', 1)

                # This is the path to execute, relative to Daytona's workspace root.
                final_exec_path_in_daytona = str(Path(remote_staging_dir_name) / main_file_in_remote_dir).replace(os.sep, '/')

                # _common_run_code_logic will derive cwd from final_exec_path_in_daytona's parent.
                # e.g. if final_exec_path is "codin_run_123/src/main.py", cwd will be "codin_run_123/src"
                return await self._common_run_code_logic(
                    code=None,
                    file_path=final_exec_path_in_daytona, # Path within Daytona
                    language=language,
                    dependencies=None, # Already handled
                    timeout=timeout,
                    env=env
                )
            except Exception as e: # Catch errors during file prep/upload
                return ExecResult('', f"Error preparing/uploading files for execution: {e}", 1)

        return ExecResult('', 'Internal error in Daytona run_code dispatch', 1)


    # Filesystem -----------------------------------------------------------
    async def list_files(self, path: str = '.') -> list[str]:
        """List files in the Daytona sandbox. Path is relative to workspace root."""
        try:
            # Ensure path is formatted correctly for Daytona (forward slashes, relative)
            query_path = path.replace(os.sep, '/')
            if query_path.startswith('/'): query_path = query_path[1:]

            data = await self._get(f'/fs/list?path={query_path}')
            return data.get('files', [])
        except Exception: # Catch potential HTTPError from _get
            return []

    async def read_file(self, path: str) -> str:
        """Read a file from the Daytona sandbox. Path is relative to workspace root."""
        import requests # For HTTPError specifically

        # Ensure path is formatted correctly
        query_path = path.replace(os.sep, '/')
        if query_path.startswith('/'): query_path = query_path[1:]

        loop = asyncio.get_event_loop()
        def _read_file_sync():
            r = self._session.get(self._url(f'/fs/read?path={query_path}'), timeout=60)
            if r.status_code >= 400:
                raise requests.HTTPError(f'Daytona API error {r.status_code}: {r.text} for URL {self._url(f"/fs/read?path={query_path}")}', response=r)
            return base64.b64decode(r.text).decode('utf-8') # Ensure utf-8 decoding

        try:
            return await loop.run_in_executor(None, _read_file_sync)
        except requests.HTTPError as e:
            # Map to FileNotFoundError for consistency if appropriate, or raise custom error
            if e.response is not None and e.response.status_code == 404:
                raise FileNotFoundError(f"File not found in Daytona: {path} (query: {query_path})") from e
            raise IOError(f"Failed to read file from Daytona '{path}': {e}") from e


    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the Daytona sandbox. Path is relative to workspace root."""
        # Ensure path is formatted correctly
        daytona_path = path.replace(os.sep, '/')
        if daytona_path.startswith('/'): daytona_path = daytona_path[1:]

        # Content for Daytona's /fs/write is expected to be plain text, not base64 encoded by client.
        # If API expects base64, then: content_b64 = base64.b64encode(content.encode('utf-8')).decode('ascii')
        # Assuming plain text based on previous example.
        try:
            await self._post('/fs/write', file_path=daytona_path, content=content)
        except Exception as e: # Catch HTTPError from _post
            raise IOError(f"Failed to write file to Daytona '{path}': {e}") from e
