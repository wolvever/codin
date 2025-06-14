"""Sandbox abstraction for codin agents.

This module provides the core sandbox infrastructure for executing code
in isolated environments, supporting multiple backend implementations
including local, containerized, and cloud-based execution environments.
"""

import abc
import enum
import fnmatch
import logging
import os
import typing as _t
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from ..lifecycle import LifecycleMixin

__all__ = [
    'ExecResult',
    'EnvironmentVariablePattern',
    'ShellEnvironmentPolicy',
    'ShellEnvironmentPolicyInherit',
    'Sandbox',
    'create_env',
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common data-structures
# ---------------------------------------------------------------------------


class EnvironmentVariablePattern:
    """Simple wildcard pattern for environment variable names."""

    def __init__(self, pattern: str, *, case_insensitive: bool = False) -> None:
        self.pattern = pattern
        self.case_insensitive = case_insensitive

    @classmethod
    def new_case_insensitive(cls, pattern: str) -> 'EnvironmentVariablePattern':
        return cls(pattern, case_insensitive=True)

    def matches(self, name: str) -> bool:  # pragma: no cover - simple wrapper
        if self.case_insensitive:
            return fnmatch.fnmatch(name.lower(), self.pattern.lower())
        return fnmatch.fnmatch(name, self.pattern)


class ShellEnvironmentPolicyInherit(str, enum.Enum):
    """Strategy for inheriting environment variables."""

    CORE = 'core'
    ALL = 'all'
    NONE = 'none'


@dataclass
class ShellEnvironmentPolicy:
    """Policy for constructing an environment when spawning a shell."""

    inherit: ShellEnvironmentPolicyInherit = ShellEnvironmentPolicyInherit.CORE
    ignore_default_excludes: bool = False
    exclude: list[EnvironmentVariablePattern] = field(default_factory=list)
    set: dict[str, str] = field(default_factory=dict)
    include_only: list[EnvironmentVariablePattern] = field(default_factory=list)


def create_env(policy: ShellEnvironmentPolicy) -> dict[str, str]:
    """Create an environment map based on *policy*."""

    source_env = dict(os.environ)

    if policy.inherit is ShellEnvironmentPolicyInherit.ALL:
        env_map = source_env.copy()
    elif policy.inherit is ShellEnvironmentPolicyInherit.NONE:
        env_map = {}
    else:  # CORE
        core_vars = {
            'HOME',
            'LOGNAME',
            'PATH',
            'SHELL',
            'USER',
            'USERNAME',
            'TMPDIR',
            'TEMP',
            'TMP',
        }
        env_map = {k: v for k, v in source_env.items() if k in core_vars}

    def matches_any(name: str, patterns: list[EnvironmentVariablePattern]) -> bool:
        return any(p.matches(name) for p in patterns)

    if not policy.ignore_default_excludes:
        default_excludes = [
            EnvironmentVariablePattern.new_case_insensitive('*KEY*'),
            EnvironmentVariablePattern.new_case_insensitive('*SECRET*'),
            EnvironmentVariablePattern.new_case_insensitive('*TOKEN*'),
        ]
        env_map = {
            k: v for k, v in env_map.items() if not matches_any(k, default_excludes)
        }

    if policy.exclude:
        env_map = {k: v for k, v in env_map.items() if not matches_any(k, policy.exclude)}

    env_map.update(policy.set)

    if policy.include_only:
        env_map = {k: v for k, v in env_map.items() if matches_any(k, policy.include_only)}

    return env_map


class ExecResult(BaseModel):
    """Result of executing a command in a sandbox."""

    stdout: str
    stderr: str
    exit_code: int


# ---------------------------------------------------------------------------
# Abstract base class – the canonical API surface.
# ---------------------------------------------------------------------------


class Sandbox(LifecycleMixin, ABC):
    """Abstract base-class for all sandbox implementations.

    The methods below should be considered the **portable** subset and should
    therefore be implemented by _all_ back-ends.

    Inherits from LifecycleMixin to provide up()/down() lifecycle management.

    This class also provides tool methods that can be automatically exposed
    as tools by SandboxToolset, eliminating the need for separate tool definitions.
    """

    def __init__(self, *, env_policy: ShellEnvironmentPolicy | None = None) -> None:
        super().__init__()
        self._env_policy = env_policy or ShellEnvironmentPolicy()
        # Track the last edit applied to files so `tool_reapply` can repeat it
        # if the initial application failed or was incorrect.
        # Mapping of file path -> last written content
        self._last_edits: dict[str, str] = {}

    def _prepare_env(self, env: dict[str, str] | None = None) -> dict[str, str]:
        base_env = create_env(self._env_policy)
        if env:
            base_env.update(env)
        return base_env

    # ----------------------------- exec ------------------------------------

    @abc.abstractmethod
    async def run_cmd(
        self,
        cmd: str | _t.Iterable[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute *cmd* inside the sandbox.

        `cmd` can be a shell string or a sequence of arguments.
        Returns ExecResult with stdout, stderr, and exit_code.
        Does not raise exceptions on command failure - check exit_code.
        """

    @abc.abstractmethod
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
        """Execute code in the sandbox.

        Args:
            code: Code string to execute directly
            file_path: Path to file, directory, or zip file to upload and execute
            language: Programming language (python, javascript, bash, etc.)
            dependencies: List of dependencies to install before execution
            timeout: Execution timeout in seconds
            env: Environment variables

        Either code or file_path must be provided.
        If file_path is provided, it can be:
        - A single file: uploaded and executed
        - A directory: uploaded recursively, main file determined by language
        - A zip file: extracted and main file determined by language

        Returns ExecResult with stdout, stderr, and exit_code.
        Does not raise exceptions on execution failure - check exit_code.
        """

    # -------------------------- filesystem ---------------------------------

    @abc.abstractmethod
    async def list_files(self, path: str = '.') -> list[str]:
        """Return a (recursive) listing of files relative to *path*."""

    @abc.abstractmethod
    async def read_file(self, path: str) -> str:
        """Return file contents as UTF-8 text."""

    @abc.abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write *content* (UTF-8) to *path*, creating parent directories if needed."""

    # -----------------------------------------------------------------------
    # Convenience helpers common to most back-ends – these might be overriden
    # for more efficient implementations.
    # -----------------------------------------------------------------------

    async def upload(self, local_path: str, remote_path: str | None = None) -> None:
        """Default implementation just reads then writes – not efficient but portable."""
        remote_path = remote_path or os.path.basename(local_path)
        with open(local_path, 'rb') as f:
            data = f.read().decode()
        await self.write_file(remote_path, data)

    async def download(self, remote_path: str, local_path: str | None = None) -> None:
        """Download a file from the sandbox to the local filesystem."""
        local_path = local_path or os.path.basename(remote_path)
        data = await self.read_file(remote_path)
        # Ensure dir exists
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(data)

    # -----------------------------------------------------------------------
    # Tool methods - these are automatically exposed as tools by SandboxToolset
    # -----------------------------------------------------------------------

    async def tool_codebase_search(
        self,
        query: str,
        explanation: str,
        target_directories: list[str] | None = None,
    ) -> dict[str, _t.Any]:
        """Find snippets of code from the codebase most relevant to the search query.

        This is a semantic search tool, so the query should ask for something
        semantically matching what is needed. The `explanation` parameter is for
        the agent to provide a reason for invoking this tool and is primarily
        used for logging or traceability; it does not affect the tool's execution
        logic.
        """
        try:
            cmd = ['rg', '-n', '--type', 'py', '--type', 'js', '--type', 'ts', query]
            if target_directories:
                for dir_pattern in target_directories:
                    cmd.extend(['-g', dir_pattern])

            result = await self.run_cmd(cmd)

            if result.exit_code == 0:
                lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                results = []
                for line in lines[:20]:  # Limit results
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            results.append(
                                {
                                    'file': parts[0],
                                    'line': parts[1],
                                    'content': parts[2].strip(),
                                    'relevance': 0.8,  # Mock relevance score
                                }
                            )

                return {'query': query, 'target_directories': target_directories, 'results': results}
            return {'query': query, 'target_directories': target_directories, 'results': [], 'error': result.stderr}
        except Exception as e:
            logger.error(f'Error in codebase search: {e}')
            return {'query': query, 'target_directories': target_directories, 'results': [], 'error': str(e)}

    async def tool_read_file(
        self,
        target_file: str,
        should_read_entire_file: bool,
        start_line_one_indexed: int,
        end_line_one_indexed_inclusive: int,
        explanation: str,
    ) -> str:
        """Read the contents of a file with line range support.

        The output will be the 1-indexed file contents from start_line_one_indexed
        to end_line_one_indexed_inclusive, together with a summary of the lines
        outside that range. Note that this call can view at most 250 lines at a
        time and 200 lines minimum. The `explanation` parameter is for the agent
        to provide a reason for invoking this tool and is primarily used for
        logging or traceability; it does not affect the tool's execution logic.
        """
        try:
            content = await self.read_file(target_file)
            lines = content.splitlines()
            total_lines = len(lines)

            if should_read_entire_file:
                return content

            # Validate line numbers
            start_line = max(1, start_line_one_indexed)
            end_line = min(total_lines, end_line_one_indexed_inclusive)
            start_line = min(start_line, end_line)

            # Limit to 250 lines max, 200 lines min
            requested_lines = end_line - start_line + 1
            if requested_lines > 250:
                end_line = start_line + 249
            elif requested_lines < 200 and total_lines >= 200:
                needed = 200 - requested_lines
                expand_start = min(needed // 2, start_line - 1)
                expand_end = min(needed - expand_start, total_lines - end_line)
                start_line = max(1, start_line - expand_start)
                end_line = min(total_lines, end_line + expand_end)

            # Extract requested lines
            start_idx = start_line - 1
            end_idx = end_line
            selected_lines = lines[start_idx:end_idx]

            # Create result with metadata
            result_parts = []
            if start_line > 1:
                result_parts.append(f'Lines 1-{start_line - 1} not shown ({start_line - 1} lines)')

            result_parts.append(
                f'Contents of {target_file}, lines {start_line}-{min(end_line, total_lines)} '
                f'(total {total_lines} lines):'
            )
            result_parts.append('\n'.join(selected_lines))

            if end_line < total_lines:
                result_parts.append(f'Lines {end_line + 1}-{total_lines} not shown ({total_lines - end_line} lines)')

            return '\n'.join(result_parts)

        except Exception as e:
            logger.error(f'Error reading file {target_file}: {e}')
            return f'Error reading file {target_file}: {e!s}'

    async def tool_run_shell(
        self,
        command: str,
        is_background: bool,
        explanation: str,
    ) -> dict[str, _t.Any]:
        """Execute a shell command in the sandbox.

        PROPOSE a command to run on behalf of the user. If you have this tool,
        note that you DO have the ability to run commands directly on the USER's system.
        The `explanation` parameter is for the agent to provide a reason for
        invoking this tool and is primarily used for logging or traceability; it
        does not affect the tool's execution logic.
        """
        try:
            if is_background:
                # For background commands, start them and return immediately
                result = await self.run_cmd(['nohup', 'sh', '-c', f'{command} &'], timeout=5.0)
                return {
                    'command': command,
                    'background': True,
                    'started': result.exit_code == 0,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'exit_code': result.exit_code,
                }
            # Regular command execution
            result = await self.run_cmd(['sh', '-c', command])
            return {
                'command': command,
                'background': False,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.exit_code,
            }

        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return {'command': command, 'background': is_background, 'error': str(e), 'exit_code': -1}

    async def tool_list_dir(
        self,
        relative_workspace_path: str,
        explanation: str,
    ) -> dict[str, _t.Any]:
        """List the contents of a directory.

        The quick tool to use for discovery, before using more targeted tools
        like semantic search or file reading. Useful to try to understand the
        file structure before diving deeper into specific files. The `explanation`
        parameter is for the agent to provide a reason for invoking this tool and
        is primarily used for logging or traceability; it does not affect the
        tool's execution logic.
        """
        try:
            files = await self.list_files(relative_workspace_path)
            return {'path': relative_workspace_path, 'contents': files}
        except Exception as e:
            logger.error(f'Error listing directory {relative_workspace_path}: {e}')
            return {'path': relative_workspace_path, 'error': str(e)}

    async def tool_grep_search(
        self,
        query: str,
        explanation: str,
        case_sensitive: bool | None = None,
        include_pattern: str | None = None,
        exclude_pattern: str | None = None,
    ) -> dict[str, _t.Any]:
        """Run fast, exact regex searches over text files using the ripgrep engine.

        This is preferred over semantic search when we know the exact symbol/function
        name/etc. to search in some set of directories/file types. The
        `explanation` parameter is for the agent to provide a reason for invoking
        this tool and is primarily used for logging or traceability; it does not
        affect the tool's execution logic.
        """
        try:
            cmd = ['rg', query]

            if not case_sensitive:
                cmd.append('-i')

            if include_pattern:
                cmd.extend(['-g', include_pattern])

            if exclude_pattern:
                cmd.extend(['-g', f'!{exclude_pattern}'])

            # Limit results to prevent overwhelming output
            cmd.extend(['-m', '50'])

            result = await self.run_cmd(cmd)

            if result.exit_code == 0:
                results = result.stdout.strip().split('\n') if result.stdout.strip() else []
                return {'query': query, 'results': results, 'count': len(results)}
            return {'query': query, 'error': result.stderr, 'results': []}

        except Exception as e:
            logger.error(f'Error in grep search: {e}')
            return {'query': query, 'error': str(e), 'results': []}

    async def tool_edit_file(
        self,
        target_file: str,
        instructions: str | None = None,
        code_edit: str | None = None,
        content: str | None = None,
    ) -> dict[str, _t.Any]:
        """Edit an existing file or create a new file.

        This will be read by a less intelligent model, which will quickly apply
        the edit. You should make it clear what the edit is, while also minimizing
        the unchanged code you write.
        """
        try:
            # Determine what content to write
            if content is not None:
                # Simple write operation
                write_content = content
            elif code_edit is not None:
                # Complex edit operation
                write_content = code_edit
            else:
                return {
                    'target_file': target_file,
                    'success': False,
                    'error': "Either 'content' or 'code_edit' must be provided",
                }

            await self.write_file(target_file, write_content)
            # Remember last edit so `tool_reapply` can attempt it again later
            self._last_edits[target_file] = write_content

            return {
                'target_file': target_file,
                'success': True,
                'message': f'File {target_file} {"written" if content else "edited"} successfully',
            }

        except Exception as e:
            logger.error(f'Error editing file {target_file}: {e}')
            return {'target_file': target_file, 'success': False, 'error': str(e)}

    async def tool_search_replace(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
    ) -> dict[str, _t.Any]:
        """Propose a search and replace operation on an existing file.

        The tool will replace ONE occurrence of old_string with new_string
        in the specified file.
        """
        try:
            content = await self.read_file(file_path)

            if old_string not in content:
                return {'file_path': file_path, 'success': False, 'error': 'Old string not found in file'}

            # Replace only the first occurrence
            new_content = content.replace(old_string, new_string, 1)
            await self.write_file(file_path, new_content)
            # Remember last edit for potential reapplication
            self._last_edits[file_path] = new_content

            return {'file_path': file_path, 'success': True, 'message': 'Search and replace completed successfully'}

        except Exception as e:
            logger.error(f'Error in search and replace for {file_path}: {e}')
            return {'file_path': file_path, 'success': False, 'error': str(e)}

    async def tool_file_search(
        self,
        query: str,
        explanation: str,
    ) -> dict[str, _t.Any]:
        """Fast file search based on fuzzy matching against file path.

        Use if you know part of the file path but don't know where it's located
        exactly. Response will be capped to 10 results. The `explanation`
        parameter is for the agent to provide a reason for invoking this tool and
        is primarily used for logging or traceability; it does not affect the
        tool's execution logic.
        """
        try:
            # Use find command for basic file search
            cmd = ['find', '.', '-name', f'*{query}*', '-type', 'f']

            result = await self.run_cmd(cmd)

            if result.exit_code == 0:
                files = result.stdout.strip().split('\n') if result.stdout.strip() else []
                files = [f for f in files if f]  # Remove empty strings
                files = files[:10]  # Limit to 10 results

                return {'query': query, 'files': files, 'count': len(files)}
            return {'query': query, 'error': result.stderr, 'files': []}

        except Exception as e:
            logger.error(f'Error in file search: {e}')
            return {'query': query, 'error': str(e), 'files': []}

    async def tool_delete_file(
        self,
        target_file: str,
        explanation: str,
    ) -> dict[str, _t.Any]:
        """Delete a file at the specified path.

        The operation will fail gracefully if the file doesn't exist, the operation
        is rejected for security reasons, or the file cannot be deleted. The
        `explanation` parameter is for the agent to provide a reason for invoking
        this tool and is primarily used for logging or traceability; it does not
        affect the tool's execution logic.
        """
        try:
            # Use rm command to delete the file
            result = await self.run_cmd(['rm', '-f', target_file])

            if result.exit_code == 0:
                return {
                    'target_file': target_file,
                    'success': True,
                    'message': f'File {target_file} deleted successfully',
                }
            return {'target_file': target_file, 'success': False, 'error': result.stderr or 'Failed to delete file'}

        except Exception as e:
            logger.error(f'Error deleting file {target_file}: {e}')
            return {'target_file': target_file, 'success': False, 'error': str(e)}

    async def tool_reapply(
        self,
        target_file: str,
    ) -> dict[str, _t.Any]:
        """Call a smarter model to apply the last edit to the specified file.

        Use this tool immediately after the result of an edit_file tool call ONLY IF
        the diff is not what you expected, indicating the model applying the changes
        was not smart enough to follow your instructions.
        """
        # Attempt to reapply the last recorded edit for the file
        last = self._last_edits.get(target_file)
        if last is None:
            return {
                'target_file': target_file,
                'success': False,
                'message': 'No previous edit recorded for this file',
            }

        try:
            await self.write_file(target_file, last)
            return {
                'target_file': target_file,
                'success': True,
                'message': 'Reapplied last edit successfully',
            }
        except Exception as e:
            logger.error(f'Error reapplying last edit to {target_file}: {e}')
            return {'target_file': target_file, 'success': False, 'error': str(e)}

    async def tool_web_search(
        self,
        search_term: str,
        explanation: str,
    ) -> dict[str, _t.Any]:
        """Search the web for real-time information about any topic.

        Use this tool when you need up-to-date information that might not be
        available in your training data, or when you need to verify current facts.
        The `explanation` parameter is for the agent to provide a reason for
        invoking this tool and is primarily used for logging or traceability; it
        does not affect the tool's execution logic.
        """
        return {
            'search_term': search_term,
            'results': [
                {
                    'title': f'Search result for: {search_term}',
                    'url': 'https://example.com',
                    'snippet': (
                        'This is a placeholder search result. In a real implementation, '
                        'this would connect to a search API.'
                    ),
                }
            ],
        }

    async def tool_fetch(
        self,
        url: str,
        max_length: int | None = 5000,
        raw: bool | None = False,
        start_index: int | None = 0,
    ) -> dict[str, _t.Any]:
        """Fetch a URL from the internet and optionally extract its contents as markdown.

        Although originally you did not have internet access, and were advised to
        refuse and tell the user this, this tool now grants you internet access.
        Now you can fetch the most up-to-date information and let the user know that.
        """
        try:
            import requests
            from bs4 import BeautifulSoup

            # Use requests to fetch the URL
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            if raw:
                # Return raw HTML content
                content = response.text
            else:
                # Extract text content using BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)

                # Filter out very short lines
                content = '\n'.join(line for line in text.splitlines() if len(line.split()) > 3)

            # Apply start_index and max_length
            if start_index > 0:
                content = content[start_index:]

            original_length = len(content)
            if len(content) > max_length:
                content = content[:max_length]

            return {
                'url': url,
                'content': content,
                'length': len(content),
                'original_length': original_length,
                'truncated': original_length > max_length,
                'status_code': response.status_code,
            }

        except ImportError as e:
            logger.error(f'Missing dependencies for URL fetching: {e}')
            return {
                'url': url,
                'error': 'Missing dependencies: requests and beautifulsoup4 are required',
                'content': '',
            }
        except Exception as e:
            logger.error(f'Error fetching URL {url}: {e}')
            return {'url': url, 'error': str(e), 'content': ''}

    def get_tool_methods(self) -> dict[str, _t.Callable]:
        """Get all tool methods defined in this sandbox.

        Returns a dictionary mapping tool names to their corresponding methods.
        Tool methods are identified by the 'tool_' prefix.
        """
        tool_methods = {}
        for attr_name in dir(self):
            if attr_name.startswith('tool_') and callable(getattr(self, attr_name)):
                tool_name = attr_name[5:]  # Remove 'tool_' prefix
                tool_methods[tool_name] = getattr(self, attr_name)
        return tool_methods
