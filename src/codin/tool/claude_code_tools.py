"""Claude Code tools implementation.

Implements all tools specified in docs/claude_code_tools.md with exact signatures.
"""

import asyncio
import fnmatch
import glob as glob_module
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .base import Tool, ToolContext


class TaskArgs(BaseModel):
    description: str = Field(description="Brief description of the task")
    prompt: str = Field(description="Detailed prompt for the agent to execute")


class Task(Tool):
    """Launch an agent to search for config files or perform research tasks."""
    
    def __init__(self):
        super().__init__(
            name="Task",
            description="Launch an agent to search for config files or perform research tasks",
            input_schema=TaskArgs
        )
    
    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        """Execute task by launching an agent."""
        validated = self.validate_input(args)
        
        # Placeholder implementation - would launch actual sub-agent
        return {
            "task_id": "task_123",
            "description": validated["description"],
            "status": "completed",
            "findings": f"Executed task: {validated['description']}",
            "details": f"Agent processed prompt: {validated['prompt']}"
        }


class BashArgs(BaseModel):
    command: str = Field(description="Shell command to execute")
    description: str | None = Field(None, description="Optional description of what the command does")
    timeout: float | None = Field(None, description="Optional timeout in seconds (max 600)")


class Bash(Tool):
    """Execute shell commands."""
    
    def __init__(self):
        super().__init__(
            name="Bash",
            description="Execute shell commands",
            input_schema=BashArgs
        )
    
    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        """Execute bash command."""
        validated = self.validate_input(args)
        command = validated["command"]
        timeout = validated.get("timeout", ctx.timeout)
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=ctx.working_dir
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return {
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "description": validated.get("description")
            }
            
        except TimeoutError:
            raise RuntimeError(f"Command timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Command failed: {e}")


class GlobArgs(BaseModel):
    pattern: str = Field(description="Glob pattern to match files against")
    path: str | None = Field(None, description="Directory to search in (defaults to current directory)")


class Glob(Tool):
    """Find files by pattern."""
    
    def __init__(self):
        super().__init__(
            name="Glob",
            description="Find files by pattern",
            input_schema=GlobArgs
        )
    
    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> list[str]:
        """Find files matching glob pattern."""
        validated = self.validate_input(args)
        pattern = validated["pattern"]
        search_path = validated.get("path", ctx.working_dir)
        
        old_cwd = os.getcwd()
        try:
            os.chdir(search_path)
            matches = glob_module.glob(pattern, recursive=True)
            matches.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
            return [os.path.abspath(match) for match in matches]
        finally:
            os.chdir(old_cwd)


class GrepArgs(BaseModel):
    pattern: str = Field(description="Regular expression pattern to search for")
    include: str | None = Field(None, description="File pattern to include in search (e.g. '*.js', '*.{ts,tsx}')")
    path: str | None = Field(None, description="Directory to search in (defaults to current directory)")


class Grep(Tool):
    """Search file contents using regular expressions."""
    
    def __init__(self):
        super().__init__(
            name="Grep",
            description="Search file contents using regular expressions",
            input_schema=GrepArgs
        )
    
    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> list[str]:
        """Search file contents with regex."""
        validated = self.validate_input(args)
        pattern = validated["pattern"]
        include = validated.get("include")
        search_path = validated.get("path", ctx.working_dir)
        
        # Find files to search
        if include:
            file_pattern = os.path.join(search_path, "**", include)
            files_to_search = glob_module.glob(file_pattern, recursive=True)
        else:
            file_pattern = os.path.join(search_path, "**", "*")
            all_files = glob_module.glob(file_pattern, recursive=True)
            files_to_search = [f for f in all_files if os.path.isfile(f)]
        
        matches = []
        regex = re.compile(pattern)
        
        for file_path in files_to_search:
            try:
                with open(file_path, encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if regex.search(content):
                        matches.append(file_path)
            except (PermissionError, UnicodeDecodeError, IsADirectoryError):
                continue
        
        matches.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
        return matches


class LSArgs(BaseModel):
    path: str = Field(description="Directory path to list")
    ignore: list[str] | None = Field(None, description="List of glob patterns to ignore")


class LS(Tool):
    """List directory contents."""
    
    def __init__(self):
        super().__init__(
            name="LS",
            description="List directory contents",
            input_schema=LSArgs
        )
    
    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> list[dict[str, Any]]:
        """List directory contents."""
        validated = self.validate_input(args)
        path = validated["path"]
        ignore_patterns = validated.get("ignore", [])
        
        target_path = Path(path)
        if not target_path.is_absolute():
            target_path = Path(ctx.working_dir) / target_path
        
        if not target_path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not target_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        
        entries = []
        for entry in target_path.iterdir():
            # Check if entry should be ignored
            should_ignore = any(fnmatch.fnmatch(entry.name, pattern) for pattern in ignore_patterns)
            if should_ignore:
                continue
            
            stat = entry.stat()
            entry_info = {
                "name": entry.name,
                "path": str(entry),
                "type": "directory" if entry.is_dir() else "file",
                "size": stat.st_size if entry.is_file() else None,
                "modified": stat.st_mtime,
                "permissions": oct(stat.st_mode)[-3:]
            }
            entries.append(entry_info)
        
        entries.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))
        return entries


class ReadArgs(BaseModel):
    file_path: str = Field(description="Path to the file to read")
    offset: int | None = Field(None, description="Line number to start reading from (0-based)")
    limit: int | None = Field(None, description="Maximum number of lines to read")


class Read(Tool):
    """Read file contents."""
    
    def __init__(self):
        super().__init__(
            name="Read",
            description="Read file contents",
            input_schema=ReadArgs
        )
    
    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> str:
        """Read file contents."""
        validated = self.validate_input(args)
        file_path = validated["file_path"]
        offset = validated.get("offset", 0)
        limit = validated.get("limit", 2000)
        
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = Path(ctx.working_dir) / target_path
        
        if not target_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        if not target_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {file_path}")
        
        # Try to read with different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(target_path, encoding=encoding) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError(f"Could not decode file: {file_path}")
        
        # Apply offset and limit
        if offset >= len(lines):
            selected_lines = []
        else:
            end_line = min(offset + limit, len(lines))
            selected_lines = lines[offset:end_line]
        
        # Format with line numbers
        formatted_lines = []
        for i, line in enumerate(selected_lines, start=offset + 1):
            if len(line) > 2000:
                line = line[:2000] + "...\n"
            formatted_lines.append(f"{i:5}→{line}")
        
        return ''.join(formatted_lines)


# Edit tool implementation
class EditArgs(BaseModel):
    file_path: str = Field(description="Path to the file to edit")
    old_string: str = Field(description="Text to replace")
    new_string: str = Field(description="Replacement text")
    replace_all: bool | None = Field(False, description="Replace all occurrences")


class Edit(Tool):
    """Edit file contents by replacing text."""
    
    def __init__(self):
        super().__init__(
            name="Edit",
            description="Edit file contents by replacing text",
            input_schema=EditArgs
        )
    
    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        """Edit file by replacing text."""
        validated = self.validate_input(args)
        file_path = validated["file_path"]
        old_string = validated["old_string"]
        new_string = validated["new_string"]
        replace_all = validated.get("replace_all", False)
        
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = Path(ctx.working_dir) / target_path
        
        if not target_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        # Read file content
        content = target_path.read_text(encoding='utf-8')
        
        # Perform replacement
        if replace_all:
            new_content = content.replace(old_string, new_string)
            replacements = content.count(old_string)
        else:
            if old_string not in content:
                raise ValueError(f"String not found in file: {old_string}")
            new_content = content.replace(old_string, new_string, 1)
            replacements = 1
        
        # Write back to file
        target_path.write_text(new_content, encoding='utf-8')
        
        return {
            "file_path": str(target_path),
            "replacements_made": replacements,
            "replace_all": replace_all
        }


# Write tool implementation  
class WriteArgs(BaseModel):
    file_path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")


class Write(Tool):
    """Write content to a file."""
    
    def __init__(self):
        super().__init__(
            name="Write",
            description="Write content to a file",
            input_schema=WriteArgs
        )
    
    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        """Write content to file."""
        validated = self.validate_input(args)
        file_path = validated["file_path"]
        content = validated["content"]
        
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = Path(ctx.working_dir) / target_path
        
        # Create parent directories if they don't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content to file
        target_path.write_text(content, encoding='utf-8')
        
        return {
            "file_path": str(target_path),
            "bytes_written": len(content.encode('utf-8'))
        }


# Claude Code tool registry
def create_claude_code_tools() -> list[Tool]:
    """Create all Claude Code tools."""
    return [
        Task(),
        Bash(),
        Glob(),
        Grep(),
        LS(),
        Read(),
        Edit(),
        Write(),
    ]


def get_claude_code_tool(name: str) -> Tool | None:
    """Get Claude Code tool by name."""
    tools = {tool.name: tool for tool in create_claude_code_tools()}
    return tools.get(name)


def get_claude_code_toolset() -> 'Toolset':
    """Get Claude Code toolset."""
    from .base import Toolset
    return Toolset(name="claude_code", tools=create_claude_code_tools())