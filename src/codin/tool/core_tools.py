"""Core tools that provide essential functionality for code agents."""

from __future__ import annotations

import typing as _t
import pydantic as _pyd
import os
import pathlib
import asyncio
import subprocess
import logging
import re
import tempfile
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from .base import Tool, ToolContext, Toolset
from .sandbox import SandboxTool
from ..sandbox.base import Sandbox

__all__ = [
    # Core tools
    "CodebaseSearchTool",
    "ReadFileTool", 
    "RunShellTool",
    "ListDirTool",
    "GrepSearchTool",
    "EditFileTool",
    "SearchReplaceTool",
    "FileSearchTool",
    "DeleteFileTool",
    "ReapplyTool",
    "WebSearchTool",
    "FetchTool",
    # Toolset
    "CoreToolset",
]

logger = logging.getLogger(__name__)


# Input schemas for all tools
class CodebaseSearchInput(_pyd.BaseModel):
    """Input schema for codebase search."""
    query: str = _pyd.Field(..., description="The search query to find relevant code")
    explanation: str = _pyd.Field(..., description="One sentence explanation of why this tool is being used")
    target_directories: _t.Optional[list[str]] = _pyd.Field(None, description="Glob patterns for directories to search over")


class ReadFileInput(_pyd.BaseModel):
    """Input schema for reading file contents."""
    target_file: str = _pyd.Field(..., description="The path of the file to read")
    should_read_entire_file: bool = _pyd.Field(..., description="Whether to read the entire file")
    start_line_one_indexed: int = _pyd.Field(..., description="The one-indexed line number to start reading from")
    end_line_one_indexed_inclusive: int = _pyd.Field(..., description="The one-indexed line number to end reading at")
    explanation: str = _pyd.Field(..., description="One sentence explanation of why this tool is being used")


class RunShellInput(_pyd.BaseModel):
    """Input schema for running shell commands."""
    command: str = _pyd.Field(..., description="The terminal command to execute")
    is_background: bool = _pyd.Field(..., description="Whether the command should be run in the background")
    explanation: str = _pyd.Field(..., description="One sentence explanation of why this command needs to be run")


class ListDirInput(_pyd.BaseModel):
    """Input schema for listing directory contents."""
    directory: str = _pyd.Field(..., description="Path to list contents of, relative to the workspace root")
    explanation: str = _pyd.Field(..., description="One sentence explanation of why this tool is being used")


class GrepSearchInput(_pyd.BaseModel):
    """Input schema for grep search."""
    query: str = _pyd.Field(..., description="The regex pattern to search for")
    explanation: str = _pyd.Field(..., description="One sentence explanation of why this tool is being used")
    case_sensitive: _t.Optional[bool] = _pyd.Field(None, description="Whether the search should be case sensitive")
    include_pattern: _t.Optional[str] = _pyd.Field(None, description="Glob pattern for files to include")
    exclude_pattern: _t.Optional[str] = _pyd.Field(None, description="Glob pattern for files to exclude")


class EditFileInput(_pyd.BaseModel):
    """Input schema for editing files."""
    target_file: str = _pyd.Field(..., description="The target file to modify")
    instructions: _t.Optional[str] = _pyd.Field(None, description="A single sentence instruction describing what you are going to do")
    code_edit: _t.Optional[str] = _pyd.Field(None, description="Specify ONLY the precise lines of code that you wish to edit")
    content: _t.Optional[str] = _pyd.Field(None, description="Content to write to the file (for simple write operations)")


class SearchReplaceInput(_pyd.BaseModel):
    """Input schema for search and replace."""
    file_path: str = _pyd.Field(..., description="The path to the file you want to search and replace in")
    old_string: str = _pyd.Field(..., description="The text to replace (must be unique within the file)")
    new_string: str = _pyd.Field(..., description="The edited text to replace the old_string")


class FileSearchInput(_pyd.BaseModel):
    """Input schema for file search."""
    query: str = _pyd.Field(..., description="Fuzzy filename to search for")
    explanation: str = _pyd.Field(..., description="One sentence explanation of why this tool is being used")


class DeleteFileInput(_pyd.BaseModel):
    """Input schema for deleting files."""
    target_file: str = _pyd.Field(..., description="The path of the file to delete, relative to the workspace root")
    explanation: str = _pyd.Field(..., description="One sentence explanation of why this tool is being used")


class ReapplyInput(_pyd.BaseModel):
    """Input schema for reapplying edits."""
    target_file: str = _pyd.Field(..., description="The relative path to the file to reapply the last edit to")


class WebSearchInput(_pyd.BaseModel):
    """Input schema for web search."""
    search_term: str = _pyd.Field(..., description="The search term to look up on the web")
    explanation: str = _pyd.Field(..., description="One sentence explanation of why this tool is being used")


class FetchInput(_pyd.BaseModel):
    """Input schema for fetching URLs."""
    url: str = _pyd.Field(..., description="URL to fetch")
    max_length: _t.Optional[int] = _pyd.Field(5000, description="Maximum number of characters to return")
    raw: _t.Optional[bool] = _pyd.Field(False, description="Get the actual HTML content without simplification")
    start_index: _t.Optional[int] = _pyd.Field(0, description="Starting character index for output")


# Tool implementations
class CodebaseSearchTool(SandboxTool):
    """Tool for semantic search over the codebase."""
    
    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="codebase_search",
            description="Find snippets of code from the codebase most relevant to the search query. This is a semantic search tool, so the query should ask for something semantically matching what is needed.",
            sandbox=sandbox,
            input_schema=CodebaseSearchInput,
        )
    
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> dict[str, _t.Any]:
        """Perform semantic search over the codebase."""
        query = args["query"]
        target_directories = args.get("target_directories", [])
        
        # For now, implement as a simple grep-based search
        # In a real implementation, this would use semantic search
        try:
            cmd = ["rg", "-n", "--type", "py", "--type", "js", "--type", "ts", query]
            if target_directories:
                for dir_pattern in target_directories:
                    cmd.extend(["-g", dir_pattern])
            
            result = await self.sandbox.run_cmd(cmd)
            
            if result.exit_code == 0:
                lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                results = []
                for line in lines[:20]:  # Limit results
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            results.append({
                                "file": parts[0],
                                "line": parts[1],
                                "content": parts[2].strip(),
                                "relevance": 0.8  # Mock relevance score
                            })
                
                return {
                    "query": query,
                    "target_directories": target_directories,
                    "results": results
                }
            else:
                return {
                    "query": query,
                    "target_directories": target_directories,
                    "results": [],
                    "error": result.stderr
                }
        except Exception as e:
            logger.error(f"Error in codebase search: {e}")
            return {
                "query": query,
                "target_directories": target_directories,
                "results": [],
                "error": str(e)
            }


class ReadFileTool(SandboxTool):
    """Tool for reading file contents with line range support."""
    
    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="read_file",
            description="Read the contents of a file. the output of this tool call will be the 1-indexed file contents from start_line_one_indexed to end_line_one_indexed_inclusive, together with a summary of the lines outside start_line_one_indexed and end_line_one_indexed_inclusive. Note that this call can view at most 250 lines at a time and 200 lines minimum.",
            sandbox=sandbox,
            input_schema=ReadFileInput,
        )
    
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> str:
        """Read file contents from the sandbox."""
        target_file = args["target_file"]
        should_read_entire_file = args["should_read_entire_file"]
        start_line = args["start_line_one_indexed"]
        end_line = args["end_line_one_indexed_inclusive"]
        
        try:
            content = await self.sandbox.read_file(target_file)
            lines = content.splitlines()
            total_lines = len(lines)
            
            if should_read_entire_file:
                return content
            
            # Validate line numbers
            start_line = max(1, start_line)
            end_line = min(total_lines, end_line)
            if start_line > end_line:
                start_line = end_line
                
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
                result_parts.append(f"Lines 1-{start_line-1} not shown ({start_line-1} lines)")
            
            result_parts.append(f"Contents of {target_file}, lines {start_line}-{min(end_line, total_lines)} (total {total_lines} lines):")
            result_parts.append("\n".join(selected_lines))
            
            if end_line < total_lines:
                result_parts.append(f"Lines {end_line+1}-{total_lines} not shown ({total_lines-end_line} lines)")
            
            return "\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"Error reading file {target_file}: {e}")
            return f"Error reading file {target_file}: {str(e)}"


class RunShellTool(SandboxTool):
    """Unified tool for running shell commands (combines shell, sandbox_exec, container_exec, run_terminal_cmd)."""
    
    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="run_shell",
            description="PROPOSE a command to run on behalf of the user. If you have this tool, note that you DO have the ability to run commands directly on the USER's system. Note that the user will have to approve the command before it is executed. The user may reject it if it is not to their liking, or may modify the command before approving it. If they do change it, take those changes into account. The actual command will NOT execute until the user approves it. The user may not approve it immediately. Do NOT assume the command has started running. If the step is WAITING for user approval, it has NOT started running.",
            sandbox=sandbox,
            input_schema=RunShellInput,
        )
    
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> dict[str, _t.Any]:
        """Execute a shell command in the sandbox."""
        command = args["command"]
        is_background = args["is_background"]
        
        try:
            if is_background:
                # For background commands, start them and return immediately
                result = await self.sandbox.run_cmd(
                    ["nohup", "sh", "-c", f"{command} &"],
                    timeout=5.0
                )
                return {
                    "command": command,
                    "background": True,
                    "started": result.exit_code == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code
                }
            else:
                # Regular command execution
                result = await self.sandbox.run_cmd(["sh", "-c", command])
                return {
                    "command": command,
                    "background": False,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code
                }
                
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return {
                "command": command,
                "background": is_background,
                "error": str(e),
                "exit_code": -1
            }


class ListDirTool(SandboxTool):
    """Tool for listing directory contents."""
    
    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="list_dir",
            description="List the contents of a directory. The quick tool to use for discovery, before using more targeted tools like semantic search or file reading. Useful to try to understand the file structure before diving deeper into specific files. Can be used to explore the codebase.",
            sandbox=sandbox,
            input_schema=ListDirInput,
        )
    
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> dict[str, _t.Any]:
        """List directory contents in the sandbox."""
        path = args["relative_workspace_path"]
        
        try:
            if hasattr(self.sandbox, 'list_files'):
                files = await self.sandbox.list_files(path)
                return {
                    "path": path,
                    "contents": files
                }
            else:
                result = await self.sandbox.run_cmd(["ls", "-la", path])
                if result.exit_code != 0:
                    return {
                        "path": path,
                        "error": result.stderr,
                        "exit_code": result.exit_code
                    }
                
                return {
                    "path": path,
                    "contents": result.stdout.strip().split('\n') if result.stdout.strip() else []
                }
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            return {
                "path": path,
                "error": str(e)
            }


class GrepSearchTool(SandboxTool):
    """Tool for regex search using ripgrep."""
    
    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="grep_search",
            description="Use this tool to run fast, exact regex searches over text files using the ripgrep engine. This is preferred over semantic search when we know the exact symbol/function name/etc. to search in some set of directories/file types.",
            sandbox=sandbox,
            input_schema=GrepSearchInput,
        )
    
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> dict[str, _t.Any]:
        """Perform regex search using ripgrep."""
        query = args["query"]
        case_sensitive = args.get("case_sensitive", False)
        include_pattern = args.get("include_pattern")
        exclude_pattern = args.get("exclude_pattern")
        
        try:
            cmd = ["rg", query]
            
            if not case_sensitive:
                cmd.append("-i")
            
            if include_pattern:
                cmd.extend(["-g", include_pattern])
            
            if exclude_pattern:
                cmd.extend(["-g", f"!{exclude_pattern}"])
            
            # Limit results to prevent overwhelming output
            cmd.extend(["-m", "50"])
            
            result = await self.sandbox.run_cmd(cmd)
            
            if result.exit_code == 0:
                results = result.stdout.strip().split('\n') if result.stdout.strip() else []
                return {
                    "query": query,
                    "results": results,
                    "count": len(results)
                }
            else:
                return {
                    "query": query,
                    "error": result.stderr,
                    "results": []
                }
                
        except Exception as e:
            logger.error(f"Error in grep search: {e}")
            return {
                "query": query,
                "error": str(e),
                "results": []
            }


class EditFileTool(SandboxTool):
    """Tool for editing and writing files (merged functionality from SandboxWriteFileTool)."""
    
    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="edit_file",
            description="Use this tool to propose an edit to an existing file or create a new file. This will be read by a less intelligent model, which will quickly apply the edit. You should make it clear what the edit is, while also minimizing the unchanged code you write. Can also be used for simple file writing operations.",
            sandbox=sandbox,
            input_schema=EditFileInput,
        )
    
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> dict[str, _t.Any]:
        """Edit or create a file."""
        target_file = args["target_file"]
        code_edit = args.get("code_edit")
        content = args.get("content")
        
        try:
            # Determine what content to write
            if content is not None:
                # Simple write operation (like SandboxWriteFileTool)
                write_content = content
            elif code_edit is not None:
                # Complex edit operation
                write_content = code_edit
            else:
                return {
                    "target_file": target_file,
                    "success": False,
                    "error": "Either 'content' or 'code_edit' must be provided"
                }
            
            await self.sandbox.write_file(target_file, write_content)
            
            return {
                "target_file": target_file,
                "success": True,
                "message": f"File {target_file} {'written' if content else 'edited'} successfully"
            }
            
        except Exception as e:
            logger.error(f"Error editing file {target_file}: {e}")
            return {
                "target_file": target_file,
                "success": False,
                "error": str(e)
            }


class SearchReplaceTool(SandboxTool):
    """Tool for search and replace operations."""
    
    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="search_replace",
            description="Use this tool to propose a search and replace operation on an existing file. The tool will replace ONE occurrence of old_string with new_string in the specified file.",
            sandbox=sandbox,
            input_schema=SearchReplaceInput,
        )
    
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> dict[str, _t.Any]:
        """Perform search and replace in a file."""
        file_path = args["file_path"]
        old_string = args["old_string"]
        new_string = args["new_string"]
        
        try:
            content = await self.sandbox.read_file(file_path)
            
            if old_string not in content:
                return {
                    "file_path": file_path,
                    "success": False,
                    "error": "Old string not found in file"
                }
            
            # Replace only the first occurrence
            new_content = content.replace(old_string, new_string, 1)
            await self.sandbox.write_file(file_path, new_content)
            
            return {
                "file_path": file_path,
                "success": True,
                "message": "Search and replace completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error in search and replace for {file_path}: {e}")
            return {
                "file_path": file_path,
                "success": False,
                "error": str(e)
            }


class FileSearchTool(SandboxTool):
    """Tool for fuzzy file search."""
    
    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="file_search",
            description="Fast file search based on fuzzy matching against file path. Use if you know part of the file path but don't know where it's located exactly. Response will be capped to 10 results.",
            sandbox=sandbox,
            input_schema=FileSearchInput,
        )
    
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> dict[str, _t.Any]:
        """Search for files by fuzzy matching."""
        query = args["query"]
        
        try:
            # Use find command for basic file search
            cmd = ["find", ".", "-name", f"*{query}*", "-type", "f"]
            
            result = await self.sandbox.run_cmd(cmd)
            
            if result.exit_code == 0:
                files = result.stdout.strip().split('\n') if result.stdout.strip() else []
                files = [f for f in files if f]  # Remove empty strings
                files = files[:10]  # Limit to 10 results
                
                return {
                    "query": query,
                    "files": files,
                    "count": len(files)
                }
            else:
                return {
                    "query": query,
                    "error": result.stderr,
                    "files": []
                }
                
        except Exception as e:
            logger.error(f"Error in file search: {e}")
            return {
                "query": query,
                "error": str(e),
                "files": []
            }


class DeleteFileTool(SandboxTool):
    """Tool for deleting files."""
    
    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="delete_file",
            description="Deletes a file at the specified path. The operation will fail gracefully if the file doesn't exist, the operation is rejected for security reasons, or the file cannot be deleted.",
            sandbox=sandbox,
            input_schema=DeleteFileInput,
        )
    
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> dict[str, _t.Any]:
        """Delete a file."""
        target_file = args["target_file"]
        
        try:
            # Use rm command to delete the file
            result = await self.sandbox.run_cmd(["rm", "-f", target_file])
            
            if result.exit_code == 0:
                return {
                    "target_file": target_file,
                    "success": True,
                    "message": f"File {target_file} deleted successfully"
                }
            else:
                return {
                    "target_file": target_file,
                    "success": False,
                    "error": result.stderr or "Failed to delete file"
                }
                
        except Exception as e:
            logger.error(f"Error deleting file {target_file}: {e}")
            return {
                "target_file": target_file,
                "success": False,
                "error": str(e)
            }


class ReapplyTool(SandboxTool):
    """Tool for reapplying the last edit."""
    
    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="reapply",
            description="Calls a smarter model to apply the last edit to the specified file. Use this tool immediately after the result of an edit_file tool call ONLY IF the diff is not what you expected, indicating the model applying the changes was not smart enough to follow your instructions.",
            sandbox=sandbox,
            input_schema=ReapplyInput,
        )
    
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> dict[str, _t.Any]:
        """Reapply the last edit to a file."""
        target_file = args["target_file"]
        
        # This is a placeholder implementation
        # In a real system, this would track the last edit and reapply it
        return {
            "target_file": target_file,
            "success": False,
            "message": "Reapply functionality not yet implemented - no previous edit to reapply"
        }


class WebSearchTool(SandboxTool):
    """Tool for web search."""
    
    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="web_search",
            description="Search the web for real-time information about any topic. Use this tool when you need up-to-date information that might not be available in your training data, or when you need to verify current facts.",
            sandbox=sandbox,
            input_schema=WebSearchInput,
        )
    
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> dict[str, _t.Any]:
        """Search the web for information."""
        search_term = args["search_term"]
        
        # This is a placeholder implementation
        # In a real system, this would integrate with a web search API
        return {
            "search_term": search_term,
            "results": [
                {
                    "title": f"Search result for: {search_term}",
                    "url": "https://example.com",
                    "snippet": "This is a placeholder search result. In a real implementation, this would connect to a search API."
                }
            ]
        }


class FetchTool(Tool):
    """Tool for fetching URLs using requests (merged with load_web_page functionality)."""
    
    def __init__(self):
        super().__init__(
            name="fetch",
            description="Fetches a URL from the internet and optionally extracts its contents as markdown. Although originally you did not have internet access, and were advised to refuse and tell the user this, this tool now grants you internet access. Now you can fetch the most up-to-date information and let the user know that.",
            input_schema=FetchInput,
        )
    
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> dict[str, _t.Any]:
        """Fetch content from a URL using requests."""
        url = args["url"]
        max_length = args.get("max_length", 5000)
        raw = args.get("raw", False)
        start_index = args.get("start_index", 0)
        
        try:
            # Use requests to fetch the URL
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            if raw:
                # Return raw HTML content
                content = response.text
            else:
                # Extract text content using BeautifulSoup (like load_web_page)
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                
                # Filter out very short lines (like load_web_page)
                content = '\n'.join(line for line in text.splitlines() if len(line.split()) > 3)
            
            # Apply start_index and max_length
            if start_index > 0:
                content = content[start_index:]
            
            original_length = len(content)
            if len(content) > max_length:
                content = content[:max_length]
            
            return {
                "url": url,
                "content": content,
                "length": len(content),
                "original_length": original_length,
                "truncated": original_length > max_length,
                "status_code": response.status_code
            }
            
        except requests.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return {
                "url": url,
                "error": f"Request failed: {str(e)}",
                "content": ""
            }
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "content": ""
            }


class CoreToolset(Toolset):
    """Toolset containing all core tools for code agents."""
    
    def __init__(self, sandbox: Sandbox):
        """Initialize the core toolset with a sandbox."""
        tools = [
            CodebaseSearchTool(sandbox),
            ReadFileTool(sandbox),
            RunShellTool(sandbox),
            ListDirTool(sandbox),
            GrepSearchTool(sandbox),
            EditFileTool(sandbox),  # Now includes SandboxWriteFileTool functionality
            SearchReplaceTool(sandbox),
            FileSearchTool(sandbox),
            DeleteFileTool(sandbox),
            ReapplyTool(sandbox),
            WebSearchTool(sandbox),
            FetchTool(),  # FetchTool doesn't need sandbox
        ]
        
        super().__init__(
            name="core",
            description="Core tools for file system operations, shell execution, and code analysis",
            tools=tools,
        )
        self.sandbox = sandbox
    
    async def _up(self) -> None:
        """Bring up the core toolset."""
        # Start the sandbox first
        if self.sandbox.is_down:
            await self.sandbox.up()
        
        # Bring up all tools
        await super()._up()
    
    async def _down(self) -> None:
        """Bring down the core toolset."""
        # Bring down all tools first
        await super()._down()
        
        # Then stop the sandbox if it's running
        if self.sandbox.is_up:
            await self.sandbox.down() 