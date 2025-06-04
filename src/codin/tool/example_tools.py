"""Example tools demonstrating the @tool decorator usage with exact argument specifications."""

from __future__ import annotations

import typing as _t
import os
import pathlib
import subprocess
import asyncio
import logging
import tempfile
from urllib.parse import urlparse

from .decorators import tool
from .base import ToolContext

__all__ = [
    "codebase_search_tool",
    "read_file_tool", 
    "run_terminal_cmd_tool",
    "list_dir_tool",
    "grep_search_tool",
    "edit_file_tool",
    "search_replace_tool",
    "file_search_tool",
    "delete_file_tool",
    "reapply_tool",
    "web_search_tool",
    "mcp_fetch_fetch_tool",
]

logger = logging.getLogger(__name__)


@tool(
    name="codebase_search",
    description="Find snippets of code from the codebase most relevant to the search query."
)
async def codebase_search_tool(
    query: str,
    explanation: str,
    target_directories: _t.Optional[list[str]] = None,
    tool_context: ToolContext = None,
) -> dict[str, _t.Any]:
    """Semantic search over the codebase."""
    # Mock implementation - in real use would connect to semantic search service
    return {
        "query": query,
        "explanation": explanation,
        "target_directories": target_directories or [],
        "results": [
            {
                "file": "example.py",
                "line": "42",
                "content": f"# Example code matching: {query}",
                "relevance": 0.9
            }
        ]
    }


@tool(
    name="read_file", 
    description="Read the contents of a file with line range support."
)
async def read_file_tool(
    target_file: str,
    should_read_entire_file: bool,
    start_line_one_indexed: int,
    end_line_one_indexed_inclusive: int,
    explanation: str,
    tool_context: ToolContext = None,
) -> str:
    """Read file contents with line range support."""
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if should_read_entire_file:
            return content
        
        lines = content.splitlines()
        total_lines = len(lines)
        
        start_line = max(1, start_line_one_indexed)
        end_line = min(total_lines, end_line_one_indexed_inclusive)
        
        start_idx = start_line - 1
        end_idx = end_line
        selected_lines = lines[start_idx:end_idx]
        
        result_parts = []
        if start_line > 1:
            result_parts.append(f"Lines 1-{start_line-1} not shown")
        
        result_parts.append(f"Contents of {target_file}, lines {start_line}-{end_line}:")
        result_parts.append("\n".join(selected_lines))
        
        if end_line < total_lines:
            result_parts.append(f"Lines {end_line+1}-{total_lines} not shown")
        
        return "\n".join(result_parts)
        
    except Exception as e:
        return f"Error reading file {target_file}: {str(e)}"


@tool(
    name="run_terminal_cmd",
    description="Execute a terminal command."
)
async def run_terminal_cmd_tool(
    command: str,
    is_background: bool,
    explanation: str,
    tool_context: ToolContext = None,
) -> dict[str, _t.Any]:
    """Execute a terminal command."""
    try:
        if is_background:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            return {
                "command": command,
                "background": True,
                "pid": process.pid,
                "started": True
            }
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            
            return {
                "command": command,
                "background": False,
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "exit_code": process.returncode
            }
    except Exception as e:
        return {
            "command": command,
            "error": str(e),
            "exit_code": -1
        }


@tool(
    name="list_dir",
    description="List the contents of a directory."
)
async def list_dir_tool(
    relative_workspace_path: str,
    explanation: str,
    tool_context: ToolContext = None,
) -> dict[str, _t.Any]:
    """List directory contents."""
    try:
        path = pathlib.Path(relative_workspace_path)
        if not path.exists():
            return {"path": relative_workspace_path, "error": "Path does not exist"}
        
        if not path.is_dir():
            return {"path": relative_workspace_path, "error": "Path is not a directory"}
        
        contents = []
        for item in path.iterdir():
            if item.is_dir():
                contents.append(f"[dir]  {item.name}/")
            else:
                size = item.stat().st_size
                contents.append(f"[file] {item.name} ({size} bytes)")
        
        return {"path": relative_workspace_path, "contents": sorted(contents)}
        
    except Exception as e:
        return {"path": relative_workspace_path, "error": str(e)}


@tool(
    name="grep_search",
    description="Perform regex search using ripgrep."
)
async def grep_search_tool(
    query: str,
    explanation: str,
    case_sensitive: _t.Optional[bool] = None,
    include_pattern: _t.Optional[str] = None,
    exclude_pattern: _t.Optional[str] = None,
    tool_context: ToolContext = None,
) -> dict[str, _t.Any]:
    """Perform regex search."""
    try:
        cmd = ["rg", query]
        
        if case_sensitive is False:
            cmd.append("-i")
        
        if include_pattern:
            cmd.extend(["-g", include_pattern])
        
        if exclude_pattern:
            cmd.extend(["-g", f"!{exclude_pattern}"])
        
        cmd.extend(["-m", "50"])
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            results = stdout.decode('utf-8', errors='replace').strip().split('\n')
            return {"query": query, "results": results, "count": len(results)}
        else:
            return {"query": query, "error": stderr.decode('utf-8'), "results": []}
            
    except Exception as e:
        return {"query": query, "error": str(e), "results": []}


@tool(
    name="edit_file",
    description="Edit or create a file."
)
async def edit_file_tool(
    target_file: str,
    instructions: str,
    code_edit: str,
    tool_context: ToolContext = None,
) -> dict[str, _t.Any]:
    """Edit or create a file."""
    try:
        # For this example, we'll write the code_edit content directly
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(code_edit)
        
        return {
            "target_file": target_file,
            "success": True,
            "message": f"File {target_file} edited successfully"
        }
        
    except Exception as e:
        return {
            "target_file": target_file,
            "success": False,
            "error": str(e)
        }


@tool(
    name="search_replace",
    description="Perform search and replace in a file."
)
async def search_replace_tool(
    file_path: str,
    old_string: str,
    new_string: str,
    tool_context: ToolContext = None,
) -> dict[str, _t.Any]:
    """Perform search and replace in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_string not in content:
            return {
                "file_path": file_path,
                "success": False,
                "error": "Old string not found in file"
            }
        
        # Replace only the first occurrence
        new_content = content.replace(old_string, new_string, 1)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return {
            "file_path": file_path,
            "success": True,
            "message": "Search and replace completed successfully"
        }
        
    except Exception as e:
        return {
            "file_path": file_path,
            "success": False,
            "error": str(e)
        }


@tool(
    name="file_search",
    description="Search for files by fuzzy matching."
)
async def file_search_tool(
    query: str,
    explanation: str,
    tool_context: ToolContext = None,
) -> dict[str, _t.Any]:
    """Search for files by fuzzy matching."""
    try:
        # Use find command for basic file search
        process = await asyncio.create_subprocess_exec(
            "find", ".", "-name", f"*{query}*", "-type", "f",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            files = stdout.decode('utf-8', errors='replace').strip().split('\n')
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
                "error": stderr.decode('utf-8'),
                "files": []
            }
            
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "files": []
        }


@tool(
    name="delete_file",
    description="Delete a file."
)
async def delete_file_tool(
    target_file: str,
    explanation: str,
    tool_context: ToolContext = None,
) -> dict[str, _t.Any]:
    """Delete a file."""
    try:
        path = pathlib.Path(target_file)
        if path.exists():
            path.unlink()
            return {
                "target_file": target_file,
                "success": True,
                "message": f"File {target_file} deleted successfully"
            }
        else:
            return {
                "target_file": target_file,
                "success": False,
                "error": "File does not exist"
            }
            
    except Exception as e:
        return {
            "target_file": target_file,
            "success": False,
            "error": str(e)
        }


@tool(
    name="reapply",
    description="Reapply the last edit to a file."
)
async def reapply_tool(
    target_file: str,
    tool_context: ToolContext = None,
) -> dict[str, _t.Any]:
    """Reapply the last edit to a file."""
    # This is a placeholder implementation
    return {
        "target_file": target_file,
        "success": False,
        "message": "Reapply functionality not yet implemented - no previous edit to reapply"
    }


@tool(
    name="web_search",
    description="Search the web for information."
)
async def web_search_tool(
    search_term: str,
    explanation: str,
    tool_context: ToolContext = None,
) -> dict[str, _t.Any]:
    """Search the web for information."""
    # This is a placeholder implementation
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


@tool(
    name="mcp_fetch_fetch",
    description="Fetch content from a URL."
)
async def mcp_fetch_fetch_tool(
    url: str,
    max_length: _t.Optional[int] = 5000,
    raw: _t.Optional[bool] = False,
    start_index: _t.Optional[int] = 0,
    tool_context: ToolContext = None,
) -> dict[str, _t.Any]:
    """Fetch content from a URL."""
    try:
        # Use curl to fetch the URL
        cmd = ["curl", "-s", "-L", url]
        if not raw:
            cmd.extend(["-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"])
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            content = stdout.decode('utf-8', errors='replace')
            
            # Apply start_index and max_length
            if start_index > 0:
                content = content[start_index:]
            
            if len(content) > max_length:
                content = content[:max_length]
            
            return {
                "url": url,
                "content": content,
                "length": len(content),
                "truncated": len(stdout.decode('utf-8', errors='replace')) > max_length
            }
        else:
            return {
                "url": url,
                "error": stderr.decode('utf-8') or "Failed to fetch URL",
                "content": ""
            }
            
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "content": ""
        } 