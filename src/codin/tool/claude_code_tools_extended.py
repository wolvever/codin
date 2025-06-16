"""Extended Claude Code tools implementation.

This module implements the remaining Claude Code tools that require additional dependencies
or more complex functionality.
"""

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import aiohttp
from urllib.parse import urljoin

from .claude_code_tools import ClaudeCodeTool, ToolContext, ToolResult


class EditTool(ClaudeCodeTool):
    """Edit file contents with exact string replacement."""
    
    def __init__(self):
        super().__init__(
            name="Edit",
            description="Edit file contents with exact string replacement"
        )
    
    async def execute(
        self,
        context: ToolContext,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: Optional[bool] = False
    ) -> ToolResult:
        """Edit a file by replacing old_string with new_string."""
        try:
            target_path = Path(file_path)
            if not target_path.is_absolute():
                target_path = Path(context.working_dir) / target_path
            
            if not target_path.exists():
                return ToolResult(
                    success=False,
                    error=f"File does not exist: {file_path}"
                )
            
            # Read file content
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if old_string exists in content
            if old_string not in content:
                return ToolResult(
                    success=False,
                    error=f"String not found in file: {old_string[:100]}..."
                )
            
            # Validate that old_string and new_string are different
            if old_string == new_string:
                return ToolResult(
                    success=False,
                    error="old_string and new_string cannot be the same"
                )
            
            # Perform replacement
            if replace_all:
                new_content = content.replace(old_string, new_string)
                replacement_count = content.count(old_string)
            else:
                # Check if old_string appears multiple times
                occurrences = content.count(old_string)
                if occurrences > 1:
                    return ToolResult(
                        success=False,
                        error=f"String appears {occurrences} times in file. Use replace_all=true to replace all occurrences, or provide a more specific string."
                    )
                new_content = content.replace(old_string, new_string, 1)
                replacement_count = 1
            
            # Write the modified content back
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            result = {
                "file_path": str(target_path),
                "replacements_made": replacement_count,
                "old_string": old_string,
                "new_string": new_string,
                "replace_all": replace_all
            }
            
            return ToolResult(
                success=True,
                output=result
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to edit"
                        },
                        "old_string": {
                            "type": "string",
                            "description": "Exact string to replace"
                        },
                        "new_string": {
                            "type": "string",
                            "description": "String to replace it with"
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "Replace all occurrences (default: false)"
                        }
                    },
                    "required": ["file_path", "old_string", "new_string"]
                }
            }
        }


class MultiEditTool(ClaudeCodeTool):
    """Make multiple edits to a single file in one operation."""
    
    def __init__(self):
        super().__init__(
            name="MultiEdit",
            description="Make multiple edits to a single file in one operation"
        )
    
    async def execute(
        self,
        context: ToolContext,
        file_path: str,
        edits: List[Dict[str, Any]]
    ) -> ToolResult:
        """Apply multiple edits to a file."""
        try:
            target_path = Path(file_path)
            if not target_path.is_absolute():
                target_path = Path(context.working_dir) / target_path
            
            if not target_path.exists():
                return ToolResult(
                    success=False,
                    error=f"File does not exist: {file_path}"
                )
            
            # Read file content
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            edits_applied = []
            
            # Validate all edits first
            for i, edit in enumerate(edits):
                old_string = edit.get("old_string")
                new_string = edit.get("new_string")
                replace_all = edit.get("replace_all", False)
                
                if not old_string or new_string is None:
                    return ToolResult(
                        success=False,
                        error=f"Edit {i}: old_string and new_string are required"
                    )
                
                if old_string == new_string:
                    return ToolResult(
                        success=False,
                        error=f"Edit {i}: old_string and new_string cannot be the same"
                    )
                
                if old_string not in content:
                    return ToolResult(
                        success=False,
                        error=f"Edit {i}: String not found in file: {old_string[:100]}..."
                    )
            
            # Apply edits sequentially
            for i, edit in enumerate(edits):
                old_string = edit["old_string"]
                new_string = edit["new_string"]
                replace_all = edit.get("replace_all", False)
                
                if replace_all:
                    replacement_count = content.count(old_string)
                    content = content.replace(old_string, new_string)
                else:
                    # Check if string appears multiple times in current content
                    occurrences = content.count(old_string)
                    if occurrences > 1:
                        return ToolResult(
                            success=False,
                            error=f"Edit {i}: String appears {occurrences} times. Use replace_all=true or provide a more specific string."
                        )
                    content = content.replace(old_string, new_string, 1)
                    replacement_count = 1
                
                edits_applied.append({
                    "edit_index": i,
                    "old_string": old_string,
                    "new_string": new_string,
                    "replace_all": replace_all,
                    "replacements_made": replacement_count
                })
            
            # Write the modified content back
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            result = {
                "file_path": str(target_path),
                "edits_applied": edits_applied,
                "total_edits": len(edits_applied)
            }
            
            return ToolResult(
                success=True,
                output=result
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to edit"
                        },
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_string": {
                                        "type": "string",
                                        "description": "String to replace"
                                    },
                                    "new_string": {
                                        "type": "string",
                                        "description": "String to replace it with"
                                    },
                                    "replace_all": {
                                        "type": "boolean",
                                        "description": "Replace all occurrences"
                                    }
                                },
                                "required": ["old_string", "new_string"]
                            },
                            "description": "Array of edit operations to perform"
                        }
                    },
                    "required": ["file_path", "edits"]
                }
            }
        }


class WriteTool(ClaudeCodeTool):
    """Write content to a file."""
    
    def __init__(self):
        super().__init__(
            name="Write",
            description="Write content to a file"
        )
    
    async def execute(
        self,
        context: ToolContext,
        file_path: str,
        content: str
    ) -> ToolResult:
        """Write content to a file."""
        try:
            target_path = Path(file_path)
            if not target_path.is_absolute():
                target_path = Path(context.working_dir) / target_path
            
            # Create parent directories if they don't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content to file
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            result = {
                "file_path": str(target_path),
                "bytes_written": len(content.encode('utf-8')),
                "lines_written": len(content.splitlines())
            }
            
            return ToolResult(
                success=True,
                output=result
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        }


class NotebookReadTool(ClaudeCodeTool):
    """Read Jupyter notebook files."""
    
    def __init__(self):
        super().__init__(
            name="NotebookRead",
            description="Read Jupyter notebook files"
        )
    
    async def execute(
        self,
        context: ToolContext,
        notebook_path: str
    ) -> ToolResult:
        """Read a Jupyter notebook and return its cells."""
        try:
            target_path = Path(notebook_path)
            if not target_path.is_absolute():
                target_path = Path(context.working_dir) / target_path
            
            if not target_path.exists():
                return ToolResult(
                    success=False,
                    error=f"Notebook does not exist: {notebook_path}"
                )
            
            # Read and parse notebook
            with open(target_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            # Extract cells with their outputs
            cells = []
            for i, cell in enumerate(nb.get('cells', [])):
                cell_info = {
                    "cell_number": i,
                    "cell_type": cell.get('cell_type', 'unknown'),
                    "source": ''.join(cell.get('source', [])),
                    "outputs": []
                }
                
                # Extract outputs for code cells
                if cell.get('cell_type') == 'code':
                    for output in cell.get('outputs', []):
                        output_info = {
                            "output_type": output.get('output_type'),
                            "text": ''.join(output.get('text', [])) if 'text' in output else None,
                            "data": output.get('data', {}),
                            "execution_count": output.get('execution_count')
                        }
                        cell_info["outputs"].append(output_info)
                
                cells.append(cell_info)
            
            result = {
                "notebook_path": str(target_path),
                "kernel_spec": nb.get('metadata', {}).get('kernelspec', {}),
                "cells": cells,
                "total_cells": len(cells)
            }
            
            return ToolResult(
                success=True,
                output=result
            )
            
        except json.JSONDecodeError:
            return ToolResult(
                success=False,
                error=f"Invalid notebook format: {notebook_path}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notebook_path": {
                            "type": "string",
                            "description": "Path to the Jupyter notebook file"
                        }
                    },
                    "required": ["notebook_path"]
                }
            }
        }


class NotebookEditTool(ClaudeCodeTool):
    """Edit Jupyter notebook cells."""
    
    def __init__(self):
        super().__init__(
            name="NotebookEdit",
            description="Edit Jupyter notebook cells"
        )
    
    async def execute(
        self,
        context: ToolContext,
        notebook_path: str,
        cell_number: int,
        new_source: str,
        cell_type: Optional[str] = None,
        edit_mode: Optional[str] = "replace"
    ) -> ToolResult:
        """Edit a specific cell in a Jupyter notebook."""
        try:
            target_path = Path(notebook_path)
            if not target_path.is_absolute():
                target_path = Path(context.working_dir) / target_path
            
            if not target_path.exists():
                return ToolResult(
                    success=False,
                    error=f"Notebook does not exist: {notebook_path}"
                )
            
            # Read notebook
            with open(target_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            cells = nb.get('cells', [])
            
            if edit_mode == "delete":
                if cell_number >= len(cells):
                    return ToolResult(
                        success=False,
                        error=f"Cell {cell_number} does not exist"
                    )
                del cells[cell_number]
                operation = f"Deleted cell {cell_number}"
                
            elif edit_mode == "insert":
                if cell_type is None:
                    return ToolResult(
                        success=False,
                        error="cell_type is required for insert mode"
                    )
                
                new_cell = {
                    "cell_type": cell_type,
                    "source": new_source.splitlines(True),
                    "metadata": {}
                }
                
                if cell_type == "code":
                    new_cell["execution_count"] = None
                    new_cell["outputs"] = []
                
                cells.insert(cell_number, new_cell)
                operation = f"Inserted new {cell_type} cell at position {cell_number}"
                
            else:  # replace mode
                if cell_number >= len(cells):
                    return ToolResult(
                        success=False,
                        error=f"Cell {cell_number} does not exist"
                    )
                
                cell = cells[cell_number]
                if cell_type:
                    cell["cell_type"] = cell_type
                    if cell_type == "code" and "execution_count" not in cell:
                        cell["execution_count"] = None
                        cell["outputs"] = []
                
                cell["source"] = new_source.splitlines(True)
                operation = f"Replaced content of cell {cell_number}"
            
            # Write notebook back
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2)
            
            result = {
                "notebook_path": str(target_path),
                "operation": operation,
                "cell_number": cell_number,
                "total_cells": len(cells)
            }
            
            return ToolResult(
                success=True,
                output=result
            )
            
        except json.JSONDecodeError:
            return ToolResult(
                success=False,
                error=f"Invalid notebook format: {notebook_path}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notebook_path": {
                            "type": "string",
                            "description": "Path to the Jupyter notebook file"
                        },
                        "cell_number": {
                            "type": "integer",
                            "description": "Index of the cell to edit (0-based)"
                        },
                        "new_source": {
                            "type": "string",
                            "description": "New source content for the cell"
                        },
                        "cell_type": {
                            "type": "string",
                            "enum": ["code", "markdown"],
                            "description": "Type of the cell (required for insert mode)"
                        },
                        "edit_mode": {
                            "type": "string",
                            "enum": ["replace", "insert", "delete"],
                            "description": "Edit operation to perform (default: replace)"
                        }
                    },
                    "required": ["notebook_path", "cell_number", "new_source"]
                }
            }
        }


class WebFetchTool(ClaudeCodeTool):
    """Fetch and analyze web content."""
    
    def __init__(self):
        super().__init__(
            name="WebFetch",
            description="Fetch and analyze web content"
        )
    
    async def execute(
        self,
        context: ToolContext,
        url: str,
        prompt: str
    ) -> ToolResult:
        """Fetch web content and process it with AI."""
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=context.timeout) as response:
                    if response.status != 200:
                        return ToolResult(
                            success=False,
                            error=f"HTTP {response.status}: {response.reason}"
                        )
                    
                    content_type = response.headers.get('content-type', '').lower()
                    
                    if 'text/html' in content_type:
                        html_content = await response.text()
                        # Convert HTML to markdown (simplified)
                        # In a real implementation, you'd use a proper HTML to markdown converter
                        import re
                        # Remove scripts and styles
                        html_content = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL)
                        html_content = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL)
                        # Extract text content (very basic)
                        text_content = re.sub(r'<[^>]+>', '', html_content)
                        text_content = re.sub(r'\s+', ' ', text_content).strip()
                    else:
                        text_content = await response.text()
            
            # Simulate AI processing of the content with the prompt
            # In a real implementation, this would use an actual LLM
            analysis = f"Processed content from {url} with prompt: {prompt}\n\nContent preview: {text_content[:500]}..."
            
            result = {
                "url": url,
                "prompt": prompt,
                "content_length": len(text_content),
                "content_type": content_type,
                "analysis": analysis,
                "content_preview": text_content[:1000]
            }
            
            return ToolResult(
                success=True,
                output=result
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Request timed out after {context.timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch content from"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Prompt describing what to extract or analyze from the content"
                        }
                    },
                    "required": ["url", "prompt"]
                }
            }
        }


# Todo management tools
@dataclass
class TodoItem:
    content: str
    status: str
    priority: str
    id: str


class TodoReadTool(ClaudeCodeTool):
    """Read current todo list."""
    
    def __init__(self):
        super().__init__(
            name="TodoRead",
            description="Read current todo list"
        )
    
    async def execute(self, context: ToolContext) -> ToolResult:
        """Read the current todo list."""
        try:
            # In a real implementation, this would read from persistent storage
            # For now, we'll return an empty list or read from a session store
            todos = []
            
            result = {
                "todos": todos,
                "total_count": len(todos)
            }
            
            return ToolResult(
                success=True,
                output=result
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }


class TodoWriteTool(ClaudeCodeTool):
    """Create or update todo list."""
    
    def __init__(self):
        super().__init__(
            name="TodoWrite",
            description="Create or update todo list"
        )
    
    async def execute(
        self,
        context: ToolContext,
        todos: List[Dict[str, str]]
    ) -> ToolResult:
        """Create or update the todo list."""
        try:
            # Validate todo items
            for i, todo in enumerate(todos):
                required_fields = ["content", "status", "priority", "id"]
                for field in required_fields:
                    if field not in todo:
                        return ToolResult(
                            success=False,
                            error=f"Todo item {i} missing required field: {field}"
                        )
            
            # In a real implementation, this would save to persistent storage
            result = {
                "todos_updated": len(todos),
                "todos": todos
            }
            
            return ToolResult(
                success=True,
                output=result
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "todos": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "Todo item content"
                                    },
                                    "status": {
                                        "type": "string",
                                        "description": "Status (pending, in_progress, completed)"
                                    },
                                    "priority": {
                                        "type": "string",
                                        "description": "Priority (low, medium, high)"
                                    },
                                    "id": {
                                        "type": "string",
                                        "description": "Unique identifier"
                                    }
                                },
                                "required": ["content", "status", "priority", "id"]
                            },
                            "description": "Array of todo items"
                        }
                    },
                    "required": ["todos"]
                }
            }
        }


class WebSearchTool(ClaudeCodeTool):
    """Search the web."""
    
    def __init__(self):
        super().__init__(
            name="WebSearch",
            description="Search the web"
        )
    
    async def execute(
        self,
        context: ToolContext,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None
    ) -> ToolResult:
        """Search the web with the given query."""
        try:
            # In a real implementation, this would use a search API
            # For now, we'll simulate search results
            results = [
                {
                    "title": f"Search result 1 for: {query}",
                    "url": "https://example.com/result1",
                    "snippet": f"This is a sample search result for the query '{query}'",
                    "domain": "example.com"
                },
                {
                    "title": f"Search result 2 for: {query}",
                    "url": "https://docs.example.com/result2", 
                    "snippet": f"Another relevant result for '{query}' with more information",
                    "domain": "docs.example.com"
                }
            ]
            
            # Apply domain filtering
            if allowed_domains:
                results = [r for r in results if any(domain in r["domain"] for domain in allowed_domains)]
            
            if blocked_domains:
                results = [r for r in results if not any(domain in r["domain"] for domain in blocked_domains)]
            
            result = {
                "query": query,
                "results": results,
                "total_results": len(results),
                "allowed_domains": allowed_domains,
                "blocked_domains": blocked_domains
            }
            
            return ToolResult(
                success=True,
                output=result
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "allowed_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Only include results from these domains"
                        },
                        "blocked_domains": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Exclude results from these domains"
                        }
                    },
                    "required": ["query"]
                }
            }
        }


class ExitPlanModeTool(ClaudeCodeTool):
    """Exit planning mode with a plan."""
    
    def __init__(self):
        super().__init__(
            name="exit_plan_mode",
            description="Exit planning mode with a plan"
        )
    
    async def execute(
        self,
        context: ToolContext,
        plan: str
    ) -> ToolResult:
        """Exit planning mode and return the plan."""
        try:
            result = {
                "plan": plan,
                "status": "exited_plan_mode",
                "timestamp": asyncio.get_event_loop().time()
            }
            
            return ToolResult(
                success=True,
                output=result
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "string",
                            "description": "The plan to execute"
                        }
                    },
                    "required": ["plan"]
                }
            }
        }


# Extended tool registry
EXTENDED_CLAUDE_CODE_TOOLS = {
    "Edit": EditTool(),
    "MultiEdit": MultiEditTool(),
    "Write": WriteTool(),
    "NotebookRead": NotebookReadTool(),
    "NotebookEdit": NotebookEditTool(),
    "WebFetch": WebFetchTool(),
    "TodoRead": TodoReadTool(),
    "TodoWrite": TodoWriteTool(),
    "WebSearch": WebSearchTool(),
    "exit_plan_mode": ExitPlanModeTool(),
}