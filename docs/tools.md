# Codin Tools Documentation

## Sandbox Tools

Codin provides several tools for interacting with sandbox environments to execute code, manage files, and perform development tasks.

### SandboxReadFileTool

**Tool Name:** `sandbox_read_file`

**Description:** Read a file from the sandbox environment. Supports reading specific line ranges to avoid token overflow.

**Parameters:**
- `path` (required): Path to the file to read
- `start_line` (optional): Starting line number (1-indexed, inclusive). If not provided, reads from beginning
- `end_line` (optional): Ending line number (1-indexed, inclusive). If not provided, reads to end
- `max_lines` (optional): Maximum number of lines to return (default: 1000, to prevent token overflow)

**Examples:**

```python
# Read entire file
await tool.run({"path": "example.py"})

# Read specific line range
await tool.run({
    "path": "large_file.py", 
    "start_line": 10, 
    "end_line": 20
})

# Read from line 50 to end
await tool.run({
    "path": "code.py",
    "start_line": 50
})

# Read first 5 lines
await tool.run({
    "path": "config.txt",
    "end_line": 5
})

# Read with custom max_lines limit
await tool.run({
    "path": "log.txt",
    "max_lines": 100
})
```

**Output Format:**
- For small files without line range: Returns content as-is
- For line ranges or large files: Returns content with metadata headers:
  ```
  # File: example.py
  # Lines 10-20 of 100 total
  # Output truncated to 1000 lines (if applicable)
  
  [actual file content]
  ```

### SandboxWriteFileTool

**Tool Name:** `sandbox_write_file`

**Description:** Write content to a file in the sandbox environment.

**Parameters:**
- `path` (required): Path to write the file to
- `content` (required): Content to write to the file

### SandboxExecTool

**Tool Name:** `sandbox_exec`

**Description:** Execute a command in the sandbox environment.

**Parameters:**
- `cmd` (required): The command to execute
- `cwd` (optional): Working directory for the command
- `timeout` (optional): Command timeout in seconds
- `env` (optional): Environment variables

### SandboxListFilesTool

**Tool Name:** `sandbox_list_files`

**Description:** List files in a directory in the sandbox environment (non-recursive by default to avoid large outputs).

**Parameters:**
- `path` (optional): Directory path to list files from (default: ".")
- `recursive` (optional): Whether to list files recursively in subdirectories (default: false)
- `max_files` (optional): Maximum number of files to return (default: 100)

## Key Features

### Line Range Reading
The enhanced `sandbox_read_file` tool supports reading specific line ranges, which is particularly useful for:

- **Large files**: Avoid token overflow by reading only relevant sections
- **Code review**: Focus on specific functions or sections
- **Debugging**: Examine error locations without loading entire files
- **Performance**: Reduce processing time for large codebases

### Automatic Truncation
- Files larger than `max_lines` (default: 1000) are automatically truncated
- Metadata headers indicate when truncation occurs
- Helps prevent token limits in LLM interactions

### Backward Compatibility
- Existing calls without line range parameters work unchanged
- Small files (≤ max_lines) return content without metadata headers
- Tool output remains a string for compatibility with existing code

### Error Handling
- Invalid line ranges (start > end) return empty content with metadata
- Line numbers beyond file length are handled gracefully
- Negative line numbers are clamped to 1
- Empty files are handled correctly 