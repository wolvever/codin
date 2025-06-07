"""Tests for core tools implementation."""

import pytest
import tempfile
import pathlib
import os
from unittest.mock import AsyncMock, MagicMock

from codin.tool.core_tools import FetchTool
from codin.tool.sandbox import SandboxToolset
from codin.tool.base import ToolContext
from codin.sandbox.local import LocalSandbox


@pytest.fixture
async def sandbox():
    """Create and initialize a LocalSandbox for testing."""
    sandbox = LocalSandbox()
    await sandbox.up()
    yield sandbox
    await sandbox.down()


@pytest.fixture
def tool_context():
    return ToolContext(tool_name="test_tool", arguments={}, session_id="test_session")


@pytest.fixture
def sandbox_toolset(sandbox):
    """Create SandboxToolset for tests."""
    return SandboxToolset(sandbox)


@pytest.mark.asyncio
async def test_codebase_search_tool(sandbox_toolset, sandbox, tool_context):
    """Test CodebaseSearchTool functionality."""
    # Create a test file with searchable content
    await sandbox.write_file("test_search.py", "def test_function():\n    return 'test'")

    tool = sandbox_toolset.get_tool("codebase_search")

    args = {"query": "test_function", "explanation": "Testing codebase search", "target_directories": None}

    result = await tool.run(args, tool_context)

    assert "query" in result
    assert "results" in result
    assert result["query"] == "test_function"


@pytest.mark.asyncio
async def test_read_file_tool(sandbox_toolset, sandbox, tool_context):
    """Test ReadFileTool functionality."""
    # Create a test file
    test_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    await sandbox.write_file("test.txt", test_content)

    tool = sandbox_toolset.get_tool("read_file")

    # Test reading entire file
    args = {
        "target_file": "test.txt",
        "should_read_entire_file": True,
        "start_line_one_indexed": 1,
        "end_line_one_indexed_inclusive": 5,
        "explanation": "Testing file reading",
    }

    result = await tool.run(args, tool_context)

    assert isinstance(result, str)
    assert "Line 1" in result
    assert "Line 5" in result


@pytest.mark.asyncio
async def test_read_file_tool_line_range(sandbox_toolset, sandbox, tool_context):
    """Test ReadFileTool with line range."""
    # Create a test file
    test_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    await sandbox.write_file("test.txt", test_content)

    tool = sandbox_toolset.get_tool("read_file")

    # Test reading specific line range
    args = {
        "target_file": "test.txt",
        "should_read_entire_file": False,
        "start_line_one_indexed": 2,
        "end_line_one_indexed_inclusive": 4,
        "explanation": "Testing line range reading",
    }

    result = await tool.run(args, tool_context)

    assert isinstance(result, str)
    assert "Line 2" in result
    assert "Line 4" in result


@pytest.mark.asyncio
async def test_run_shell_tool(sandbox_toolset, sandbox, tool_context):
    """Test RunShellTool functionality."""
    tool = sandbox_toolset.get_tool("run_shell")

    # Test regular command execution
    args = {"command": "echo Hello World", "is_background": False, "explanation": "Testing command execution"}

    result = await tool.run(args, tool_context)

    assert isinstance(result, dict)
    assert "command" in result
    assert "stdout" in result
    assert "exit_code" in result
    assert result["command"] == "echo Hello World"
    assert result["background"] is False


@pytest.mark.asyncio
async def test_run_shell_tool_background(sandbox_toolset, sandbox, tool_context):
    """Test RunShellTool background execution."""
    tool = sandbox_toolset.get_tool("run_shell")

    # Test background command execution (use a simple command that works on all platforms)
    args = {"command": "echo Background Test", "is_background": True, "explanation": "Testing background execution"}

    result = await tool.run(args, tool_context)

    assert isinstance(result, dict)
    assert "command" in result
    assert "background" in result
    assert result["background"] is True


@pytest.mark.asyncio
async def test_list_dir_tool(sandbox_toolset, sandbox, tool_context):
    """Test ListDirTool functionality."""
    # Create some test files
    await sandbox.write_file("test1.txt", "content1")
    await sandbox.write_file("test2.py", "content2")

    tool = sandbox_toolset.get_tool("list_dir")

    args = {"relative_workspace_path": ".", "explanation": "Testing directory listing"}

    result = await tool.run(args, tool_context)

    assert "path" in result
    assert "contents" in result
    assert result["path"] == "."
    assert isinstance(result["contents"], list)


@pytest.mark.asyncio
async def test_grep_search_tool(sandbox_toolset, sandbox, tool_context):
    """Test GrepSearchTool functionality."""
    # Create a test file with searchable content
    await sandbox.write_file("test_grep.py", "def test_function():\n    return 'test'")

    tool = sandbox_toolset.get_tool("grep_search")

    args = {
        "query": "test_function",
        "explanation": "Testing grep search",
        "case_sensitive": False,
        "include_pattern": "*.py",
    }

    result = await tool.run(args, tool_context)

    assert "query" in result
    assert "results" in result
    # The tool should always return these fields, even if there's an error
    # On Windows, rg might not be available, so we just check the structure


@pytest.mark.asyncio
async def test_edit_file_tool(sandbox_toolset, sandbox, tool_context):
    """Test EditFileTool functionality."""
    tool = sandbox_toolset.get_tool("edit_file")

    args = {
        "target_file": "test_edit.py",
        "instructions": "Create a test file",
        "code_edit": "def test():\n    return 'hello'",
    }

    result = await tool.run(args, tool_context)

    assert "target_file" in result
    assert "success" in result
    assert result["success"] is True

    # Verify file was created
    content = await sandbox.read_file("test_edit.py")
    assert "def test():" in content


@pytest.mark.asyncio
async def test_search_replace_tool(sandbox_toolset, sandbox, tool_context):
    """Test SearchReplaceTool functionality."""
    # Create a test file
    await sandbox.write_file("test_replace.txt", "Line 1\nLine 2\nLine 3")

    tool = sandbox_toolset.get_tool("search_replace")

    args = {"file_path": "test_replace.txt", "old_string": "Line 1", "new_string": "Modified Line 1"}

    result = await tool.run(args, tool_context)

    assert "file_path" in result
    assert "success" in result

    # Verify replacement worked
    if result["success"]:
        content = await sandbox.read_file("test_replace.txt")
        assert "Modified Line 1" in content


@pytest.mark.asyncio
async def test_file_search_tool(sandbox_toolset, sandbox, tool_context):
    """Test FileSearchTool functionality."""
    # Create some test files
    await sandbox.write_file("test_search_file.py", "content")
    await sandbox.write_file("another_test.txt", "content")

    tool = sandbox_toolset.get_tool("file_search")

    args = {"query": "test", "explanation": "Testing file search"}

    result = await tool.run(args, tool_context)

    assert "query" in result
    assert "files" in result
    # The tool should always return these fields, even if there's an error
    # On Windows, find command syntax is different, so we just check the structure


@pytest.mark.asyncio
async def test_delete_file_tool(sandbox_toolset, sandbox, tool_context):
    """Test DeleteFileTool functionality."""
    # Create a test file to delete
    await sandbox.write_file("test_delete.txt", "content to delete")

    tool = sandbox_toolset.get_tool("delete_file")

    args = {"target_file": "test_delete.txt", "explanation": "Testing file deletion"}

    result = await tool.run(args, tool_context)

    assert "target_file" in result
    assert "success" in result


@pytest.mark.asyncio
async def test_reapply_tool(sandbox_toolset, sandbox, tool_context):
    """Test ReapplyTool functionality."""
    tool = sandbox_toolset.get_tool("reapply")

    args = {"target_file": "test.txt"}

    result = await tool.run(args, tool_context)

    assert "target_file" in result
    assert "success" in result
    # Should be False since it's not implemented
    assert result["success"] is False


@pytest.mark.asyncio
async def test_web_search_tool(sandbox_toolset, sandbox, tool_context):
    """Test WebSearchTool functionality."""
    tool = sandbox_toolset.get_tool("web_search")

    args = {"search_term": "python programming", "explanation": "Testing web search"}

    result = await tool.run(args, tool_context)

    assert "search_term" in result
    assert "results" in result


@pytest.mark.asyncio
async def test_fetch_tool(tool_context):
    """Test FetchTool functionality."""
    tool = FetchTool()  # No longer takes sandbox parameter

    args = {
        "url": "https://httpbin.org/html",  # Use a reliable test URL
        "max_length": 1000,
        "raw": False,
        "start_index": 0,
    }

    result = await tool.run(args, tool_context)

    assert "url" in result
    assert "content" in result
    # Should have status_code for successful requests
    if "error" not in result:
        assert "status_code" in result


@pytest.mark.asyncio
async def test_sandbox_toolset(sandbox):
    """Test SandboxToolset initialization and functionality."""
    toolset = SandboxToolset(sandbox)

    # Check that all expected tools are present
    assert len(toolset.tools) == 12  # All 12 sandbox tools

    tool_names = [tool.name for tool in toolset.tools]
    expected_tools = [
        "codebase_search",
        "read_file",
        "run_shell",
        "list_dir",
        "grep_search",
        "edit_file",
        "search_replace",
        "file_search",
        "delete_file",
        "reapply",
        "web_search",
        "fetch",
    ]

    for expected_tool in expected_tools:
        assert expected_tool in tool_names

    # Test toolset lifecycle
    await toolset.up()

    # Test getting tools
    read_file_tool = toolset.get_tool("read_file")
    assert read_file_tool is not None
    assert read_file_tool.name == "read_file"


@pytest.mark.asyncio
async def test_tool_lifecycle(sandbox_toolset, sandbox, tool_context):
    """Test tool lifecycle."""
    tool = sandbox_toolset.get_tool("list_dir")

    # Initially down
    from codin.lifecycle import LifecycleState

    assert tool.state == LifecycleState.DOWN
    assert not tool.is_up

    # Bring up
    await tool.up()
    assert tool.state == LifecycleState.UP
    assert tool.is_up

    # Bring down
    await tool.down()
    assert tool.state == LifecycleState.DOWN
    assert not tool.is_up


@pytest.mark.asyncio
async def test_tool_error_handling(sandbox_toolset, sandbox, tool_context):
    """Test tool error handling."""
    tool = sandbox_toolset.get_tool("read_file")

    # Test with non-existent file
    args = {
        "target_file": "nonexistent.txt",
        "should_read_entire_file": True,
        "start_line_one_indexed": 1,
        "end_line_one_indexed_inclusive": 5,
        "explanation": "Testing error handling",
    }

    result = await tool.run(args, tool_context)

    assert isinstance(result, str)
    assert "Error reading file" in result
