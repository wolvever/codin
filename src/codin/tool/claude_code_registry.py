"""Unified Claude Code tools registry.

This module provides a complete registry of all Claude Code compatible tools
with a clean interface for tool discovery and execution.
"""

from typing import Any

from .claude_code_tools import CLAUDE_CODE_TOOLS, ClaudeCodeTool, ToolContext, ToolResult
from .claude_code_tools_extended import EXTENDED_CLAUDE_CODE_TOOLS


class ClaudeCodeRegistry:
    """Registry for all Claude Code compatible tools."""
    
    def __init__(self):
        self._tools: dict[str, ClaudeCodeTool] = {}
        self._load_default_tools()
    
    def _load_default_tools(self):
        """Load all default Claude Code tools."""
        self._tools.update(CLAUDE_CODE_TOOLS)
        self._tools.update(EXTENDED_CLAUDE_CODE_TOOLS)
    
    def get_tool(self, name: str) -> ClaudeCodeTool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[str]:
        """Get list of all available tool names."""
        return list(self._tools.keys())
    
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI function calling schemas for all tools."""
        return [tool.get_schema() for tool in self._tools.values()]
    
    def get_tool_info(self, name: str) -> dict[str, Any] | None:
        """Get detailed information about a tool."""
        tool = self.get_tool(name)
        if not tool:
            return None
        
        return {
            "name": tool.name,
            "description": tool.description,
            "schema": tool.get_schema()
        }
    
    def register_tool(self, tool: ClaudeCodeTool) -> None:
        """Register a custom tool."""
        self._tools[tool.name] = tool
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    async def execute_tool(
        self, 
        name: str, 
        context: ToolContext, 
        **kwargs
    ) -> ToolResult:
        """Execute a tool with the given parameters."""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found"
            )
        
        try:
            return await tool.execute(context, **kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )


# Global registry instance
claude_code_registry = ClaudeCodeRegistry()


# Convenience functions
def get_tool(name: str) -> ClaudeCodeTool | None:
    """Get a Claude Code tool by name."""
    return claude_code_registry.get_tool(name)


def list_tools() -> list[str]:
    """Get list of all available Claude Code tools."""
    return claude_code_registry.list_tools()


def get_tool_schemas() -> list[dict[str, Any]]:
    """Get OpenAI function calling schemas for all Claude Code tools."""
    return claude_code_registry.get_tool_schemas()


async def execute_tool(
    name: str, 
    context: ToolContext, 
    **kwargs
) -> ToolResult:
    """Execute a Claude Code tool."""
    return await claude_code_registry.execute_tool(name, context, **kwargs)


def get_tool_info(name: str) -> dict[str, Any] | None:
    """Get detailed information about a Claude Code tool."""
    return claude_code_registry.get_tool_info(name)


def register_custom_tool(tool: ClaudeCodeTool) -> None:
    """Register a custom Claude Code compatible tool."""
    claude_code_registry.register_tool(tool)