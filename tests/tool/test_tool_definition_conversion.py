from __future__ import annotations

"""Tests for Tool to ToolDefinition conversion functionality."""

import pytest
import pydantic as pyd

from src.codin.tool.base import Tool, ToolDefinition, ToolContext, Toolset
from src.codin.tool.base import to_tool_definition, to_tool_definitions


class ToolInputSchema(pyd.BaseModel):
    """Input schema for test tools."""
    input: str
    optional_param: int = 10


class SimpleTool(Tool):
    """Simple test tool for validation."""
    
    def __init__(self, name: str = "test_tool", description: str = "A test tool"):
        super().__init__(name, description, input_schema=ToolInputSchema)
    
    async def run(self, args: dict, context: ToolContext):
        """Simple run implementation."""
        return f"Executed {self.name} with: {args}"


class TestToolToToolDefinitionConversion:
    """Test Tool to ToolDefinition conversion functionality."""

    def test_tool_to_tool_definition_method(self):
        """Test Tool.to_tool_definition() method."""
        tool = SimpleTool()
        tool_def = tool.to_tool_definition()
        
        assert isinstance(tool_def, ToolDefinition)
        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert tool_def.parameters["type"] == "object"
        assert "input" in tool_def.parameters["properties"]
        assert "optional_param" in tool_def.parameters["properties"]
        assert tool_def.parameters["required"] == ["input"]
        
        # Check metadata
        assert tool_def.metadata is not None
        assert tool_def.metadata["version"] == "1.0.0"
        assert tool_def.metadata["is_generative"] is False

    def test_tool_to_tool_definition_with_custom_metadata(self):
        """Test Tool.to_tool_definition() with custom metadata."""
        tool = SimpleTool()
        tool.metadata = {"custom": "value", "priority": 5}
        tool_def = tool.to_tool_definition()
        
        assert tool_def.metadata["custom"] == "value"
        assert tool_def.metadata["priority"] == 5
        assert tool_def.metadata["version"] == "1.0.0"

    def test_to_tool_definition_helper_with_tool(self):
        """Test to_tool_definition() helper function with Tool object."""
        tool = SimpleTool("helper_tool", "Tool for helper test")
        tool_def = to_tool_definition(tool)
        
        assert isinstance(tool_def, ToolDefinition)
        assert tool_def.name == "helper_tool"
        assert tool_def.description == "Tool for helper test"

    def test_to_tool_definition_helper_with_tool_definition(self):
        """Test to_tool_definition() helper function with ToolDefinition object."""
        original_def = ToolDefinition(
            name="direct_def",
            description="Direct definition",
            parameters={"type": "object", "properties": {}},
            metadata={"source": "direct"}
        )
        
        # Should return the same object (passthrough)
        result = to_tool_definition(original_def)
        assert result is original_def

    def test_to_tool_definitions_with_tool_list(self):
        """Test to_tool_definitions() with list of Tool objects."""
        tools = [
            SimpleTool("tool1", "First tool"),
            SimpleTool("tool2", "Second tool")
        ]
        
        tool_defs = to_tool_definitions(tools)
        
        assert len(tool_defs) == 2
        assert all(isinstance(td, ToolDefinition) for td in tool_defs)
        assert tool_defs[0].name == "tool1"
        assert tool_defs[1].name == "tool2"

    def test_to_tool_definitions_with_mixed_list(self):
        """Test to_tool_definitions() with mixed Tool and ToolDefinition objects."""
        tool = SimpleTool("tool_obj", "Tool object")
        tool_def = ToolDefinition("def_obj", "Definition object", {"type": "object"})
        
        mixed_list = [tool, tool_def]
        result = to_tool_definitions(mixed_list)
        
        assert len(result) == 2
        assert all(isinstance(td, ToolDefinition) for td in result)
        assert result[0].name == "tool_obj"
        assert result[1].name == "def_obj"
        assert result[1] is tool_def  # Should be the same object

    def test_to_tool_definitions_with_empty_list(self):
        """Test to_tool_definitions() with empty list."""
        result = to_tool_definitions([])
        assert result == []

    def test_to_tool_definitions_with_none(self):
        """Test to_tool_definitions() with None."""
        result = to_tool_definitions(None)
        assert result == []


class TestToolsetToToolDefinitions:
    """Test Toolset.to_tool_definitions() functionality."""

    def test_toolset_to_tool_definitions(self):
        """Test Toolset.to_tool_definitions() method."""
        tools = [
            SimpleTool("toolset_tool1", "First toolset tool"),
            SimpleTool("toolset_tool2", "Second toolset tool")
        ]
        
        toolset = Toolset("Test Toolset", "Test toolset description", tools)
        tool_defs = toolset.to_tool_definitions()
        
        assert len(tool_defs) == 2
        assert all(isinstance(td, ToolDefinition) for td in tool_defs)
        assert tool_defs[0].name == "toolset_tool1"
        assert tool_defs[1].name == "toolset_tool2"

    def test_empty_toolset_to_tool_definitions(self):
        """Test Toolset.to_tool_definitions() with empty toolset."""
        toolset = Toolset("Empty Toolset", "Empty toolset")
        tool_defs = toolset.to_tool_definitions()
        
        assert tool_defs == []


class TestToolDefinitionStructure:
    """Test ToolDefinition structure and properties."""

    def test_tool_definition_creation(self):
        """Test ToolDefinition creation and properties."""
        params = {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input parameter"}
            },
            "required": ["input"]
        }
        
        tool_def = ToolDefinition(
            name="test_def",
            description="Test definition",
            parameters=params,
            metadata={"version": "2.0.0"}
        )
        
        assert tool_def.name == "test_def"
        assert tool_def.description == "Test definition"
        assert tool_def.parameters == params
        assert tool_def.metadata == {"version": "2.0.0"}

    def test_tool_definition_frozen(self):
        """Test that ToolDefinition is frozen (immutable)."""
        tool_def = ToolDefinition("test", "Test", {"type": "object"})
        
        with pytest.raises(AttributeError):
            tool_def.name = "changed"  # Should raise error because it's frozen 