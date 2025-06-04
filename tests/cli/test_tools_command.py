"""Tests for the tools command functionality."""

import pytest
from unittest.mock import MagicMock, AsyncMock
import pydantic as _pyd

from codin.cli.commands import format_tool_signature, _extract_python_type
from codin.tool.base import Tool, ToolContext


class MockInputSchema(_pyd.BaseModel):
    """Mock input schema for testing."""
    name: str = _pyd.Field(..., description="The name parameter")
    age: int = _pyd.Field(None, description="The age parameter")
    active: bool = _pyd.Field(True, description="Whether active")
    tags: list[str] = _pyd.Field([], description="List of tags")
    metadata: dict[str, str] = _pyd.Field({}, description="Metadata dictionary")


class MockTool(Tool):
    """Mock tool for testing."""
    
    def __init__(self, name: str, description: str, input_schema=None):
        super().__init__(
            name=name,
            description=description,
            input_schema=input_schema or MockInputSchema
        )
    
    async def run(self, args: dict, tool_context: ToolContext):
        return {"result": "mock"}


class TestFormatToolSignature:
    """Test the format_tool_signature function."""
    
    def test_tool_with_parameters(self):
        """Test formatting a tool with parameters."""
        tool = MockTool("test_tool", "A test tool")
        signature = format_tool_signature(tool)
        
        # Should include all parameters with correct types
        assert "test_tool(" in signature
        assert "name: str" in signature
        assert "age: int = None" in signature
        assert "active: bool = None" in signature
        assert "tags: list[str] = None" in signature
        assert "metadata: dict[str, str] = None" in signature
    
    def test_tool_without_parameters(self):
        """Test formatting a tool without parameters."""
        class EmptySchema(_pyd.BaseModel):
            pass
        
        tool = MockTool("empty_tool", "Tool with no params", EmptySchema)
        signature = format_tool_signature(tool)
        
        assert signature == "empty_tool()"
    
    def test_tool_with_required_parameters(self):
        """Test formatting a tool with required parameters."""
        class RequiredSchema(_pyd.BaseModel):
            required_param: str = _pyd.Field(..., description="Required parameter")
            optional_param: str = _pyd.Field(None, description="Optional parameter")
        
        tool = MockTool("required_tool", "Tool with required params", RequiredSchema)
        signature = format_tool_signature(tool)
        
        assert "required_param: str" in signature
        assert "optional_param: str = None" in signature


class TestExtractPythonType:
    """Test the _extract_python_type function."""
    
    def test_basic_types(self):
        """Test extraction of basic JSON Schema types."""
        assert _extract_python_type({"type": "string"}) == "str"
        assert _extract_python_type({"type": "integer"}) == "int"
        assert _extract_python_type({"type": "number"}) == "float"
        assert _extract_python_type({"type": "boolean"}) == "bool"
        assert _extract_python_type({"type": "array"}) == "list"
        assert _extract_python_type({"type": "object"}) == "dict"
    
    def test_array_with_items(self):
        """Test extraction of array types with item specifications."""
        schema = {
            "type": "array",
            "items": {"type": "string"}
        }
        assert _extract_python_type(schema) == "list[str]"
    
    def test_object_with_additional_properties(self):
        """Test extraction of object types with additional properties."""
        schema = {
            "type": "object",
            "additionalProperties": True
        }
        assert _extract_python_type(schema) == "dict[str, Any]"
        
        schema = {
            "type": "object",
            "additionalProperties": {"type": "string"}
        }
        assert _extract_python_type(schema) == "dict[str, str]"
    
    def test_anyof_union_types(self):
        """Test extraction of anyOf union types."""
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "null"}
            ]
        }
        assert _extract_python_type(schema) == "str"
        
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        }
        assert _extract_python_type(schema) == "Union[str, int]"
    
    def test_fallback_for_unknown_type(self):
        """Test fallback for unknown types."""
        assert _extract_python_type({"type": "unknown"}) == "unknown"
        assert _extract_python_type({}) == "Any"


if __name__ == "__main__":
    pytest.main([__file__]) 