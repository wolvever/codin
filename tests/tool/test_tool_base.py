from typing import Any
from unittest.mock import Mock

import pytest

from codin.tool.base import Tool, ToolContext, Toolset
from codin.tool.specs.base import ToolSpec


class MockToolSpec(ToolSpec):
    def __init__(self, name: str, description: str = "Mock tool"):
        self.name = name
        self.description = description
        self.parameters = {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input parameter"}
            },
            "required": ["input"]
        }
    
    def get_name(self) -> str:
        return self.name
    
    def get_description(self) -> str:
        return self.description
    
    def get_parameters(self) -> dict[str, Any]:
        return self.parameters


class MockTool(Tool):
    def __init__(self, name: str, spec: ToolSpec = None):
        super().__init__(spec or MockToolSpec(name))
        self.name = name
        self._execute_calls = []
    
    async def execute(self, context: ToolContext, **kwargs) -> Any:
        self._execute_calls.append(kwargs)
        return f"Mock result for {self.name} with {kwargs}"


class TestTool:
    
    def test_tool_creation(self):
        """Test creating a tool with spec."""
        spec = MockToolSpec("test_tool")
        tool = MockTool("test_tool", spec)
        
        assert tool.spec == spec
        assert tool.name == "test_tool"
    
    def test_validate_input_valid(self):
        """Test validating valid input."""
        tool = MockTool("test_tool")
        
        # This should not raise an exception
        result = tool.validate_input({"input": "test_value"})
        assert result is True  # or whatever validation returns
    
    def test_validate_input_invalid_missing_required(self):
        """Test validating input missing required parameter."""
        tool = MockTool("test_tool")
        
        with pytest.raises(ValueError):
            tool.validate_input({})  # Missing required 'input' parameter
    
    def test_validate_input_invalid_type(self):
        """Test validating input with wrong type."""
        tool = MockTool("test_tool")
        
        with pytest.raises(ValueError):
            tool.validate_input({"input": 123})  # Should be string, not int
    
    def test_get_spec(self):
        """Test getting tool specification."""
        spec = MockToolSpec("test_tool", "Test description")
        tool = MockTool("test_tool", spec)
        
        retrieved_spec = tool.get_spec()
        
        assert retrieved_spec == spec
        assert retrieved_spec.get_name() == "test_tool"
        assert retrieved_spec.get_description() == "Test description"
    
    @pytest.mark.asyncio
    async def test_execute(self):
        """Test tool execution."""
        tool = MockTool("test_tool")
        context = Mock(spec=ToolContext)
        
        result = await tool.execute(context, input="test_input")
        
        assert "Mock result for test_tool" in result
        assert len(tool._execute_calls) == 1
        assert tool._execute_calls[0]["input"] == "test_input"
    
    def test_to_definition(self):
        """Test converting tool to definition format."""
        spec = MockToolSpec("test_tool", "Test tool description")
        tool = MockTool("test_tool", spec)
        
        definition = tool.to_definition()
        
        assert definition["name"] == "test_tool"
        assert definition["description"] == "Test tool description"
        assert "parameters" in definition
        assert definition["parameters"]["type"] == "object"
    
    def test_to_mcp_schema(self):
        """Test converting tool to MCP schema format."""
        tool = MockTool("test_tool")
        
        schema = tool.to_mcp_schema()
        
        assert schema["name"] == "test_tool"
        assert "description" in schema
        assert "inputSchema" in schema
        assert schema["inputSchema"]["type"] == "object"
    
    def test_to_openai_schema(self):
        """Test converting tool to OpenAI schema format."""
        tool = MockTool("test_tool")
        
        schema = tool.to_openai_schema()
        
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


class TestToolset:
    
    @pytest.fixture
    def toolset(self):
        return Toolset()
    
    @pytest.fixture
    def sample_tools(self):
        return [
            MockTool("tool1"),
            MockTool("tool2"),
            MockTool("tool3")
        ]
    
    def test_toolset_creation_empty(self, toolset):
        """Test creating empty toolset."""
        assert len(toolset._tools) == 0
    
    def test_toolset_creation_with_tools(self, sample_tools):
        """Test creating toolset with initial tools."""
        toolset = Toolset(tools=sample_tools)
        
        assert len(toolset._tools) == 3
        assert "tool1" in toolset._tools
        assert "tool2" in toolset._tools
        assert "tool3" in toolset._tools
    
    def test_add_tool(self, toolset):
        """Test adding a tool to toolset."""
        tool = MockTool("new_tool")
        
        toolset.add(tool)
        
        assert "new_tool" in toolset._tools
        assert toolset._tools["new_tool"] == tool
    
    def test_add_duplicate_tool_overwrites(self, toolset):
        """Test adding duplicate tool overwrites existing."""
        tool1 = MockTool("same_name")
        tool2 = MockTool("same_name")
        
        toolset.add(tool1)
        toolset.add(tool2)
        
        assert len(toolset._tools) == 1
        assert toolset._tools["same_name"] == tool2
    
    def test_get_existing_tool(self, toolset):
        """Test getting an existing tool."""
        tool = MockTool("test_tool")
        toolset.add(tool)
        
        retrieved = toolset.get("test_tool")
        
        assert retrieved == tool
    
    def test_get_nonexistent_tool(self, toolset):
        """Test getting a non-existent tool."""
        retrieved = toolset.get("nonexistent")
        
        assert retrieved is None
    
    def test_list_tools_empty(self, toolset):
        """Test listing tools from empty toolset."""
        tools = toolset.list_tools()
        
        assert len(tools) == 0
        assert isinstance(tools, list)
    
    def test_list_tools_with_tools(self, toolset, sample_tools):
        """Test listing tools from populated toolset."""
        for tool in sample_tools:
            toolset.add(tool)
        
        tools = toolset.list_tools()
        
        assert len(tools) == 3
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"tool1", "tool2", "tool3"}
    
    def test_to_definitions(self, toolset, sample_tools):
        """Test converting toolset to definitions format."""
        for tool in sample_tools:
            toolset.add(tool)
        
        definitions = toolset.to_definitions()
        
        assert len(definitions) == 3
        definition_names = {defn["name"] for defn in definitions}
        assert definition_names == {"tool1", "tool2", "tool3"}
        
        for definition in definitions:
            assert "description" in definition
            assert "parameters" in definition
    
    def test_to_mcp_schemas(self, toolset, sample_tools):
        """Test converting toolset to MCP schemas format."""
        for tool in sample_tools:
            toolset.add(tool)
        
        schemas = toolset.to_mcp_schemas()
        
        assert len(schemas) == 3
        schema_names = {schema["name"] for schema in schemas}
        assert schema_names == {"tool1", "tool2", "tool3"}
        
        for schema in schemas:
            assert "description" in schema
            assert "inputSchema" in schema
    
    def test_to_openai_schemas(self, toolset, sample_tools):
        """Test converting toolset to OpenAI schemas format."""
        for tool in sample_tools:
            toolset.add(tool)
        
        schemas = toolset.to_openai_schemas()
        
        assert len(schemas) == 3
        
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]
    
    def test_toolset_iteration(self, toolset, sample_tools):
        """Test iterating over toolset."""
        for tool in sample_tools:
            toolset.add(tool)
        
        # Test if toolset is iterable
        tool_names = []
        for tool in toolset:
            tool_names.append(tool.name)
        
        assert set(tool_names) == {"tool1", "tool2", "tool3"}
    
    def test_toolset_contains(self, toolset):
        """Test checking if toolset contains a tool."""
        tool = MockTool("test_tool")
        toolset.add(tool)
        
        assert "test_tool" in toolset
        assert "nonexistent_tool" not in toolset
    
    def test_toolset_len(self, toolset, sample_tools):
        """Test getting toolset length."""
        assert len(toolset) == 0
        
        for tool in sample_tools:
            toolset.add(tool)
        
        assert len(toolset) == 3


class TestToolContext:
    
    def test_tool_context_creation(self):
        """Test creating tool context."""
        context = ToolContext(
            user_id="user123",
            session_id="session456",
            runner_id="runner789"
        )
        
        assert context.user_id == "user123"
        assert context.session_id == "session456"
        assert context.runner_id == "runner789"
    
    def test_tool_context_optional_fields(self):
        """Test tool context with optional fields."""
        context = ToolContext()
        
        # Check that optional fields can be None or have defaults
        assert hasattr(context, 'user_id')
        assert hasattr(context, 'session_id')
        assert hasattr(context, 'runner_id')


class TestToolValidation:
    
    def test_tool_spec_validation(self):
        """Test tool spec validation."""
        # Valid spec
        spec = MockToolSpec("valid_tool")
        assert spec.get_name() == "valid_tool"
        assert spec.get_description() == "Mock tool"
        assert "properties" in spec.get_parameters()
    
    def test_tool_parameter_validation_complex(self):
        """Test complex parameter validation."""
        class ComplexToolSpec(ToolSpec):
            def get_name(self):
                return "complex_tool"
            
            def get_description(self):
                return "Tool with complex parameters"
            
            def get_parameters(self):
                return {
                    "type": "object",
                    "properties": {
                        "required_string": {"type": "string"},
                        "optional_number": {"type": "number"},
                        "enum_choice": {"type": "string", "enum": ["option1", "option2"]},
                        "nested_object": {
                            "type": "object",
                            "properties": {
                                "nested_field": {"type": "boolean"}
                            }
                        }
                    },
                    "required": ["required_string"]
                }
        
        tool = MockTool("complex_tool", ComplexToolSpec())
        
        # Valid input
        valid_input = {
            "required_string": "test",
            "optional_number": 42.5,
            "enum_choice": "option1",
            "nested_object": {"nested_field": True}
        }
        assert tool.validate_input(valid_input) is True
        
        # Invalid enum choice
        with pytest.raises(ValueError):
            tool.validate_input({
                "required_string": "test",
                "enum_choice": "invalid_option"
            })


class TestToolIntegration:
    
    @pytest.mark.asyncio
    async def test_tool_execution_with_context(self):
        """Test tool execution with proper context."""
        tool = MockTool("integration_tool")
        context = ToolContext(
            user_id="test_user",
            session_id="test_session",
            runner_id="test_runner"
        )
        
        result = await tool.execute(context, input="test_data")
        
        assert "Mock result for integration_tool" in result
        assert "test_data" in str(result)
    
    def test_toolset_bulk_operations(self):
        """Test bulk operations on toolset."""
        tools = [MockTool(f"tool_{i}") for i in range(10)]
        toolset = Toolset()
        
        # Bulk add
        for tool in tools:
            toolset.add(tool)
        
        assert len(toolset) == 10
        
        # Bulk conversion
        definitions = toolset.to_definitions()
        mcp_schemas = toolset.to_mcp_schemas()
        openai_schemas = toolset.to_openai_schemas()
        
        assert len(definitions) == 10
        assert len(mcp_schemas) == 10
        assert len(openai_schemas) == 10
        
        # Verify consistency
        def_names = {d["name"] for d in definitions}
        mcp_names = {s["name"] for s in mcp_schemas}
        openai_names = {s["function"]["name"] for s in openai_schemas}
        
        assert def_names == mcp_names == openai_names