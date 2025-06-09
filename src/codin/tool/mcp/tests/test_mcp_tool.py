"""Unit tests for MCPTool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from .. import mcp_types
from ..exceptions import MCPConnectionError, MCPInputError, MCPProtocolError, MCPToolError
from ..mcp_tool import MCPTool, _DefaultInputModel, MCP_METHOD_MAP
from ..session_manager import MCPSessionManager # Abstract, will be mocked
from ...base.lifecycle_manager import LifecycleState


@pytest.fixture
def mock_session_manager():
    mock = AsyncMock(spec=MCPSessionManager)
    mock.server_capabilities = mcp_types.ServerCapabilities(tools={"listChanged": True}) # Example capability
    mock.initialize = AsyncMock() # Standard initialize

    # Mock specific methods used by MCP_METHOD_MAP
    mock.list_resources = AsyncMock()
    mock.get_prompt = AsyncMock()
    # ... add other specific methods as needed for tests ...
    mock.call_tool = AsyncMock() # For custom tools
    return mock

@pytest.fixture
def basic_mcp_tool(mock_session_manager):
    return MCPTool(
        name="test_tool",
        description="A test tool",
        session_manager=mock_session_manager
    )

@pytest.fixture
def resources_list_tool(mock_session_manager):
    # Tool specifically named to match a standard MCP method
    return MCPTool(
        name="resources/list",
        description="Lists resources via MCP standard method",
        session_manager=mock_session_manager
    )

# --- MCPTool Tests ---

@pytest.mark.asyncio
async def test_mcp_tool_instantiation(mock_session_manager):
    annotations = mcp_types.ToolAnnotation(title="My MCP Tool")
    schema_dict = {"type": "object", "properties": {"test": {"type": "string"}}}
    tool = MCPTool(
        name="custom_tool",
        description="Custom description",
        session_manager=mock_session_manager,
        tool_annotations=annotations,
        mcp_input_schema_dict=schema_dict
    )
    assert tool.name == "custom_tool"
    assert tool.tool_annotations == annotations
    assert tool.mcp_input_schema_dict == schema_dict
    assert tool._state == LifecycleState.DISCONNECTED

@pytest.mark.asyncio
async def test_mcp_tool_initialize_success(basic_mcp_tool, mock_session_manager):
    await basic_mcp_tool.initialize()
    mock_session_manager.initialize.assert_called_once()
    assert basic_mcp_tool._state == LifecycleState.UP

@pytest.mark.asyncio
async def test_mcp_tool_initialize_failure_mcp_error(basic_mcp_tool, mock_session_manager):
    mock_session_manager.initialize = AsyncMock(side_effect=MCPConnectionError("Connection failed"))

    with pytest.raises(MCPToolError) as exc_info:
        await basic_mcp_tool.initialize()

    assert "Session initialization failed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, MCPConnectionError)
    assert basic_mcp_tool._state == LifecycleState.ERROR

@pytest.mark.asyncio
async def test_mcp_tool_initialize_failure_unexpected_error(basic_mcp_tool, mock_session_manager):
    mock_session_manager.initialize = AsyncMock(side_effect=RuntimeError("Unexpected issue"))

    with pytest.raises(MCPToolError) as exc_info:
        await basic_mcp_tool.initialize()

    assert "Unexpected error during session initialization" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert basic_mcp_tool._state == LifecycleState.ERROR


# --- MCPTool.run() Tests ---

@pytest.mark.asyncio
async def test_mcp_tool_run_standard_method_success(resources_list_tool, mock_session_manager):
    # Setup tool for "resources/list"
    await resources_list_tool.initialize() # Sets state to UP
    assert resources_list_tool._state == LifecycleState.UP

    mock_response_data = {"resources": [{"name": "res1", "uri": "file:///res1.txt"}]}
    mock_pydantic_response = mcp_types.ListResourcesResult.model_validate(mock_response_data)
    mock_session_manager.list_resources = AsyncMock(return_value=mock_pydantic_response)

    args = {"cursor": "some_cursor"} # Valid args for ListResourcesRequestParams
    result = await resources_list_tool.run(args, tool_context=MagicMock())

    mock_session_manager.list_resources.assert_called_once()
    # Check that args were validated and passed as the Pydantic model
    called_with_params = mock_session_manager.list_resources.call_args[0][0]
    assert isinstance(called_with_params, mcp_types.ListResourcesRequestParams)
    assert called_with_params.cursor == "some_cursor"

    assert result == mock_pydantic_response.model_dump(exclude_none=True)


@pytest.mark.asyncio
async def test_mcp_tool_run_standard_method_input_error(resources_list_tool, mock_session_manager):
    await resources_list_tool.initialize()

    # `ListResourcesRequestParams` allows `cursor` to be optional.
    # To test input error, let's imagine a method that requires a specific param.
    # We'll mock `get_prompt` which requires `name`.
    get_prompt_tool = MCPTool(name="prompts/get", description="test", session_manager=mock_session_manager)
    await get_prompt_tool.initialize()

    invalid_args = {"wrong_param": "value"} # Missing 'name'
    with pytest.raises(MCPInputError) as exc_info:
        await get_prompt_tool.run(invalid_args, tool_context=MagicMock())

    assert "Invalid arguments for prompts/get" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, pydantic.ValidationError)


@pytest.mark.asyncio
async def test_mcp_tool_run_custom_tool_success_text_content(basic_mcp_tool, mock_session_manager):
    basic_mcp_tool.name = "my_custom_tool" # Override name
    await basic_mcp_tool.initialize()

    mock_call_tool_response = {
        "content": [{"type": "text", "text": "Custom tool output"}],
        "isError": False
    }
    mock_session_manager.call_tool = AsyncMock(return_value=mock_call_tool_response)

    args = {"input_param": "value"}
    result = await basic_mcp_tool.run(args, tool_context=MagicMock())

    mock_session_manager.call_tool.assert_called_once_with("my_custom_tool", args)
    assert result == "Custom tool output"

@pytest.mark.asyncio
async def test_mcp_tool_run_custom_tool_multiple_content_types(basic_mcp_tool, mock_session_manager):
    basic_mcp_tool.name = "multi_content_tool"
    await basic_mcp_tool.initialize()

    mock_call_tool_response = {
        "content": [
            {"type": "text", "text": "First part."},
            {"type": "image", "mimeType": "image/png", "data": "imgdata"},
            {"type": "resource", "resource": {"uri": "file:///tmp/doc.pdf", "text": "Embedded PDF text"}}
        ]
    }
    mock_session_manager.call_tool = AsyncMock(return_value=mock_call_tool_response)
    result = await basic_mcp_tool.run({}, tool_context=MagicMock())

    expected_output = (
        "First part.\n"
        "[Image content: image/png, data: imgdata...]\n"
        "[Embedded text resource: file:///tmp/doc.pdf]"
    )
    assert result == expected_output

@pytest.mark.asyncio
async def test_mcp_tool_run_custom_tool_non_calltoolresult_response(basic_mcp_tool, mock_session_manager):
    basic_mcp_tool.name = "simple_tool"
    await basic_mcp_tool.initialize()

    simple_response = {"message": "This is a simple JSON response, not CallToolResult."}
    mock_session_manager.call_tool = AsyncMock(return_value=simple_response)
    result = await basic_mcp_tool.run({}, tool_context=MagicMock())
    assert result == simple_response


@pytest.mark.asyncio
async def test_mcp_tool_run_session_error_propagation(resources_list_tool, mock_session_manager):
    await resources_list_tool.initialize()

    mock_session_manager.list_resources = AsyncMock(side_effect=MCPProtocolError("Server messed up"))

    with pytest.raises(MCPProtocolError) as exc_info:
        await resources_list_tool.run({}, tool_context=MagicMock())
    assert "Server messed up" in str(exc_info.value)


@pytest.mark.asyncio
@patch('codin.tool.mcp.mcp_tool.MCPTool.initialize', new_callable=AsyncMock) # Patch MCPTool's own initialize
async def test_mcp_tool_run_retry_on_connection_error(
    mock_initialize_method, # This is the patched MCPTool.initialize
    resources_list_tool,
    mock_session_manager # This is the MCPSessionManager mock
):
    # Configure the tool to be initially UP, so run() doesn't call _reinitialize_session at the start
    resources_list_tool._state = LifecycleState.UP

    # First call to list_resources raises MCPConnectionError, second call succeeds
    mock_response_data = {"resources": [{"name": "res1", "uri": "file:///res1.txt"}]}
    mock_pydantic_response = mcp_types.ListResourcesResult.model_validate(mock_response_data)

    mock_session_manager.list_resources = AsyncMock(
        side_effect=[
            MCPConnectionError("Simulated connection error"),
            mock_pydantic_response # Success on retry
        ]
    )
    # The patched MCPTool.initialize should set the state to UP
    mock_initialize_method.side_effect = lambda: setattr(resources_list_tool, '_state', LifecycleState.UP)


    args = {}
    result = await resources_list_tool.run(args, tool_context=MagicMock())

    # Check that initialize was called (by _reinitialize_session)
    mock_initialize_method.assert_called_once()

    # Check that list_resources was called twice
    assert mock_session_manager.list_resources.call_count == 2
    assert result == mock_pydantic_response.model_dump(exclude_none=True)
    assert resources_list_tool._state == LifecycleState.UP


@pytest.mark.asyncio
async def test_mcp_tool_run_tool_not_up_reinitialize_fails(basic_mcp_tool, mock_session_manager):
    basic_mcp_tool._state = LifecycleState.DISCONNECTED # Start in a non-UP state

    # Mock session manager's initialize to fail, so re-initialization also fails
    mock_session_manager.initialize = AsyncMock(side_effect=MCPConnectionError("Persistent connection failure"))

    with pytest.raises(MCPToolError) as exc_info:
        await basic_mcp_tool.run({}, tool_context=MagicMock())

    assert f"MCPTool {basic_mcp_tool.name} could not be reinitialized. Current state: {LifecycleState.ERROR}" in str(exc_info.value)
    mock_session_manager.initialize.assert_called_once() # Attempted to initialize
    assert basic_mcp_tool._state == LifecycleState.ERROR


# Placeholder for more tests
# TODO:
# - Test custom tool call where CallToolResult.isError is true.
# - Test _reinitialize_session more directly if possible, or more scenarios with retry_on_closed_resource.
# - Test with actual input schema validation if MCPTool starts using self.mcp_input_schema_dict for that.

```
