"""Unit tests for MCP Pydantic type definitions."""

import pytest
from pydantic import ValidationError
import json

from .. import mcp_types

def test_request_id():
    assert isinstance(mcp_types.RequestId, type(Union[str, int]))

def test_json_rpc_request_serialization():
    req = mcp_types.JSONRPCRequest(method="test/method", id=1, params={"arg1": "val1"})
    assert req.jsonrpc == "2.0"
    assert req.method == "test/method"
    assert req.id == 1
    assert req.params == {"arg1": "val1"}

    dumped = req.model_dump()
    assert dumped == {"jsonrpc": "2.0", "method": "test/method", "id": 1, "params": {"arg1": "val1"}}

    req_no_params = mcp_types.JSONRPCRequest(method="test/method", id="req-2")
    assert req_no_params.params is None
    dumped_no_params = req_no_params.model_dump(exclude_none=True)
    assert dumped_no_params == {"jsonrpc": "2.0", "method": "test/method", "id": "req-2"}


def test_json_rpc_response_serialization():
    resp = mcp_types.JSONRPCResponse(id=1, result={"data": "success"})
    assert resp.jsonrpc == "2.0"
    assert resp.id == 1
    assert resp.result == {"data": "success"}

    dumped = resp.model_dump()
    assert dumped == {"jsonrpc": "2.0", "id": 1, "result": {"data": "success"}}

def test_json_rpc_error_serialization():
    err_data = mcp_types.JSONRPCErrorData(code=-32000, message="Server error", data="Extra info")
    err = mcp_types.JSONRPCError(id=1, error=err_data)
    assert err.jsonrpc == "2.0"
    assert err.id == 1
    assert err.error.code == -32000
    assert err.error.message == "Server error"
    assert err.error.data == "Extra info"

    dumped = err.model_dump()
    assert dumped == {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32000, "message": "Server error", "data": "Extra info"}
    }

def test_initialize_request_and_result():
    # InitializeRequest
    client_caps = mcp_types.ClientCapabilities(experimental={"foo": {"bar": True}})
    client_info = mcp_types.Implementation(name="test-client", version="0.1")
    init_params = mcp_types.InitializeRequestParams(
        capabilities=client_caps,
        clientInfo=client_info,
        protocolVersion="0.1.0"
    )
    init_req = mcp_types.InitializeRequest(id="init-1", params=init_params)

    assert init_req.method == "initialize"
    assert init_req.params.capabilities.experimental["foo"]["bar"] is True

    dumped_req = init_req.model_dump(exclude_none=True)
    expected_req_dict = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": "init-1",
        "params": {
            "capabilities": {"experimental": {"foo": {"bar": True}}},
            "clientInfo": {"name": "test-client", "version": "0.1"},
            "protocolVersion": "0.1.0"
        }
    }
    assert dumped_req == expected_req_dict

    # InitializeResult
    server_caps = mcp_types.ServerCapabilities(prompts={"listChanged": True})
    server_info = mcp_types.Implementation(name="test-server", version="1.0")
    init_result_data = {
        "capabilities": server_caps.model_dump(exclude_none=True),
        "serverInfo": server_info.model_dump(exclude_none=True),
        "protocolVersion": "0.1.0",
        "instructions": "Use wisely"
    }
    init_result = mcp_types.InitializeResult.model_validate(init_result_data)
    assert init_result.capabilities.prompts["listChanged"] is True
    assert init_result.serverInfo.name == "test-server"
    assert init_result.instructions == "Use wisely"

    dumped_result = init_result.model_dump(exclude_none=True)
    assert dumped_result == init_result_data


def test_list_tools_result():
    tool_data = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {"type": "object", "properties": {"param1": {"type": "string"}}},
        "annotations": {"title": "Test Tool Title"}
    }
    list_tools_result_data = {
        "tools": [tool_data],
        "nextCursor": "cursor123"
    }
    list_tools_res = mcp_types.ListToolsResult.model_validate(list_tools_result_data)

    assert len(list_tools_res.tools) == 1
    assert list_tools_res.tools[0].name == "test_tool"
    assert list_tools_res.tools[0].inputSchema["properties"]["param1"]["type"] == "string"
    assert list_tools_res.tools[0].annotations.title == "Test Tool Title"
    assert list_tools_res.nextCursor == "cursor123"

    dumped = list_tools_res.model_dump(exclude_none=True)
    assert dumped == list_tools_result_data

def test_call_tool_request_and_result():
    # CallToolRequest
    call_params = mcp_types.CallToolRequestParams(name="calculator/add", arguments={"a": 1, "b": 2})
    call_req = mcp_types.CallToolRequest(id=10, params=call_params)
    assert call_req.method == "tools/call"
    assert call_req.params.name == "calculator/add"
    assert call_req.params.arguments == {"a": 1, "b": 2}

    # CallToolResult and Content Types
    text_content_data = {"type": "text", "text": "Hello"}
    image_content_data = {"type": "image", "mimeType": "image/png", "data": "base64data"}
    audio_content_data = {"type": "audio", "mimeType": "audio/mp3", "data": "base64audiodata"}

    text_res_contents_data = {"uri": "resource:/text/1", "text": "Embedded text"}
    embedded_res_data = {"type": "resource", "resource": text_res_contents_data} # Test with TextResourceContents

    call_result_data = {
        "content": [text_content_data, image_content_data, audio_content_data, embedded_res_data],
        "isError": False,
        "_meta": {"traceId": "xyz"}
    }
    call_res = mcp_types.CallToolResult.model_validate(call_result_data)
    assert len(call_res.content) == 4
    assert isinstance(call_res.content[0], mcp_types.TextContent)
    assert call_res.content[0].text == "Hello"
    assert isinstance(call_res.content[1], mcp_types.ImageContent)
    assert call_res.content[1].mimeType == "image/png"
    assert isinstance(call_res.content[2], mcp_types.AudioContent)
    assert call_res.content[2].mimeType == "audio/mp3"
    assert isinstance(call_res.content[3], mcp_types.EmbeddedResource)
    assert isinstance(call_res.content[3].resource, mcp_types.TextResourceContents)
    assert call_res.content[3].resource.uri == "resource:/text/1"
    assert call_res.isError is False
    assert call_res._meta == {"traceId": "xyz"}

    # Test deserialization of union CallToolResultContent
    parsed_text = mcp_types.CallToolResultContent.model_validate(text_content_data)
    assert isinstance(parsed_text, mcp_types.TextContent)
    parsed_image = mcp_types.CallToolResultContent.model_validate(image_content_data)
    assert isinstance(parsed_image, mcp_types.ImageContent)
    parsed_audio = mcp_types.CallToolResultContent.model_validate(audio_content_data)
    assert isinstance(parsed_audio, mcp_types.AudioContent)
    parsed_embed = mcp_types.CallToolResultContent.model_validate(embedded_res_data)
    assert isinstance(parsed_embed, mcp_types.EmbeddedResource)


def test_role_enum():
    assert mcp_types.RoleEnum.USER == "user"
    assert mcp_types.RoleEnum.ASSISTANT == "assistant"
    with pytest.raises(ValidationError):
        mcp_types.PromptMessage(role="invalid_role", content={"type": "text", "text": "hi"})

def test_logging_level_enum():
    assert mcp_types.LoggingLevelEnum.INFO == "info"
    assert mcp_types.LoggingLevelEnum.ERROR == "error"
    with pytest.raises(ValidationError):
        mcp_types.SetLevelRequestParams(level="invalid_level")

def test_tool_annotations():
    # Test defaults
    anno = mcp_types.ToolAnnotation()
    assert anno.readOnlyHint is False
    assert anno.idempotentHint is False
    assert anno.destructiveHint is True # Default from schema
    assert anno.openWorldHint is True # Default from schema

    # Test setting values
    anno_set = mcp_types.ToolAnnotation(
        title="My Tool",
        readOnlyHint=True,
        idempotentHint=True,
        destructiveHint=False,
        openWorldHint=False
    )
    assert anno_set.title == "My Tool"
    assert anno_set.readOnlyHint is True
    assert anno_set.idempotentHint is True
    assert anno_set.destructiveHint is False
    assert anno_set.openWorldHint is False

# Placeholder for more tests
# TODO:
# - Test models with more complex nesting or specific validation rules if any were missed.
# - Test deserialization with missing optional fields.
# - Test cases that should raise ValidationError for required fields or incorrect types.
# - Test `Annotations` model.
# - Test `*RequestParams` for various methods.
# - Test `*Result` for various methods, especially those with unions or lists.

from typing import Union # For RequestId type hint

```
