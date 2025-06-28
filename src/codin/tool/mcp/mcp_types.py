"""
Pydantic models for MCP (Model Context Protocol) types.

Generated from schema.json version 2025-03-26.
"""
import enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

# Base JSON-RPC Types
RequestId = Union[str, int]

class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    id: RequestId
    params: dict[str, Any] | None = None

class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: RequestId
    result: Any # Actual type varies based on the method

class JSONRPCErrorData(BaseModel):
    code: int
    message: str
    data: Any | None = None

class JSONRPCError(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: RequestId
    error: JSONRPCErrorData

class JSONRPCNotification(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] | None = None

# Client and Server Capabilities
class ClientCapabilities(BaseModel):
    experimental: dict[str, dict[str, Any]] | None = None
    roots: dict[str, Any] | None = None # Further definition if needed: {"listChanged": bool}
    sampling: dict[str, Any] | None = None # additionalProperties: true

class ServerCapabilities(BaseModel):
    completions: dict[str, Any] | None = None # additionalProperties: true
    experimental: dict[str, dict[str, Any]] | None = None
    logging: dict[str, Any] | None = None # additionalProperties: true
    prompts: dict[str, Any] | None = None # Further definition if needed: {"listChanged": bool}
    resources: dict[str, Any] | None = None # Further definition if needed: {"listChanged": bool, "subscribe": bool}
    tools: dict[str, Any] | None = None # Further definition if needed: {"listChanged": bool}

class Implementation(BaseModel):
    name: str
    version: str

# Initialize
class InitializeRequestParams(BaseModel):
    capabilities: ClientCapabilities
    clientInfo: Implementation
    protocolVersion: str

class InitializeRequest(JSONRPCRequest):
    method: Literal["initialize"] = "initialize"
    params: InitializeRequestParams

class InitializeResult(BaseModel):
    capabilities: ServerCapabilities
    serverInfo: Implementation
    protocolVersion: str
    instructions: str | None = None
    _meta: dict[str, Any] | None = None


# Tools
class ListToolsRequestParams(BaseModel):
    cursor: str | None = None

class ListToolsRequest(JSONRPCRequest):
    method: Literal["tools/list"] = "tools/list"
    params: ListToolsRequestParams | None = None

class ToolAnnotation(BaseModel): # Corresponds to ToolAnnotations in schema
    title: str | None = None
    readOnlyHint: bool | None = Field(default=False)
    idempotentHint: bool | None = Field(default=False)
    destructiveHint: bool | None = Field(default=True)
    openWorldHint: bool | None = Field(default=True)


# class ToolInputSchema(BaseModel):
#     type: Literal["object"] = "object"
#     properties: Optional[Dict[str, Dict[str, Any]]] = None # additionalProperties: true for inner dict
#     required: Optional[List[str]] = None


class Tool(BaseModel):
    name: str
    inputSchema: dict[str, Any] # As per instruction: "schema for inputSchema can be dict for now"
    description: str | None = None
    annotations: ToolAnnotation | None = None


class ListToolsResult(BaseModel):
    tools: list[Tool]
    nextCursor: str | None = None
    _meta: dict[str, Any] | None = None


class CallToolRequestParams(BaseModel):
    name: str
    arguments: dict[str, Any] | None = None

class CallToolRequest(JSONRPCRequest):
    method: Literal["tools/call"] = "tools/call"
    params: CallToolRequestParams

class Annotations(BaseModel):
    priority: float | None = None # Min 0, Max 1
    audience: list[Literal["user", "assistant"]] | None = None


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str
    annotations: Annotations | None = None

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    mimeType: str
    data: str # base64 encoded
    annotations: Annotations | None = None

class AudioContent(BaseModel):
    type: Literal["audio"] = "audio"
    mimeType: str
    data: str # base64 encoded
    annotations: Annotations | None = None

class TextResourceContents(BaseModel):
    uri: str # format: uri
    text: str
    mimeType: str | None = None

class BlobResourceContents(BaseModel):
    uri: str # format: uri
    blob: str # format: byte (base64 encoded)
    mimeType: str | None = None

class EmbeddedResource(BaseModel):
    type: Literal["resource"] = "resource"
    resource: TextResourceContents | BlobResourceContents
    annotations: Annotations | None = None

CallToolResultContent = Union[TextContent, ImageContent, AudioContent, EmbeddedResource]

class CallToolResult(BaseModel):
    content: list[CallToolResultContent]
    isError: bool | None = False
    _meta: dict[str, Any] | None = None


# Notifications
class CancelledNotificationParams(BaseModel):
    requestId: RequestId
    reason: str | None = None

class CancelledNotification(JSONRPCNotification):
    method: Literal["notifications/cancelled"] = "notifications/cancelled"
    params: CancelledNotificationParams

ProgressToken = Union[str, int]

class ProgressNotificationParams(BaseModel):
    progressToken: ProgressToken
    progress: float
    total: float | None = None
    message: str | None = None

class ProgressNotification(JSONRPCNotification):
    method: Literal["notifications/progress"] = "notifications/progress"
    params: ProgressNotificationParams

# Resources
class ListResourcesRequestParams(BaseModel):
    cursor: str | None = None

class ListResourcesRequest(JSONRPCRequest):
    method: Literal["resources/list"] = "resources/list"
    params: ListResourcesRequestParams | None = None

class Resource(BaseModel):
    name: str
    uri: str # format: uri
    description: str | None = None
    mimeType: str | None = None
    size: int | None = None
    annotations: Annotations | None = None

class ListResourcesResult(BaseModel):
    resources: list[Resource]
    nextCursor: str | None = None
    _meta: dict[str, Any] | None = None


class ListResourceTemplatesRequestParams(BaseModel):
    cursor: str | None = None

class ListResourceTemplatesRequest(JSONRPCRequest):
    method: Literal["resources/templates/list"] = "resources/templates/list"
    params: ListResourceTemplatesRequestParams | None = None

class ResourceTemplate(BaseModel):
    name: str
    uriTemplate: str # format: uri-template
    description: str | None = None
    mimeType: str | None = None
    annotations: Annotations | None = None

class ListResourceTemplatesResult(BaseModel):
    resourceTemplates: list[ResourceTemplate]
    nextCursor: str | None = None
    _meta: dict[str, Any] | None = None


class ReadResourceRequestParams(BaseModel):
    uri: str # format: uri

class ReadResourceRequest(JSONRPCRequest):
    method: Literal["resources/read"] = "resources/read"
    params: ReadResourceRequestParams

ReadResourceResultContents = Union[TextResourceContents, BlobResourceContents]

class ReadResourceResult(BaseModel):
    contents: list[ReadResourceResultContents]
    _meta: dict[str, Any] | None = None


# Subscription
class SubscribeRequestParams(BaseModel):
    uri: str # format: uri

class SubscribeRequest(JSONRPCRequest):
    method: Literal["resources/subscribe"] = "resources/subscribe"
    params: SubscribeRequestParams

class UnsubscribeRequestParams(BaseModel):
    uri: str # format: uri

class UnsubscribeRequest(JSONRPCRequest):
    method: Literal["resources/unsubscribe"] = "resources/unsubscribe"
    params: UnsubscribeRequestParams


# Prompts
class ListPromptsRequestParams(BaseModel):
    cursor: str | None = None

class ListPromptsRequest(JSONRPCRequest):
    method: Literal["prompts/list"] = "prompts/list"
    params: ListPromptsRequestParams | None = None

class PromptArgument(BaseModel):
    name: str
    description: str | None = None
    required: bool | None = None

class Prompt(BaseModel):
    name: str
    arguments: list[PromptArgument] | None = None
    description: str | None = None

class ListPromptsResult(BaseModel):
    prompts: list[Prompt]
    nextCursor: str | None = None
    _meta: dict[str, Any] | None = None


class GetPromptRequestParams(BaseModel):
    name: str
    arguments: dict[str, str] | None = None

class GetPromptRequest(JSONRPCRequest):
    method: Literal["prompts/get"] = "prompts/get"
    params: GetPromptRequestParams

PromptMessageContent = Union[TextContent, ImageContent, AudioContent, EmbeddedResource]

class RoleEnum(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"

class PromptMessage(BaseModel):
    role: RoleEnum
    content: PromptMessageContent

class GetPromptResult(BaseModel):
    messages: list[PromptMessage]
    description: str | None = None
    _meta: dict[str, Any] | None = None


# Logging
class LoggingLevelEnum(str, enum.Enum):
    EMERGENCY = "emergency"
    ALERT = "alert"
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    NOTICE = "notice"
    INFO = "info"
    DEBUG = "debug"

class SetLevelRequestParams(BaseModel):
    level: LoggingLevelEnum

class SetLevelRequest(JSONRPCRequest):
    method: Literal["logging/setLevel"] = "logging/setLevel"
    params: SetLevelRequestParams


# Completion
class CompleteRequestParamsArgument(BaseModel):
    name: str
    value: str

class PromptReference(BaseModel):
    type: Literal["ref/prompt"] = "ref/prompt"
    name: str

class ResourceReference(BaseModel):
    type: Literal["ref/resource"] = "ref/resource"
    uri: str # format: uri-template

class CompleteRequestParams(BaseModel):
    argument: CompleteRequestParamsArgument
    ref: PromptReference | ResourceReference


class CompleteRequest(JSONRPCRequest):
    method: Literal["completion/complete"] = "completion/complete"
    params: CompleteRequestParams

class CompletionData(BaseModel):
    values: list[str]
    hasMore: bool | None = None
    total: int | None = None

class CompleteResult(BaseModel):
    completion: CompletionData
    _meta: dict[str, Any] | None = None


# __all__ definition
__all__ = [
    "RequestId",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCErrorData",
    "JSONRPCError",
    "JSONRPCNotification",
    "ClientCapabilities",
    "ServerCapabilities",
    "Implementation",
    "InitializeRequestParams",
    "InitializeRequest",
    "InitializeResult",
    "ListToolsRequestParams",
    "ListToolsRequest",
    "ToolAnnotation",
    # "ToolInputSchema", # No longer a separate model
    "Tool",
    "ListToolsResult",
    "CallToolRequestParams",
    "CallToolRequest",
    "Annotations",
    "TextContent",
    "ImageContent",
    "AudioContent",
    "TextResourceContents",
    "BlobResourceContents",
    "EmbeddedResource",
    "CallToolResultContent",
    "CallToolResult",
    "CancelledNotificationParams",
    "CancelledNotification",
    "ProgressToken",
    "ProgressNotificationParams",
    "ProgressNotification",
    "ListResourcesRequestParams",
    "ListResourcesRequest",
    "Resource",
    "ListResourcesResult",
    "ListResourceTemplatesRequestParams",
    "ListResourceTemplatesRequest",
    "ResourceTemplate",
    "ListResourceTemplatesResult",
    "ReadResourceRequestParams",
    "ReadResourceRequest",
    "ReadResourceResultContents",
    "ReadResourceResult",
    "SubscribeRequestParams",
    "SubscribeRequest",
    "UnsubscribeRequestParams",
    "UnsubscribeRequest",
    "ListPromptsRequestParams",
    "ListPromptsRequest",
    "PromptArgument",
    "Prompt",
    "ListPromptsResult",
    "GetPromptRequestParams",
    "GetPromptRequest",
    "PromptMessageContent",
    "RoleEnum",
    "PromptMessage",
    "GetPromptResult",
    "LoggingLevelEnum",
    "SetLevelRequestParams",
    "SetLevelRequest",
    "CompleteRequestParamsArgument",
    "PromptReference",
    "ResourceReference",
    "CompleteRequestParams",
    "CompleteRequest",
    "CompletionData",
    "CompleteResult",
]

# TODO:
# - Review all Optional fields and default values based on schema.
# - For fields that are objects with additionalProperties, ensure dict[str, Any] is used if not already.
#   (e.g. JSONRPCRequest.params, JSONRPCNotification.params, ClientCapabilities.experimental etc.)
# - Double check `Tool.inputSchema` - current definition is `ToolInputSchema`, not `dict`.
#   The instruction was "schema for inputSchema can be dict for now".
#   I've used a Pydantic model `ToolInputSchema` which is more specific than `dict`.
#   If a plain `dict` is strictly required, this should be changed.
# - Review `Role` enum, it's used in `Annotations` and `PromptMessage`.
# - Review `LoggingLevel` enum.
# - Add missing imports if any, sort them.
# - Add `mcp_types` to `src/codin/tool/mcp/__init__.py` and its `__all__`.

"""
Notes during generation:
- `JSONRPCRequest.params`, `JSONRPCNotification.params`: The schema often defines specific structures for these `params` depending on the `method`.
  I've used `Optional[Dict[str, Any]]` for the base classes and then specific Pydantic models for params in derived request/notification classes.
- `ClientCapabilities.experimental`: `additionalProperties: {additionalProperties: true, properties: {}, type: "object"}` -> `Dict[str, Dict[str, Any]]`
- `ServerCapabilities.experimental`: Same as above.
- `ServerCapabilities.completions`, `logging`, `prompts`, `resources`, `tools`: Some have specific sub-properties like `listChanged`.
  I've used `Optional[Dict[str, Any]]` for now, but these could be more strictly typed if needed.
  For example, `ServerCapabilities.prompts` could be `Optional[PromptsCapability]` where `PromptsCapability(BaseModel): listChanged: Optional[bool] = None`.
  The schema also uses `additionalProperties: true` for some of these, which `Dict[str, Any]` covers.
- `Tool.inputSchema`: Schema defines properties like `type: "object"`, `properties`, `required`.
  I've created `ToolInputSchema` for this. If it should strictly be `dict`, then `inputSchema: Dict[str, Any]` should be used.
  The current `ToolInputSchema` is more type-safe.
- `CallToolResultContent`: Defined as a Union of `TextContent`, `ImageContent`, `AudioContent`, `EmbeddedResource`.
- `ReadResourceResultContents`: Defined as a Union of `TextResourceContents`, `BlobResourceContents`.
- `PromptMessageContent`: Defined as a Union of `TextContent`, `ImageContent`, `AudioContent`, `EmbeddedResource`.
- `Role`: Defined as `RoleEnum` with "user" and "assistant". Used in `Annotations` and `PromptMessage`.
- `LoggingLevel`: Defined as `LoggingLevelEnum`.
- `_meta` fields: Added as `Optional[Dict[str, Any]]`.
- `Annotations` model: Defined for reuse in `TextContent`, `ImageContent`, `AudioContent`, `EmbeddedResource`, `Resource`, `ResourceTemplate`, `Tool`.
- `ToolAnnotation` vs `Annotations`: The schema has `ToolAnnotations` for `Tool.annotations` and `Annotations` for content types. I've named them `ToolAnnotation` and `Annotations` respectively.
- `CompleteRequestParams.ref`: Union of `PromptReference` and `ResourceReference`.
- `JSONRPCResponse.result`: This is `Any` because it varies significantly. Specific response types will embed their specific result model (e.g. `InitializeResult`, `ListToolsResult`).
  The task asks for `JSONRPCResponse` model. It's a generic wrapper. The actual result parsing would happen based on context.

Final check on constraints:
- All models use Pydantic's `BaseModel`. (Yes)
- Appropriate type hints. (Yes, to the best of my ability from the schema)
- `additionalProperties` -> `dict[str, Any]`. (Yes, where applicable or simplified to `Dict[str, Any]`)
- Enums: `typing_extensions.Literal` or `enum.Enum`. (Used `Literal` for fixed strings like `jsonrpc: "2.0"` and `type: "text"`, and `enum.Enum` for `RoleEnum`, `LoggingLevelEnum`)
- Optional fields with `Optional[...]` and default values. (Yes, used `Optional` and `Field(default=...)` where appropriate, e.g. for booleans in `ToolAnnotation`)
- `__all__` added. (Yes)
"""
