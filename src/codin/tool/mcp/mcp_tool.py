"""MCP tool implementation for codin agents.

This module provides the MCPTool class that wraps individual tools
from MCP servers and integrates them into the codin tool system.
"""

from __future__ import annotations

import logging
import typing as _t

import pydantic as _pyd

from ..base import LifecycleState, Tool, ToolContext
from .session_manager import MCPSessionManager
from .utils import retry_on_closed_resource
from . import mcp_types
from .exceptions import ( # Import custom exceptions
    MCPError,
    MCPConnectionError,
    MCPInputError,
    MCPProtocolError,
    MCPToolError
)

__all__ = [
    'MCPTool',
]

_logger = logging.getLogger(__name__)

MCP_METHOD_MAP: dict[str, tuple[str, _t.Type[_pyd.BaseModel] | None, _t.Type[_pyd.BaseModel] | None ]] = {
    "resources/list": ("list_resources", mcp_types.ListResourcesRequestParams, mcp_types.ListResourcesResult),
    "resources/templates/list": ("list_resource_templates", mcp_types.ListResourceTemplatesRequestParams, mcp_types.ListResourceTemplatesResult),
    "resources/read": ("read_resource", mcp_types.ReadResourceRequestParams, mcp_types.ReadResourceResult),
    "resources/subscribe": ("subscribe_resource", mcp_types.SubscribeRequestParams, None),
    "resources/unsubscribe": ("unsubscribe_resource", mcp_types.UnsubscribeRequestParams, None),
    "prompts/list": ("list_prompts", mcp_types.ListPromptsRequestParams, mcp_types.ListPromptsResult),
    "prompts/get": ("get_prompt", mcp_types.GetPromptRequestParams, mcp_types.GetPromptResult),
    "logging/setLevel": ("set_logging_level", mcp_types.SetLevelRequestParams, None),
    "completion/complete": ("get_completion", mcp_types.CompleteRequestParams, mcp_types.CompleteResult),
}

class _DefaultInputModel(_pyd.BaseModel):
    class Config:
        extra = 'allow'


class MCPTool(Tool):
    tool_annotations: mcp_types.ToolAnnotation | None
    mcp_input_schema_dict: dict[str, _t.Any] | None

    def __init__(
        self,
        *,
        name: str,
        description: str,
        session_manager: MCPSessionManager,
        input_schema: type[_pyd.BaseModel] | None = None,
        is_generative: bool = False,
        tool_annotations: mcp_types.ToolAnnotation | None = None,
        mcp_input_schema_dict: dict[str, _t.Any] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            input_schema=input_schema or _DefaultInputModel,
            is_generative=is_generative,
        )
        self._session_manager = session_manager
        self.tool_annotations = tool_annotations
        self.mcp_input_schema_dict = mcp_input_schema_dict
        self._state = LifecycleState.DISCONNECTED # Initial state

    async def initialize(self) -> None:
        """Initialize the MCP tool by calling the session manager's initialize method."""
        _logger.debug(f"MCPTool {self.name}: Initializing session.")
        try:
            await self._session_manager.initialize()
            if self._session_manager.server_capabilities:
                _logger.info(
                    f"MCPTool {self.name}: Session initialized. Server capabilities obtained."
                )
            else:
                _logger.warning(
                    f"MCPTool {self.name}: Session initialized, but server capabilities not available/retrieved."
                )
            self._state = LifecycleState.UP
        except MCPError as e: # Catch specific MCP errors from session manager
            _logger.error(f'MCPTool {self.name}: Failed to initialize MCP session due to MCPError: {e}')
            self._state = LifecycleState.ERROR
            raise MCPToolError(f"Session initialization failed for {self.name}: {e}") from e
        except Exception as e: # Catch any other unexpected errors
            _logger.exception(f'MCPTool {self.name}: Unexpected error during session initialization: {e}')
            self._state = LifecycleState.ERROR
            raise MCPToolError(f"Unexpected error during session initialization for {self.name}: {e}") from e

    async def cleanup(self) -> None:
        _logger.debug(f"MCPTool {self.name}: Cleaning up.")
        # Session manager's close is typically handled by its owner (e.g. MCPToolset or main app)
        self._state = LifecycleState.DISCONNECTED

    @retry_on_closed_resource('_reinitialize_session')
    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> _t.Any:
        _logger.debug("MCPTool '%s': Running with args: %s", self.name, args)
        if self._state != LifecycleState.UP:
            _logger.warning(f"MCPTool {self.name} is not in UP state ({self._state}). Attempting to reinitialize.")
            await self._reinitialize_session() # This will call self.initialize()
            if self._state != LifecycleState.UP: # Check state again after reinitialization attempt
                 raise MCPToolError(f"MCPTool {self.name} could not be reinitialized. Current state: {self._state}")


        if self.name in MCP_METHOD_MAP:
            session_method_name, RequestParamsModel, ResponseModel = MCP_METHOD_MAP[self.name]
            _logger.debug(f"MCPTool {self.name}: Matched standard MCP method. Session method: {session_method_name}")

            try:
                params_obj = None
                if RequestParamsModel:
                    params_obj = RequestParamsModel.model_validate(args)

                session_method = getattr(self._session_manager, session_method_name)

                if ResponseModel is None: # For methods like subscribe/unsubscribe that return None
                    await session_method(params_obj)
                    return {"status": "success", "message": f"{self.name} completed successfully."} # Provide some feedback
                else:
                    response_obj = await session_method(params_obj)
                    return response_obj.model_dump(exclude_none=True)

            except _pyd.ValidationError as e:
                _logger.error("MCPTool %s: Pydantic validation error for input args: %s. Args: %s", self.name, e, args)
                raise MCPInputError(f"Invalid arguments for {self.name}: {e}") from e
            except MCPError as e: # Propagate MCP specific errors from session manager
                _logger.error("MCPTool %s: MCPError from session manager: %s", self.name, e)
                raise # Re-raise directly as they are already specific
            except Exception as e: # Catch any other unexpected error from session manager call
                _logger.exception("MCPTool %s: Unexpected error calling standard MCP method '%s': %s", self.name, self.name, e)
                raise MCPToolError(f"Unexpected error executing {self.name}: {e}") from e
        else:
            # Fallback to custom tool call via session_manager.call_tool
            _logger.debug("MCPTool %s: Not found in MCP_METHOD_MAP, treating as custom tool call.", self.name)
            try:
                raw_result = await self._session_manager.call_tool(self.name, args)

                try:
                    call_tool_result = mcp_types.CallToolResult.model_validate(raw_result)
                    _logger.debug(f"MCPTool {self.name}: Parsed custom tool result as CallToolResult.")
                    processed_content = []
                    for item in call_tool_result.content:
                        if isinstance(item, mcp_types.TextContent):
                            processed_content.append(item.text)
                        elif isinstance(item, mcp_types.ImageContent):
                            processed_content.append(f"[Image content: {item.mimeType}, data: {item.data[:30]}...]")
                        elif isinstance(item, mcp_types.AudioContent):
                            processed_content.append(f"[Audio content: {item.mimeType}, data: {item.data[:30]}...]")
                        elif isinstance(item, mcp_types.EmbeddedResource):
                            res_uri = item.resource.uri
                            res_type = "text" if isinstance(item.resource, mcp_types.TextResourceContents) else "blob"
                            processed_content.append(f"[Embedded {res_type} resource: {res_uri}]")
                        else:
                            # This case should ideally not happen if CallToolResultContent is exhaustive
                            _logger.warning(f"MCPTool {self.name}: Unknown content type in CallToolResult: {type(item)}")
                            processed_content.append(f"[Unknown content type: {type(item)}]")

                    if not processed_content: return None
                    return "\n".join(processed_content) if len(processed_content) > 1 else processed_content[0]

                except _pyd.ValidationError as e:
                    _logger.warning(
                        "MCPTool %s: Result for custom tool is not a valid CallToolResult model. Error: %s. Returning raw result.",
                        self.name, e
                    )
                    # Consider raising MCPProtocolError here if strict parsing is required.
                    # For now, returning raw_result for compatibility with tools not fully adhering to CallToolResult.
                    return raw_result

            except MCPError as e: # Propagate MCP specific errors
                _logger.error("MCPTool %s: MCPError from session manager during custom tool call: %s", self.name, e)
                raise
            except Exception as e:
                _logger.exception("MCPTool %s: Unexpected error calling custom MCP tool '%s': %s", self.name, self.name, e)
                raise MCPToolError(f"Unexpected error executing custom tool {self.name}: {e}") from e

    async def _reinitialize_session(self) -> None:
        """Re-initialize the session if connection was lost or state is ERROR."""
        _logger.info(f"MCPTool {self.name}: Attempting to reinitialize session.")
        # Set state to disconnected before attempting to initialize again
        self._state = LifecycleState.DISCONNECTED
        try:
            await self.initialize() # This method now handles setting state to UP or ERROR
        except MCPToolError: # self.initialize already logs and raises MCPToolError
            _logger.error(f"MCPTool {self.name}: Failed to reinitialize session.")
            # State will be ERROR, as set by self.initialize()
        except Exception: # Should be caught by self.initialize, but as a safeguard
            _logger.exception(f"MCPTool {self.name}: Unexpected error during _reinitialize_session.")
            self._state = LifecycleState.ERROR

