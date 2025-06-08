"""Prompt execution engine for codin agents.

This module provides the core prompt execution engine that handles
template rendering, LLM interaction, and response processing.

Elegant and concise prompt engine with automatic LLM capability detection.
"""

from __future__ import annotations

import logging
import typing as _t
import uuid
from datetime import datetime

# Use agent protocol types directly
from codin.agent.types import Message, Role, TextPart

from ..model import BaseLLM, ModelRegistry
from .base import PromptResponse, RenderedPrompt, ToolDefinition
from .registry import get_registry

__all__ = ["PromptEngine"]

logger = logging.getLogger("codin.prompt.engine")


class PromptEngine:
    """Elegant prompt engine with automatic LLM detection and simple API."""

    def __init__(self, llm: BaseLLM | str | None = None, endpoint: str | None = None):
        """Initialize prompt engine.

        Args:
            llm: Optional. An LLM instance, model name (str), or None.
                 If None or not provided, the `run` method will attempt to create an
                 LLM from the environment settings. If this creation fails, `run`
                 will raise a ValueError.
                 The `render` method does not require an LLM instance.
            endpoint: Optional. The storage endpoint URL for prompt templates
                 (e.g., "fs://./prompt_templates"). Defaults to a local filesystem
                 path derived from environment variables or a standard default.
        """
        self.llm = self._resolve_llm(llm) if llm is not None else None
        self.endpoint = endpoint

    def _resolve_llm(self, llm: BaseLLM | str) -> BaseLLM:
        """Resolve LLM from string or return instance."""
        if isinstance(llm, str):
            return ModelRegistry.create_llm(llm)
        return llm

    def _detect_capabilities(self, llm: BaseLLM) -> dict[str, _t.Any]:
        """Detect LLM capabilities for template selection."""
        model_name = getattr(llm, "model", llm.__class__.__name__.lower())

        # Simple model family detection
        if "claude" in model_name.lower():
            family, provider = "claude", "anthropic"
        elif "gpt" in model_name.lower() or "openai" in model_name.lower():
            family, provider = "openai", "openai"
        elif "gemini" in model_name.lower() or "google" in model_name.lower():
            family, provider = "google", "google"
        else:
            family, provider = "unknown", "unknown"

        return {
            "model": model_name,
            "model_family": family,
            "model_provider": provider,
            "tool_support": hasattr(llm, "generate_with_tools"),
            "multimodal": getattr(llm, "supports_multimodal", False),
        }

    def _create_message(
        self, content: str, role: Role = Role.agent, context_id: str | None = None, task_id: str | None = None
    ) -> Message:
        """Create an A2A message."""
        return Message(
            messageId=str(uuid.uuid4()),
            role=role,
            parts=[TextPart(text=content)],
            contextId=context_id,
            taskId=task_id,
            kind="message",
        )

    async def _prepare_variables(
        self, variables: dict[str, _t.Any], **kwargs
    ) -> dict[str, _t.Any]:
        """Prepare template variables for rendering by merging and processing tools.

        This method takes an initial `variables` dictionary, merges `kwargs` into it
        (kwargs take precedence), and then processes a "tools" key expected within
        the original `variables` dictionary. The processed tool information (such as
        tool descriptions, names, and a flag indicating presence of tools) is added
        to the resulting variable set.

        Args:
            variables: The base dictionary of variables. If it contains a "tools"
                key, this key is expected to hold a list of `ToolDefinition`
                objects or dictionaries that can be converted to `ToolDefinition`.
            **kwargs: Additional keyword arguments to be merged into the variables.
                These will override any keys with the same name in `variables`.

        Returns:
            A new dictionary containing the merged variables from `variables` and
            `kwargs`, augmented with specific context derived from the processed
            "tools" (e.g., `has_tools`, `tool_names`, `tool_descriptions`, and
            a representation of "tools" suitable for templates).
        """
        # Make a copy of the original variables to avoid modifying it if passed by reference elsewhere
        processed_vars = variables.copy()
        processed_vars.update(kwargs) # Merge kwargs. kwargs can override initial variables.

        # Extract tools from the 'variables' dictionary (original ones, before kwargs merge for safety,
        # or from processed_vars if kwargs are allowed to provide/override tools)
        # Subtask says "retrieve tools ... from the variables dictionary" (the parameter).
        # Let's stick to extracting from the original 'variables' parameter.
        tools_data = variables.get("tools") # This could be List[ToolDefinition] or List[dict]

        if tools_data:
            # Ensure tools_data is in the ToolDefinition format if it's not already
            # This logic might need adjustment depending on what format 'tools' is expected in 'variables'
            # For now, assume it's List[ToolDefinition] or compatible dicts
            tool_defs = []
            if isinstance(tools_data, list):
                for item in tools_data:
                    if isinstance(item, ToolDefinition):
                        tool_defs.append(item)
                    elif isinstance(item, dict): # If tools are passed as dicts
                        tool_defs.append(ToolDefinition(**item))

            tool_dicts = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tool_defs]
            processed_vars.update(
                {
                    "tools": tool_dicts, # This will be list of dicts for the template
                    "has_tools": True,
                    "tool_descriptions": [f"- {t.name}: {t.description}" for t in tool_defs],
                    "tool_names": [t.name for t in tool_defs],
                }
            )
        else:
            processed_vars.update({"has_tools": False, "tool_descriptions": [], "tool_names": []})

        return processed_vars

    async def run(
        self,
        name: str,
        /,
        *,
        version: str | None = None,
        variables: dict[str, _t.Any] | None = None,
        conditions: dict[str, _t.Any] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> PromptResponse:
        """Execute a prompt template with elegant simplicity.

        Args:
            name: Template name
            version: Template version (optional)
            variables: A dictionary of template variables. This dictionary can
                optionally include the following special keys:
                - "tools": A list of `ToolDefinition` objects or dictionaries
                  that can be converted to `ToolDefinition`. These are used for
                  providing tool context to the template and for enabling
                  tool use with the LLM if supported.
                - "context_id": An optional string identifier for the context
                  of the A2A message generated by the `run` method.
                - "task_id": An optional string identifier for the task
                  associated with the A2A message.
                Other keys in this dictionary are passed as variables to the
                prompt template during rendering.
            conditions: Template selection conditions
            stream: Stream response
            **kwargs: Additional variables

        Returns:
            A2AResponse with result

        Note:
            If an LLM was not provided during engine initialization, this method
            will attempt to create one from environment settings. If this fails,
            a ValueError will be raised.
        """
        if self.llm is None:
            # Attempt to create LLM from environment if not provided during construction
            # and also not explicitly passed to run (though run doesn't accept llm directly)
            # This logic primarily covers the case where __init__ receives None.
            # If an LLM is critical, it should be ensured by the caller or by a specific method call.
            # However, the request is to raise error if self.llm is None at this point.
            from ..model.factory import create_llm_from_env

            try:
                self.llm = create_llm_from_env()
            except Exception as e: # Broad exception to catch if create_llm_from_env fails
                 raise ValueError(
                    "LLM instance is required for execution, but none was provided or could be created from environment."
                 ) from e

        # If after attempting to create from env, llm is still None, then raise error.
        # This check is now more specific after the attempt above.
        if self.llm is None:
            raise ValueError("LLM instance is required for execution, but none was provided.")

        try:
            # Load template
            registry = get_registry(self.endpoint)
            template = await registry.get_async(name, version)

            # Prepare LLM if needed
            if hasattr(self.llm, "prepare"):
                # Check if LLM has _prepared attribute to track preparation state
                if hasattr(self.llm, "_prepared"):
                    if not self.llm._prepared:
                        await self.llm.prepare()
                else:
                    # LLM doesn't track preparation state, so always prepare
                    await self.llm.prepare()

            # Detect capabilities and merge with conditions
            capabilities = self._detect_capabilities(self.llm)
            final_conditions = (conditions or {}) | capabilities

            # Get best variant
            variant = template.get_best_variant(final_conditions)
            if not variant:
                raise ValueError(f"No suitable variant found for template {name}")

            # Prepare all variables
            # 'variables' here is the parameter to run(), which might be None.
            # _prepare_variables expects a dict.
            current_variables = variables or {}
            all_variables = await self._prepare_variables(current_variables, **kwargs)

            # The 'tools' variable for LLM interaction should come from the processed 'all_variables'
            # or more directly from what was in the input 'variables' dict.
            # The 'tools' entry in 'all_variables' is now a list of dicts for the template.
            # For LLM interaction, we need the original ToolDefinition objects or compatible dicts.
            # Let's re-fetch from original 'variables' for clarity for the LLM part.
            llm_tools_source = current_variables.get("tools") # This would be List[ToolDefinition] or List[dict]


            all_variables.update(
                {
                    "model": capabilities["model"],
                    "model_family": capabilities["model_family"],
                    "model_provider": capabilities["model_provider"],
                    "streaming": stream,
                }
            )

            # Render template (full rendering with system prompt and messages)
            rendered_prompt = template.render(conditions=final_conditions, **all_variables)

            logger.debug("Calling LLM %s with template %s@%s", capabilities["model"], template.name, template.version)

            # Execute with LLM
            model_options = rendered_prompt.model_options

            # Determine if we should use messages format or single text
            if rendered_prompt.system_prompt or rendered_prompt.messages:
                # Use structured message format if available
                messages = rendered_prompt.to_messages()

                # 'llm_tools_source' was extracted above from current_variables.get("tools")
                if llm_tools_source and capabilities["tool_support"] and hasattr(self.llm, "generate_with_tools"):
                    # Convert tools for LLM if they are ToolDefinition objects
                    llm_api_tools = []
                    if isinstance(llm_tools_source, list):
                        for tool_item in llm_tools_source:
                            if isinstance(tool_item, ToolDefinition):
                                llm_api_tools.append({
                                    "type": "function",
                                    "function": {"name": tool_item.name, "description": tool_item.description, "parameters": tool_item.parameters},
                                })
                            elif isinstance(tool_item, dict): # Assume it's already in LLM API format or compatible
                                llm_api_tools.append(tool_item)

                    # Check if LLM supports message-based generation with tools
                    if hasattr(self.llm, "generate_messages_with_tools"):
                        completion = await self.llm.generate_messages_with_tools(
                            messages, tools=llm_api_tools, stream=stream, **model_options
                        )
                    else:
                        # Fallback to text-based generation
                        completion = await self.llm.generate_with_tools(
                            rendered_prompt.text, tools=llm_api_tools, stream=stream, **model_options
                        )
                # Check if LLM supports message-based generation
                elif hasattr(self.llm, "generate_messages"):
                    completion = await self.llm.generate_messages(messages, stream=stream, **model_options)
                else:
                    # Fallback to text-based generation
                    completion = await self.llm.generate(rendered_prompt.text, stream=stream, **model_options)
            # Use traditional single text approach
            # 'llm_tools_source' was extracted above from current_variables.get("tools")
            elif llm_tools_source and capabilities["tool_support"] and hasattr(self.llm, "generate_with_tools"):
                llm_api_tools = []
                if isinstance(llm_tools_source, list):
                    for tool_item in llm_tools_source:
                        if isinstance(tool_item, ToolDefinition):
                            llm_api_tools.append({
                                "type": "function",
                                "function": {"name": tool_item.name, "description": tool_item.description, "parameters": tool_item.parameters},
                            })
                        elif isinstance(tool_item, dict):
                             llm_api_tools.append(tool_item)

                completion = await self.llm.generate_with_tools(
                    rendered_prompt.text, tools=llm_api_tools, stream=stream, **model_options
                )
            else:
                completion = await self.llm.generate(rendered_prompt.text, stream=stream, **model_options)

            # Create A2A response
            response = PromptResponse(streaming=stream)

            if stream and hasattr(completion, "__aiter__"):
                response.content = completion
            else:
                content_str = str(completion)
                # Get context_id and task_id from all_variables
                _context_id = all_variables.get("context_id")
                _task_id = all_variables.get("task_id")
                response.message = self._create_message(
                    content=content_str, role=Role.agent, context_id=_context_id, task_id=_task_id
                )
                response.content = content_str

            return response

        except Exception as e:
            logger.error(f"Error in prompt execution: {e}", exc_info=True)
            return PromptResponse(
                error={"type": "execution_error", "message": str(e), "timestamp": datetime.now().isoformat()}
            )

    async def render(
        self,
        name: str,
        /,
        *,
        version: str | None = None,
        variables: dict[str, _t.Any] | None = None,
        conditions: dict[str, _t.Any] | None = None,
        **kwargs,
    ) -> str:
        """Render a template's text content without executing LLM.

        Args:
            name: Template name
            version: Template version (optional)
            variables: Template variables
            conditions: Template selection conditions
            **kwargs: Additional variables

        Returns:
            The rendered text content of the prompt.
        """
        registry = get_registry(self.endpoint)
        template = await registry.get_async(name, version)

        all_variables = (variables or {}) | kwargs
        rendered_prompt = template.render(conditions=conditions, **all_variables)
        return rendered_prompt.text
