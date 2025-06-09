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
            llm: LLM instance, model name, or None to auto-select
            endpoint: Storage endpoint (None for default)
        """
        self.llm = self._resolve_llm(llm) if llm else None
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
        self, variables: dict[str, _t.Any], tools: list[ToolDefinition] | None, **kwargs
    ) -> dict[str, _t.Any]:
        """Prepare template variables with tools context."""
        all_vars = (variables or {}) | kwargs

        # Add tool context
        if tools:
            tool_dicts = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools]
            all_vars.update(
                {
                    "tools": tool_dicts,
                    "has_tools": True,
                    "tool_descriptions": [f"- {t.name}: {t.description}" for t in tools],
                    "tool_names": [t.name for t in tools],
                }
            )
        else:
            all_vars.update({"has_tools": False, "tool_descriptions": [], "tool_names": []})

        return all_vars

    async def run(
        self,
        name: str,
        /,
        *,
        version: str | None = None,
        variables: dict[str, _t.Any] | None = None,
        tools: list[ToolDefinition] | None = None,
        conditions: dict[str, _t.Any] | None = None,
        stream: bool = False,
        context_id: str | None = None,
        task_id: str | None = None,
        **kwargs,
    ) -> PromptResponse:
        """Execute a prompt template with elegant simplicity.

        Args:
            name: Template name
            version: Template version (optional)
            variables: Template variables
            tools: Available tools
            conditions: Template selection conditions
            stream: Stream response
            context_id: A2A context ID
            task_id: A2A task ID
            **kwargs: Additional variables

        Returns:
            A2AResponse with result
        """
        try:
            # Load template
            registry = get_registry(self.endpoint)
            template = await registry.get_async(name, version)

            # Get or create LLM
            if not self.llm:
                from ..model.factory import create_llm_from_env

                self.llm = await create_llm_from_env() # create_llm_from_env is now async

            # LLM client is prepared in its async __init__. No separate prepare() call needed.
            # capabilities = self._detect_capabilities(self.llm) # self.llm must be non-None here
            if not self.llm: # Should be caught by earlier logic or raise specific error
                 raise RuntimeError("LLM could not be initialized in PromptEngine.")

            # Detect capabilities and merge with conditions
            capabilities = self._detect_capabilities(self.llm)
            final_conditions = (conditions or {}) | capabilities

            # Get best variant
            variant = template.get_best_variant(final_conditions)
            if not variant:
                raise ValueError(f"No suitable variant found for template {name}")

            # Prepare all variables
            all_variables = await self._prepare_variables(variables, tools, **kwargs)
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

                if tools and capabilities["tool_support"] and hasattr(self.llm, "generate_with_tools"):
                    # Convert tools for LLM
                    llm_tools = [
                        {
                            "type": "function",
                            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                        }
                        for t in tools
                    ]

                    # Check if LLM supports message-based generation with tools
                    if hasattr(self.llm, "generate_messages_with_tools"):
                        completion = await self.llm.generate_messages_with_tools(
                            messages, tools=llm_tools, stream=stream, **model_options
                        )
                    else:
                        # Fallback to text-based generation
                        completion = await self.llm.generate_with_tools(
                            rendered_prompt.text, tools=llm_tools, stream=stream, **model_options
                        )
                # Check if LLM supports message-based generation
                elif hasattr(self.llm, "generate_messages"):
                    completion = await self.llm.generate_messages(messages, stream=stream, **model_options)
                else:
                    # Fallback to text-based generation
                    completion = await self.llm.generate(rendered_prompt.text, stream=stream, **model_options)
            # Use traditional single text approach
            elif tools and capabilities["tool_support"] and hasattr(self.llm, "generate_with_tools"):
                # Convert tools for LLM
                llm_tools = [
                    {
                        "type": "function",
                        "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                    }
                    for t in tools
                ]

                completion = await self.llm.generate_with_tools(
                    rendered_prompt.text, tools=llm_tools, stream=stream, **model_options
                )
            else:
                completion = await self.llm.generate(rendered_prompt.text, stream=stream, **model_options)

            # Create A2A response
            response = PromptResponse(streaming=stream)

            if stream and hasattr(completion, "__aiter__"):
                response.content = completion
            else:
                content_str = str(completion)
                response.message = self._create_message(
                    content=content_str, role=Role.agent, context_id=context_id, task_id=task_id
                )
                response.content = content_str

            return response

        except Exception as e:
            logger.error(f"Error in prompt execution: {e}", exc_info=True)
            return PromptResponse(
                error={"type": "execution_error", "message": str(e), "timestamp": datetime.now().isoformat()}
            )

    async def render_only(
        self,
        name: str,
        /,
        *,
        version: str | None = None,
        variables: dict[str, _t.Any] | None = None,
        conditions: dict[str, _t.Any] | None = None,
        **kwargs,
    ) -> RenderedPrompt:
        """Render a template without executing LLM.

        Args:
            name: Template name
            version: Template version (optional)
            variables: Template variables
            conditions: Template selection conditions
            **kwargs: Additional variables

        Returns:
            RenderedPrompt instance
        """
        registry = get_registry(self.endpoint)
        template = await registry.get_async(name, version)

        all_variables = (variables or {}) | kwargs
        return template.render(conditions=conditions, **all_variables)
