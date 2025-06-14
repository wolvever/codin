"""BasePlanner is a planner that uses LLM to generate execution steps, adapted from CodeAgent."""

import json
import logging
import re
import typing as _t
import uuid
from datetime import datetime

from ..id import new_id
from ..prompt.run import prompt_run
from ..tool.base import to_tool_definitions
from ..utils.message import (
    format_history_for_prompt, # extract_text_from_message removed
    format_tool_results_for_conversation,
)
from .base import Planner
from .types import (
    ErrorStep,
    FinishStep,
    Message,
    MessageStep,
    Role,
    State,
    Step,
    TextPart, # TextPart is used by Message.from_text, but not directly here anymore
    ThinkStep,
    ToolCall,
    ToolCallStep,
)

__all__ = [
    "BasePlanner",
]

logger = logging.getLogger("codin.agent.base_planner")


class BasePlanner(Planner):
    """Planner that uses LLM to generate execution steps, adapted from CodeAgent."""

    def __init__(
        self,
        *,
        prompt_name: str = "code_agent_loop",
        max_tool_calls_per_turn: int = 10,
        enable_thinking: bool = True,
        enable_streaming: bool = True,
        rules: str | None = None,
    ):
        """Initialize the BasePlanner."""

        self.prompt_name = prompt_name
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.thinking_enabled = enable_thinking
        self.streaming_enabled = enable_streaming
        self.rules = rules

        logger.info("Initialized BasePlanner")

    async def next(self, state: State) -> _t.AsyncGenerator[Step]:
        """Generate execution steps based on current state."""
        step_count = 0

        # Build the prompt variables from state
        variables = await self._build_prompt_variables(state)

        logger.debug(
            f"Planning for session {state.session_id}, turn {state.turn_count}, variables: {variables}"
        )

        try:
            # Get LLM response using prompt_run
            response = await prompt_run(
                self.prompt_name,
                variables=variables,
                tools=variables.get("tools", []),
                stream=self.streaming_enabled,
            )

            stream_chunks: list[str] = []
            if (
                response.streaming
                and self.streaming_enabled
                and isinstance(response.content, _t.AsyncIterator)
            ):
                step = MessageStep(
                    step_id=str(uuid.uuid4()),
                    is_streaming=True,
                    created_at=datetime.now(),
                    metadata={"turn": state.turn_count},
                )

                async def _iter() -> _t.AsyncIterator[str]:
                    async for chunk in response.content:  # type: ignore[arg-type]
                        stream_chunks.append(str(chunk))
                        yield str(chunk)
                    step.message = Message.from_text( # Updated
                        text="".join(stream_chunks),
                        role=Role.agent,
                        contextId=state.session_id,
                        messageId=str(uuid.uuid4())
                    )

                step.message_stream = _iter()
                yield step
                message_content = "".join(stream_chunks)
            else:
                message_content = ""
                if hasattr(response, "message") and response.message:
                    # If Message.parts is now list[Part], we can use get_text_content
                    message_content = response.message.get_text_content()
                elif hasattr(response, "content") and response.content:
                    message_content = str(response.content)

            parsed_response = self._parse_structured_response(message_content) # Pass message_content

            # Emit thinking step if enabled and thinking is present
            if self.thinking_enabled and parsed_response.get("thinking"):
                yield ThinkStep(
                    step_id=str(uuid.uuid4()),
                    thinking=parsed_response["thinking"],
                    created_at=datetime.now(),
                )
                step_count += 1

            # Emit tool call steps if any
            tool_calls = parsed_response.get("tool_calls", [])
            if tool_calls:
                for tool_call in tool_calls:
                    yield ToolCallStep(
                        step_id=new_id("tool_call", uuid=True),
                        tool_call=tool_call,
                        created_at=datetime.now(),
                    )
                    step_count += 1

                    # Respect max tool calls limit
                    if step_count >= self.max_tool_calls_per_turn:
                        break

            # Emit message step if there's a message
            llm_message_text = parsed_response.get("message", "") # Use 'message' from parsed_response
            if llm_message_text and not (response.streaming and self.streaming_enabled):
                message = Message.from_text( # Updated
                    text=llm_message_text,
                    role=Role.agent,
                    contextId=state.session_id,
                    messageId=str(uuid.uuid4())
                )

                yield MessageStep(
                    step_id=str(uuid.uuid4()),
                    message=message,
                    is_streaming=False,
                    created_at=datetime.now(),
                    metadata={"turn": state.turn_count},
                )
                step_count += 1

            # Check if task should finish
            should_continue = parsed_response.get("should_continue", True)
            if not should_continue or not tool_calls:
                # Task is complete
                final_message = None
                # Use llm_message_text which is parsed_response.get("message", "")
                if llm_message_text and not (response.streaming and self.streaming_enabled):
                    final_message = Message.from_text( # Updated
                        text=llm_message_text, # Use the already extracted message
                        role=Role.agent,
                        contextId=state.session_id,
                        messageId=str(uuid.uuid4())
                    )
                elif not llm_message_text and (response.streaming and self.streaming_enabled and message_content):
                    # If it was a streaming response and no specific message in parsed_response, use full stream content
                    final_message = Message.from_text(
                        text=message_content,
                        role=Role.agent,
                        contextId=state.session_id,
                        messageId=str(uuid.uuid4())
                    )


                yield FinishStep(
                    step_id=str(uuid.uuid4()),
                    final_message=final_message,
                    reason="Task completed based on LLM decision",
                    created_at=datetime.now(),
                    metadata={
                        "turn": state.turn_count,
                        "should_continue": should_continue,
                        "had_tool_calls": len(tool_calls) > 0,
                    },
                )

        except Exception as e:
            logger.error(f"Error in planner execution: {e}", exc_info=True)

            error_message = Message.from_text( # Updated
                text=f"I encountered an error while planning: {e!s}",
                role=Role.agent,
                contextId=state.session_id,
                messageId=str(uuid.uuid4())
            )

            yield ErrorStep(
                step_id=str(uuid.uuid4()),
                message=error_message,
                error=str(e),
                created_at=datetime.now(),
            )

            # Finish with error
            yield FinishStep(
                step_id=str(uuid.uuid4()),
                final_message=error_message,
                reason=f"Planning error: {e!s}",
                created_at=datetime.now(),
                metadata={"turn": state.turn_count, "error": str(e)},
            )

    async def _build_prompt_variables(self, state: State) -> dict[str, _t.Any]:
        """Build variables for the prompt from current state."""
        # Get tool definitions
        tool_definitions_objects = to_tool_definitions(state.tools)
        tool_definitions = [
            {
                "name": td.name,
                "description": td.description,
                "parameters": self._clean_parameters(td.parameters),
            }
            for td in tool_definitions_objects
        ]

        # Format conversation history
        history_messages = []
        # Iterate up to the potential last user message
        history_to_format = state.history
        if state.history and state.history[-1].role == Role.user:
            history_to_format = state.history[:-1]

        for msg in history_to_format:
            history_messages.append(
                {
                    "role": "user" if msg.role == Role.user else "assistant",
                    "content": msg.get_text_content(), # Updated
                }
            )

        # Get current user input (last message if it's from user)
        current_input = ""
        if state.history and state.history[-1].role == Role.user:
            current_input = state.history[-1].get_text_content() # Updated

        # Format tool results from previous turn if any
        tool_results_text = ""
        if state.last_tool_results:
            tool_results_text = format_tool_results_for_conversation(state.last_tool_results)

        return {
            "agent_name": "BasePlanner",
            "task_id": state.session_id, # Using session_id as task_id for planner context
            "turn_count": state.turn_count,
            "has_tools": len(tool_definitions) > 0,
            "tools": tool_definitions,
            "has_history": len(history_messages) > 0,
            "history_text": format_history_for_prompt(history_messages),
            "user_input": current_input if current_input else None,
            "tool_results": bool(tool_results_text),
            "tool_results_text": tool_results_text,
            "task_list": state.task_list,
            "rules": self.rules,
        }

    def _parse_structured_response(self, content: str) -> dict[str, _t.Any]: # Added content type
        """Parse the structured response from prompt_run, adapted from CodeAgent."""
        # Initialize default response structure
        parsed_response = {
            "thinking": "",
            "task_list": {"completed": [], "pending": []},
            "tool_calls": [],
            "message": "",
            "should_continue": True,
            "raw_content": content,
        }

        # Try to extract JSON from markdown code blocks first
        json_pattern = r"```json\s*(\{.*?\})\s*```"
        json_matches = re.findall(json_pattern, content, re.DOTALL)

        response_data = None
        if json_matches:
            # Use the first JSON block found
            try:
                response_data = json.loads(json_matches[0])
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from markdown block: {e}")

        # If no JSON block found, try parsing the entire content as JSON
        if not response_data:
            try:
                response_data = json.loads(content.strip())
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract information from text
                logger.debug("No valid JSON found, extracting from text content")
                parsed_response["message"] = content # Assign full content as message
                parsed_response["tool_calls"] = self._parse_tool_calls_from_text(content)
                # If there are tool calls, we assume it should continue unless message implies otherwise
                parsed_response["should_continue"] = not bool(parsed_response["tool_calls"])
                return parsed_response

        # Extract fields from parsed JSON
        if response_data and isinstance(response_data, dict):
            parsed_response["thinking"] = response_data.get("thinking", "")
            parsed_response["message"] = response_data.get("message", "")
            parsed_response["should_continue"] = response_data.get("should_continue", True)

            # Extract task list
            task_list = response_data.get("task_list", {})
            if isinstance(task_list, dict):
                parsed_response["task_list"]["completed"] = task_list.get("completed", [])
                parsed_response["task_list"]["pending"] = task_list.get("pending", [])

            # Extract tool calls and convert to ToolCall objects
            tool_calls_data = response_data.get("tool_calls", [])
            if isinstance(tool_calls_data, list):
                for call_data in tool_calls_data:
                    if isinstance(call_data, dict) and "name" in call_data:
                        tool_call = ToolCall(
                            call_id=str(uuid.uuid4()),
                            name=call_data["name"],
                            arguments=call_data.get("arguments", {}),
                        )
                        parsed_response["tool_calls"].append(tool_call)
            # If tool_calls are present, default should_continue to True unless explicitly set false
            if parsed_response["tool_calls"] and "should_continue" not in response_data:
                parsed_response["should_continue"] = True


        return parsed_response

    def _parse_tool_calls_from_text(self, content: str) -> list[ToolCall]:
        """Parse tool calls from text content when JSON parsing fails."""
        tool_calls = []

        # Pattern for function call tags
        pattern = r'<function_call name="([^"]+)">\s*(\{.*?\})\s*</function_call>'
        matches = re.findall(pattern, content, re.DOTALL)

        for tool_name, args_str in matches:
            try:
                arguments = json.loads(args_str)
                tool_call = ToolCall(call_id=str(uuid.uuid4()), name=tool_name, arguments=arguments)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse arguments for tool {tool_name}: {args_str}")

        return tool_calls

    def _clean_parameters(self, parameters: dict) -> dict:
        """Clean parameters to remove Undefined values and make them JSON serializable."""

        def clean_value(value):
            """Recursively clean a value to make it JSON serializable."""
            if hasattr(value, "__class__") and value.__class__.__name__ == "Undefined":
                return None
            if isinstance(value, dict):
                return {
                    k: clean_value(v)
                    for k, v in value.items()
                    if not (hasattr(v, "__class__") and v.__class__.__name__ == "Undefined")
                }
            if isinstance(value, list | tuple):
                return [
                    clean_value(item)
                    for item in value
                    if not (hasattr(item, "__class__") and item.__class__.__name__ == "Undefined")
                ]
            if isinstance(value, str | int | float | bool | type(None)):
                return value
            # Try to convert to string for other types
            try:
                json.dumps(value)  # Test if it's JSON serializable
                return value
            except (TypeError, ValueError):
                return str(value)

        return clean_value(parameters)

    def get_config(self) -> dict[str, _t.Any]:
        """Get planner configuration."""
        return {
            "prompt_name": self.prompt_name,
            "max_tool_calls_per_turn": self.max_tool_calls_per_turn,
            "thinking_enabled": self.thinking_enabled,
            "streaming_enabled": self.streaming_enabled,
            "rules": self.rules,
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    async def reset(self, state: State) -> None:
        """Reset the planner to the initial state."""
        # For BasePlanner, resetting means clearing any internal state
        # Since BasePlanner is stateless (each call to next() is independent),
        # we don't need to do anything specific here
        logger.debug(f"BasePlanner reset for session {state.session_id}")

        # If we had any internal state to clear, we would do it here
        # For example, clearing conversation history, resetting counters, etc.
