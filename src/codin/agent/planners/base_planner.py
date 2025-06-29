"""CodeAct Planner - specialized for code generation and execution tasks."""

import json
import logging
import re
import typing as _t
import uuid
from datetime import datetime

try:  # Support import as both 'codin' and 'src.codin'
    from ...id import new_id
except Exception:  # pragma: no cover - fallback for regular package
    from ..id import new_id

try:
    from ..prompt.run import prompt_run
except Exception:  # pragma: no cover - fallback when namespaced under src
    from ...prompt.run import prompt_run

try:
    from ..sandbox.local import LocalSandbox
except Exception:  # pragma: no cover - fallback for namespaced import
    from ...sandbox.local import LocalSandbox

try:
    from ..tool.base import to_definitions as to_tool_definitions
except Exception:  # pragma: no cover
    from ...tool.base import to_definitions as to_tool_definitions

try:
    from ..utils.message import format_history_for_prompt, format_tool_results_for_conversation
except Exception:  # pragma: no cover
    from ...utils.message import (
        format_history_for_prompt,
        format_tool_results_for_conversation,
    )
try:
    from ..base import Planner
except Exception:  # pragma: no cover
    from .base import Planner
from ..types import (
    ErrorStep,
    FinishStep,
    Message,
    MessageStep,
    Role,
    State,
    Step,
    ThinkStep,
    ToolCall,
    ToolCallStep,
)

__all__ = ["CodeActPlanner"]

logger = logging.getLogger("codin.agent.codeact_planner")


class CodeActPlanner(Planner):
    """
    CodeAct Planner specialized for code generation and execution.

    This planner focuses on:
    - Code generation from natural language
    - Automatic code execution
    - Iterative refinement based on execution results
    - Integration with sandbox environments
    """

    _CODE_RE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)
    _JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

    def __init__(
        self,
        *,
        prompt_name: str = "code_agent_basic",
        sandbox: LocalSandbox | None = None,
        max_iterations: int = 5,
        auto_execute: bool = True,
        enable_thinking: bool = True,
        enable_streaming: bool = True,
        language: str = "python",
        rules: str | None = None,
    ) -> None:
        """Initialize the CodeAct Planner."""
        self.prompt_name = prompt_name
        self.sandbox = sandbox or LocalSandbox()
        self.max_iterations = max_iterations
        self.auto_execute = auto_execute
        self.thinking_enabled = enable_thinking
        self.streaming_enabled = enable_streaming
        self.language = language
        self.rules = rules
        self._iteration_count = 0

        logger.info(f"Initialized CodeAct Planner with {language} support")

    async def next(self, state: State) -> _t.AsyncGenerator[Step]:
        """Generate execution steps with code generation and execution."""
        try:
            self._iteration_count = 0

            while self._iteration_count < self.max_iterations:
                self._iteration_count += 1

                # Get LLM response
                variables = await self._build_prompt_variables(state)
                response = await prompt_run(
                    self.prompt_name,
                    variables=variables,
                    tools=variables.get("tools", []),
                    stream=self.streaming_enabled,
                )

                # Handle streaming response
                message_content = await self._handle_response(response, state)

                # Parse structured response
                parsed_response = self._parse_response(message_content)

                # Emit thinking step if enabled
                if self.thinking_enabled and parsed_response.get("thinking"):
                    yield ThinkStep(
                        step_id=str(uuid.uuid4()),
                        thinking=parsed_response["thinking"],
                        created_at=datetime.now(),
                    )

                # Handle tool calls
                tool_calls = parsed_response.get("tool_calls", [])
                if tool_calls:
                    for tool_call in tool_calls:
                        yield ToolCallStep(
                            step_id=new_id("tool_call", uuid=True),
                            tool_call=tool_call,
                            created_at=datetime.now(),
                        )

                # Extract and execute code blocks
                code_blocks = self._extract_code_blocks(message_content)
                if code_blocks and self.auto_execute:
                    # Emit the message with code
                    code_message = Message.from_text(
                        text=message_content,
                        role=Role.agent,
                        contextId=state.session_id,
                        messageId=new_id("msg"),
                    )
                    yield MessageStep(
                        step_id=str(uuid.uuid4()),
                        message=code_message,
                        created_at=datetime.now(),
                    )

                    # Execute each code block
                    for i, code_block in enumerate(code_blocks):
                        try:
                            execution_result = await self._execute_code(code_block)

                            # Emit execution result
                            result_message = Message.from_text(
                                text=f"Code execution result:\n```\n{execution_result}\n```",
                                role=Role.agent,
                                contextId=state.session_id,
                                messageId=new_id("msg"),
                            )
                            yield MessageStep(
                                step_id=str(uuid.uuid4()),
                                message=result_message,
                                created_at=datetime.now(),
                                metadata={"execution_result": True, "block_index": i},
                            )

                            # Add execution result to state for next iteration
                            state.last_tool_results = [
                                {
                                    "tool_name": "code_execution",
                                    "result": execution_result,
                                    "success": True,
                                }
                            ]

                        except Exception as e:
                            error_message = Message.from_text(
                                text=f"Code execution error: {e!s}",
                                role=Role.agent,
                                contextId=state.session_id,
                                messageId=new_id("msg"),
                            )
                            yield ErrorStep(
                                step_id=str(uuid.uuid4()),
                                message=error_message,
                                error=str(e),
                                created_at=datetime.now(),
                            )

                            # Add error to state for next iteration
                            state.last_tool_results = [
                                {
                                    "tool_name": "code_execution",
                                    "result": str(e),
                                    "success": False,
                                }
                            ]

                    # Continue iterating if there were execution results
                    continue

                # Emit regular message if no code or execution
                regular_message = parsed_response.get("message", message_content)
                if regular_message:
                    msg = Message.from_text(
                        text=regular_message,
                        role=Role.agent,
                        contextId=state.session_id,
                        messageId=new_id("msg"),
                    )
                    yield MessageStep(
                        step_id=str(uuid.uuid4()),
                        message=msg,
                        created_at=datetime.now(),
                    )

                # Check if should continue
                should_continue = parsed_response.get("should_continue", False)
                if not should_continue and not tool_calls and not code_blocks:
                    final_message = Message.from_text(
                        text=regular_message or "CodeAct execution complete",
                        role=Role.agent,
                        contextId=state.session_id,
                        messageId=new_id("msg"),
                    )
                    yield FinishStep(
                        step_id=str(uuid.uuid4()),
                        final_message=final_message,
                        reason="CodeAct task completed",
                        created_at=datetime.now(),
                        metadata={"iterations": self._iteration_count},
                    )
                    break

            # Max iterations reached
            if self._iteration_count >= self.max_iterations:
                final_message = Message.from_text(
                    text=f"Reached maximum iterations ({self.max_iterations})",
                    role=Role.agent,
                    contextId=state.session_id,
                    messageId=new_id("msg"),
                )
                yield FinishStep(
                    step_id=str(uuid.uuid4()),
                    final_message=final_message,
                    reason="Max iterations reached",
                    created_at=datetime.now(),
                    metadata={"iterations": self._iteration_count},
                )

        except Exception as e:
            logger.error(f"Error in CodeAct planner: {e}", exc_info=True)
            error_message = Message.from_text(
                text=f"CodeAct planner error: {e!s}",
                role=Role.agent,
                contextId=state.session_id,
                messageId=new_id("msg"),
            )
            yield ErrorStep(
                step_id=str(uuid.uuid4()),
                message=error_message,
                error=str(e),
                created_at=datetime.now(),
            )

    async def _handle_response(self, response: _t.Any, state: State) -> str:
        """Handle LLM response and extract content."""
        if (
            hasattr(response, "streaming")
            and response.streaming
            and self.streaming_enabled
            and isinstance(response.content, _t.AsyncIterator)
        ):
            # Handle streaming response
            chunks = []
            async for chunk in response.content:
                chunks.append(str(chunk))
            return "".join(chunks)
        else:
            # Handle non-streaming response
            if hasattr(response, "message") and response.message:
                return response.message.get_text_content()
            elif hasattr(response, "content"):
                return str(response.content)
            return ""

    def _parse_response(self, content: str) -> dict[str, _t.Any]:
        """Parse response content for structured data."""
        parsed = {
            "thinking": "",
            "message": content,
            "tool_calls": [],
            "should_continue": False,
        }

        # Try to extract JSON structure
        json_matches = self._JSON_RE.findall(content)
        if json_matches:
            try:
                data = json.loads(json_matches[0])
                if isinstance(data, dict):
                    parsed.update(data)
                    return parsed
            except json.JSONDecodeError:
                pass

        # Try to parse as full JSON
        try:
            data = json.loads(content.strip())
            if isinstance(data, dict):
                parsed.update(data)
                return parsed
        except json.JSONDecodeError:
            pass

        # Extract tool calls from text
        parsed["tool_calls"] = self._parse_tool_calls_from_text(content)

        return parsed

    def _parse_tool_calls_from_text(self, content: str) -> list[ToolCall]:
        """Parse tool calls from text content."""
        tool_calls = []
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

    def _extract_code_blocks(self, content: str) -> list[str]:
        """Extract code blocks from content."""
        return self._CODE_RE.findall(content)

    async def _execute_code(self, code: str) -> str:
        """Execute code in sandbox environment."""
        await self.sandbox.up()
        try:
            result = await self.sandbox.run_code(code, language=self.language)
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            return output.strip()
        finally:
            await self.sandbox.down()

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
        history_to_format = state.history
        if state.history and state.history[-1].role == Role.user:
            history_to_format = state.history[:-1]

        for msg in history_to_format:
            history_messages.append(
                {
                    "role": "user" if msg.role == Role.user else "assistant",
                    "content": msg.get_text_content(),
                }
            )

        # Get current user input
        current_input = ""
        if state.history and state.history[-1].role == Role.user:
            current_input = state.history[-1].get_text_content()

        # Format tool results
        tool_results_text = ""
        if state.last_tool_results:
            tool_results_text = format_tool_results_for_conversation(state.last_tool_results)

        return {
            "agent_name": "CodeActPlanner",
            "task_id": state.session_id,
            "turn_count": state.turn_count,
            "iteration_count": self._iteration_count,
            "max_iterations": self.max_iterations,
            "language": self.language,
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

    def _clean_parameters(self, parameters: dict) -> dict:
        """Clean parameters to remove Undefined values and make them JSON serializable."""

        def clean_value(value):
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
            try:
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                return str(value)

        return clean_value(parameters)

    async def reset(self, state: State) -> None:
        """Reset the planner state."""
        self._iteration_count = 0
        logger.debug(f"CodeAct Planner reset for session {state.session_id}")
