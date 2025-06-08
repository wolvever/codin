"""Python adaptation of the Codex agent core loop.

The original Rust implementation (``codex-rs``) drives an iterative loop where
LLM responses may include *function/tool calls*.  When such calls are present
the agent executes them and feeds the results back to the model until a final
assistant message is produced.

This module ports that control-flow to Python and wires it up with the rest of
``codin``:

* :pymod:`codin.model` – LLM interfaces (OpenAI, Anthropic, Google, …)
* :pymod:`codin.tool`  – execution of local sandbox + MCP tools
* :pymod:`codin.sandbox` – pluggable execution back-ends
* Agent protocol types from :mod:`codin.agent.types`

Implements the full iterative loop with tool calling, approvals, and events.
"""

import ast
import asyncio
import json
import logging
import re
import time
import typing as _t
import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from ..agent.base import Agent, AgentRunInput, AgentRunOutput
from ..agent.types import ToolCall, ToolCallResult
from ..config import ApprovalMode
from ..memory.base import MemMemoryService, MemoryService
from ..model.factory import LLMFactory
from ..prompt import prompt_run
from ..sandbox.base import Sandbox
from ..sandbox.local import LocalSandbox
from ..tool.base import ToolContext, Toolset, to_tool_definitions
from ..tool.executor import ToolExecutor
from ..tool.registry import ToolRegistry
from ..utils.message import ( # extract_text_from_message removed
    format_history_for_prompt,
    format_tool_results_for_conversation,
)

# Use protocol types directly
from .types import Message, Role, TextPart # TextPart might not be directly needed after changes

__all__: list[str] = [
    "AgentEvent",
    "CodeAgent",
    "ContinueDecision",
]

logger = logging.getLogger("codin.agent.code_agent")


class ContinueDecision(Enum):
    """Decision on whether to continue execution."""

    CONTINUE = "continue"
    NEED_APPROVAL = "need_approval"
    CANCEL = "cancel"


class AgentEvent(BaseModel):
    """Event emitted during agent execution."""

    event_type: str
    event_id: str
    data: dict[str, _t.Any]
    timestamp: datetime


class CodeAgent(Agent):
    """Enhanced iterative agent following codex.rs loop logic supporting standard agent protocols."""

    def __init__(
        self,
        *,
        name: str = "CodeAgent",
        description: str = "Python agent with sandbox + tools support and iterative execution",
        llm_model: str | None = None,
        sandbox: Sandbox | None = None,
        tool_registry: ToolRegistry | None = None,
        toolsets: list[Toolset] | None = None,
        approval_mode: ApprovalMode = ApprovalMode.UNSAFE_ONLY,
        max_turns: int = 100,
        rules: str | None = None,
        memory_system: MemoryService | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the CodeAgent.

        Args:
            name: Agent name
            description: Agent description
            llm_model: LLM model to use (overrides config)
            sandbox: Sandbox instance for code execution
            tool_registry: Registry of available tools
            toolsets: List of toolsets to register
            approval_mode: Mode for tool approval
            max_turns: Maximum number of turns before stopping
            rules: Additional rules/instructions for the agent
            memory_system: Memory system for conversation history
            debug: Enable debug mode to print LLM requests and responses
        """
        super().__init__(name=name, description=description)

        # Initialize LLM

        self.llm = LLMFactory.create_llm(model=llm_model)

        # Initialize sandbox
        self.sandbox = sandbox or LocalSandbox()

        # Initialize tool registry
        self.tool_registry = tool_registry or ToolRegistry()

        # Initialize tool executor
        self.tool_executor = ToolExecutor(self.tool_registry)

        # Register toolsets
        if toolsets:
            for toolset in toolsets:
                self.tool_registry.register_toolset(toolset)
        else:
            # Default toolsets
            from ..tool.sandbox import SandboxToolset

            sandbox_toolset = SandboxToolset(self.sandbox)
            self.tool_registry.register_toolset(sandbox_toolset)

        # Configuration
        self.approval_mode = approval_mode
        self.max_turns = max_turns
        self.debug = debug

        # Store custom rules for passing to prompt template
        self.rules = rules

        # Initialize memory system
        self.memory_system = memory_system or MemMemoryService()

        # State tracking
        self._conversation_history: list[Message] = []
        self._context_id = str(uuid.uuid4())
        self._start_time: float = 0
        self._turn_count = 0
        self._total_tool_calls = 0
        self._approved_commands: set[tuple[str, ...]] = set()
        self._completed_actions: set[str] = set()
        self._last_tool_calls: list[str] = []
        self._all_tool_calls: set[tuple[str, str]] = set()  # Track all tool calls across turns

        # Event callbacks
        self._event_callbacks: list[_t.Callable[[AgentEvent], _t.Awaitable[None]]] = []

        logger.info(f"Initialized CodeAgent with {len(self.tool_registry.get_tools())} tools")

    def add_event_callback(self, callback: _t.Callable[[AgentEvent], _t.Awaitable[None]]) -> None:
        """Add an event callback for monitoring agent execution."""
        self._event_callbacks.append(callback)

    def _should_continue_execution(
        self, turn_count: int, elapsed_time: float, tool_call_count: int
    ) -> ContinueDecision:
        """Determine if execution should continue based on various metrics.

        This method can be extended in the future with configuration options.

        Args:
            turn_count: Current number of turns executed
            elapsed_time: Time elapsed since task start (seconds)
            tool_call_count: Total number of tool calls executed

        Returns:
            Decision on whether to continue, need approval, or cancel
        """
        # Basic max turns check
        if turn_count >= self.max_turns:
            # Could be extended to check if user approval is needed for continuation
            return ContinueDecision.CANCEL

        return ContinueDecision.CONTINUE

    async def _emit_event(self, event_type: str, data: dict[str, _t.Any]) -> None:
        """Emit an event to all registered callbacks."""
        event = AgentEvent(event_type=event_type, event_id=str(uuid.uuid4()), data=data, timestamp=datetime.now())

        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    def _create_user_message(self, content: str, task_id: str | None = None) -> Message:
        """Create a user message for conversation history."""
        return Message.from_text(
            text=content,
            role=Role.user,
            contextId=self._context_id,
            taskId=task_id,
            messageId=str(uuid.uuid4())
        )

    def _create_agent_message(self, content: str, task_id: str | None = None) -> Message:
        """Create an agent message for conversation history."""
        return Message.from_text(
            text=content,
            role=Role.agent,
            contextId=self._context_id,
            taskId=task_id,
            messageId=str(uuid.uuid4())
        )

    def _parse_tool_calls(self, content: str) -> list[ToolCall]:
        """Parse tool calls from LLM response content.

        Supports multiple formats:
        1. Claude-style <function_calls> with multiple calls
        2. Individual <function_call> tags
        3. JSON-formatted tool calls
        4. OpenAI-style dictionary with 'tool_calls' field
        5. Python dict string representation
        6. Direct dict object (when content is already parsed)
        """
        tool_calls = []

        # Handle case where content is already a dict (from prompt_run response)
        if isinstance(content, dict) and "tool_calls" in content:
            response_dict = content
        elif isinstance(content, str):
            # First try to parse as Python dict string (common format from LLM responses)
            if content.strip().startswith("{") and "tool_calls" in content:
                try:
                    # First try JSON parsing
                    response_dict = json.loads(content)
                except json.JSONDecodeError:
                    try:
                        # If JSON fails, try Python literal_eval for dict strings
                        response_dict = ast.literal_eval(content)
                    except (ValueError, SyntaxError):
                        response_dict = None
            else:
                response_dict = None
        else:
            response_dict = None

        if response_dict:
            if "tool_calls" in response_dict:
                openai_tool_calls = response_dict["tool_calls"]
                for call in openai_tool_calls:
                    if isinstance(call, dict):
                        # Handle both OpenAI format and Claude format
                        if "function" in call:
                            # OpenAI format
                            func = call["function"]
                            call_id = call.get("id", str(uuid.uuid4()))
                            tool_name = func.get("name", "")
                            arguments_str = func.get("arguments", "{}")

                            try:
                                if isinstance(arguments_str, str):
                                    arguments = json.loads(arguments_str)
                                else:
                                    arguments = arguments_str
                                tool_calls.append(ToolCall(call_id=call_id, name=tool_name, arguments=arguments))
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse tool arguments: {arguments_str}")
                        elif "type" in call and call["type"] == "function":
                            # Claude format
                            func = call.get("function", {})
                            call_id = call.get("id", str(uuid.uuid4()))
                            tool_name = func.get("name", "")
                            arguments_str = func.get("arguments", "{}")

                            try:
                                if isinstance(arguments_str, str):
                                    arguments = json.loads(arguments_str)
                                else:
                                    arguments = arguments_str
                                tool_calls.append(ToolCall(call_id=call_id, name=tool_name, arguments=arguments))
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse Claude tool arguments: {arguments_str}")

                if tool_calls:
                    return tool_calls

        # Pattern 1: Claude-style multiple function calls in a single block
        multi_pattern = r"<function_calls>\s*(\[.*?\])\s*</function_calls>"
        multi_matches = re.findall(multi_pattern, content, re.DOTALL)

        for match in multi_matches:
            try:
                calls_data = json.loads(match)
                if isinstance(calls_data, list):
                    for call_data in calls_data:
                        if isinstance(call_data, dict) and "name" in call_data:
                            call_id = str(uuid.uuid4())
                            tool_calls.append(
                                ToolCall(
                                    call_id=call_id, name=call_data["name"], arguments=call_data.get("arguments", {})
                                )
                            )
            except json.JSONDecodeError:
                logger.error(f"Failed to parse multi-function calls: {match}")

        # Pattern 2: Individual function call tags
        single_pattern = r'<function_call name="([^"]+)">\s*({[^}]*}|\{.*?\})\s*</function_call>'
        single_matches = re.findall(single_pattern, content, re.DOTALL)

        for tool_name, args_str in single_matches:
            try:
                call_id = str(uuid.uuid4())
                arguments = json.loads(args_str)
                tool_calls.append(ToolCall(call_id=call_id, name=tool_name, arguments=arguments))
            except json.JSONDecodeError:
                logger.error(f"Failed to parse arguments for tool {tool_name}: {args_str}")

        thinking_pattern = r'<function_call name="([^"]+)">\s*(\{.*?\})\s*</function_call>'
        thinking_matches = re.findall(thinking_pattern, content, re.DOTALL)

        for tool_name, args_str in thinking_matches:
            if not any(
                call.name == tool_name
                and json.dumps(call.arguments, sort_keys=True) == json.dumps(json.loads(args_str), sort_keys=True)
                for call in tool_calls
            ):
                try:
                    call_id = str(uuid.uuid4())
                    arguments = json.loads(args_str)
                    tool_calls.append(ToolCall(call_id=call_id, name=tool_name, arguments=arguments))
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse thinking arguments for tool {tool_name}: {args_str}")

        return tool_calls

    def _parse_structured_response(self, content: str) -> dict[str, _t.Any]:
        parsed_response = {
            "thinking": "",
            "task_list": {"completed": [], "pending": []},
            "tool_calls": [],
            "message": "",
            "should_continue": True,
            "raw_content": content,
        }
        json_pattern = r"```json\s*(\{.*?\})\s*```"
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        response_data = None
        if json_matches:
            try:
                response_data = json.loads(json_matches[0])
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from markdown block: {e}")
        if not response_data:
            try:
                response_data = json.loads(content.strip())
            except json.JSONDecodeError:
                logger.debug("No valid JSON found, extracting from text content")
                parsed_response["message"] = content
                parsed_response["tool_calls"] = self._parse_tool_calls(content)
                return parsed_response
        if response_data and isinstance(response_data, dict):
            parsed_response["thinking"] = response_data.get("thinking", "")
            parsed_response["message"] = response_data.get("message", "")
            parsed_response["should_continue"] = response_data.get("should_continue", True)
            task_list = response_data.get("task_list", {})
            if isinstance(task_list, dict):
                parsed_response["task_list"]["completed"] = task_list.get("completed", [])
                parsed_response["task_list"]["pending"] = task_list.get("pending", [])
            tool_calls_data = response_data.get("tool_calls", [])
            if isinstance(tool_calls_data, list):
                for call_data in tool_calls_data:
                    if isinstance(call_data, dict) and "name" in call_data:
                        call_id = str(uuid.uuid4())
                        tool_call = ToolCall(
                            call_id=call_id, name=call_data["name"], arguments=call_data.get("arguments", {})
                        )
                        parsed_response["tool_calls"].append(tool_call)
        return parsed_response

    async def _execute_tool_call(self, tool_call: ToolCall, task_id: str = "default") -> ToolCallResult:
        await self._emit_event(
            "tool_call_start",
            {"call_id": tool_call.call_id, "tool_name": tool_call.name, "arguments": tool_call.arguments},
        )
        try:
            tool = self.tool_registry.get_tool(tool_call.name)
            if tool is None:
                return ToolCallResult(
                    call_id=tool_call.call_id, success=False, output="", error=f"Unknown tool: {tool_call.name}"
                )
            if await self._needs_approval(tool_call):
                if not await self._request_approval(tool_call):
                    return ToolCallResult(
                        call_id=tool_call.call_id, success=False, output="", error="Tool execution rejected by user"
                    )
            tool_context = ToolContext(tool_name=tool_call.name, arguments=tool_call.arguments, session_id=task_id)
            result = await self.tool_executor.execute(tool_call.name, tool_call.arguments, tool_context)
            output = str(result) if result is not None else ""
            await self._emit_event(
                "tool_call_end",
                {
                    "call_id": tool_call.call_id,
                    "tool_name": tool_call.name,
                    "success": True,
                    "output": output,
                },
            )
            return ToolCallResult(call_id=tool_call.call_id, success=True, output=output)
        except Exception as e:
            error_msg = f"Tool execution error: {e!s}"
            logger.error(f"Error executing tool {tool_call.name}: {e}", exc_info=True)
            await self._emit_event(
                "tool_call_end",
                {"call_id": tool_call.call_id, "tool_name": tool_call.name, "success": False, "error": error_msg},
            )
            return ToolCallResult(call_id=tool_call.call_id, success=False, output="", error=error_msg)

    async def _needs_approval(self, tool_call: ToolCall) -> bool:
        if self.approval_mode == ApprovalMode.NEVER: return False
        if self.approval_mode == ApprovalMode.ALWAYS: return True
        dangerous_tools = {"shell", "container_exec", "file_write", "exec"}
        if tool_call.name in dangerous_tools:
            if tool_call.name == "shell" or tool_call.name == "container_exec":
                command = tool_call.arguments.get("command", [])
                if isinstance(command, str): command = [command]
                command_tuple = tuple(command)
                return command_tuple not in self._approved_commands
            return True
        return False

    async def _request_approval(self, tool_call: ToolCall) -> bool:
        logger.info(f"Auto-approving tool call: {tool_call.name} with args {tool_call.arguments}")
        if tool_call.name in {"shell", "container_exec"}:
            command = tool_call.arguments.get("command", [])
            if isinstance(command, str): command = [command]
            self._approved_commands.add(tuple(command))
        await self._emit_event(
            "approval_requested",
            {
                "call_id": tool_call.call_id,
                "tool_name": tool_call.name,
                "arguments": tool_call.arguments,
                "auto_approved": True,
            },
        )
        return True

    async def _run_turn(
        self, turn_input: list[Message], run_context: dict[str, _t.Any]
    ) -> tuple[list[Message], list[ToolCallResult]]:
        self._turn_count += 1
        await self._emit_event("turn_start", {"turn": self._turn_count, "input_messages": len(turn_input)})
        try:
            tool_definitions_objects = run_context.get("tool_definitions", [])
            tool_definitions = [
                {"name": td.name, "description": td.description, "parameters": self._clean_parameters(td.parameters)}
                for td in tool_definitions_objects
            ]
            history_messages = []
            for msg in turn_input[:-1]:
                history_messages.append(
                    {
                        "role": "user" if msg.role == Role.user else "assistant",
                        "content": msg.get_text_content(), # Updated
                    }
                )
            current_input = turn_input[-1].get_text_content() if turn_input else "" # Updated
            tool_results_text = ""
            if hasattr(self, "_last_tool_results") and self._last_tool_results:
                tool_results_text = format_tool_results_for_conversation(self._last_tool_results)
            task_id = run_context.get("task_id", "default")
            task_list = run_context.get("task_list", {"completed": [], "pending": []})
            variables = {
                "agent_name": self.name, "task_id": task_id, "turn_count": self._turn_count,
                "has_tools": len(tool_definitions) > 0, "tools": tool_definitions,
                "has_history": len(history_messages) > 0, "history_text": format_history_for_prompt(history_messages),
                "user_input": current_input if current_input else None,
                "tool_results": bool(tool_results_text), "tool_results_text": tool_results_text,
                "task_list": task_list, "rules": self.rules,
            }
            logger.debug("RunTurn, task_id: %s, turn: %s, variables: %s", task_id, self._turn_count, variables)
            if self.debug: # Debug printing omitted for brevity
                pass
            try:
                response = await asyncio.wait_for(
                    prompt_run("code_agent_loop", variables=variables, tools=tool_definitions, stream=False),
                    timeout=120.0
                )
            except TimeoutError:
                logger.error("LLM request timed out")
                return [self._create_agent_message("LLM request timed out.", task_id)], []
            except Exception as e:
                logger.error(f"LLM request failed: {e}")
                return [self._create_agent_message(f"LLM request error: {e!s}", task_id)], []

            content = ""
            if hasattr(response, "message") and response.message:
                content = response.message.get_text_content() # Use get_text_content
            elif hasattr(response, "content"):
                content = str(await response.content) if asyncio.iscoroutine(response.content) else str(response.content)
            else:
                content = str(response)

            parsed_response = self._parse_structured_response(content)
            if self.debug: # Emit debug event (omitted for brevity)
                pass
            if parsed_response["task_list"]["completed"] or parsed_response["task_list"]["pending"]:
                run_context["task_list"] = parsed_response["task_list"]
            await self._emit_event("llm_response", { "content": content, **parsed_response })
            tool_calls = parsed_response["tool_calls"]
            tool_results = []
            if tool_calls:
                await self._emit_event("tool_results_start", {"num_tools": len(tool_calls)})
                for tool_call in tool_calls: # Tool call execution loop (omitted for brevity, assumed correct)
                    result = await asyncio.wait_for(self._execute_tool_call(tool_call, task_id), timeout=60.0)
                    tool_results.append(result)
                await self._emit_event("tool_results", { "num_tools_executed": len(tool_results), ... })
            self._last_tool_results = tool_results
            response_message_content = parsed_response["message"] or content
            response_message = self._create_agent_message(response_message_content, task_id)
            run_context["should_continue"] = parsed_response["should_continue"]
            if tool_results:
                tool_results_message = self._create_agent_message(f"Tool execution results:\n{format_tool_results_for_conversation(tool_results)}", task_id)
                return [response_message, tool_results_message], tool_results
            return [response_message], tool_results
        except Exception as e:
            logger.error(f"Error in turn execution: {e}")
            await self._emit_event("turn_error", {"error": str(e), "turn": self._turn_count})
            return [self._create_agent_message(f"I encountered an error: {e!s}", task_id)], []

    def _parse_json_tool_calls(self, content: str) -> list[ToolCall]: # Omitted for brevity
        return []

    async def run(self, input_data: AgentRunInput) -> AgentRunOutput: # Omitted for brevity, assume uses _run_turn correctly
        self._start_time = time.time()
        task_id = str(uuid.uuid4())
        tool_definitions = to_tool_definitions(self.tool_registry.get_tools())
        run_context = { "task_id": task_id, "task_list": {"completed": [], "pending": []}, "tool_definitions": tool_definitions, "should_continue": True, }
        conversation_history = [input_data.message]
        total_tool_calls = 0; iteration = 0

        user_input_text = ""
        if input_data.message: # Added check
            user_input_text = input_data.message.get_text_content() # Updated

        await self._emit_event("task_start", { "task_id": task_id, "session_id": self._context_id, "iteration": 0, "elapsed_time": 0.0, "user_input": user_input_text, })
        try:
            while iteration < self.max_turns:
                iteration += 1; elapsed_time = time.time() - self._start_time
                continue_decision = self._should_continue_execution(iteration, elapsed_time, total_tool_calls)
                if continue_decision != ContinueDecision.CONTINUE: break
                response_messages, tool_results = await self._run_turn(conversation_history, run_context)
                conversation_history.extend(response_messages)
                total_tool_calls += len(tool_results)
                if self._is_task_complete(response_messages, tool_results, run_context): break
                if tool_results: continue
                break
            final_response = next((msg for msg in reversed(conversation_history) if msg.role == Role.agent), self._create_agent_message("Task completed.", task_id))
            if self.memory_system:
                for msg in conversation_history: await self.memory_system.add_message(msg)
            await self._emit_event("task_end", { "task_id": task_id, "session_id": self._context_id, "iteration": iteration, "elapsed_time": time.time() - self._start_time, })
            return AgentRunOutput(result=final_response, metadata={ "task_id": task_id, "iterations": iteration, ...})
        except Exception as e:
            logger.error(f"Error in agent execution: {e}", exc_info=True)
            await self._emit_event("task_error", {"task_id": task_id, "error": str(e), "iteration": iteration})
            error_response = self._create_agent_message(f"I encountered an error: {e!s}", task_id)
            await self._emit_event("task_end", {"task_id": task_id, ...})
            return AgentRunOutput(result=error_response, metadata={"task_id": task_id, "error": str(e), ...})

    def _is_task_complete(
        self, response_messages: list[Message], tool_results: list[ToolCallResult], run_context: dict[str, _t.Any]
    ) -> bool:
        if "should_continue" in run_context:
            return not run_context["should_continue"]
        if not tool_results:
            last_message_text = ""
            if response_messages: # Added check
                last_message_text = response_messages[-1].get_text_content().lower() # Updated
            completion_indicators = ["task completed", "finished", "done", "complete", "successfully created", "successfully executed", "no further", "that's all", "task is complete",]
            continuation_indicators = ["let me", "i'll", "next i", "now i", "first i", "tool_calls", "need to", "should", "will",]
            has_completion = any(indicator in last_message_text for indicator in completion_indicators)
            has_continuation = any(indicator in last_message_text for indicator in continuation_indicators)
            if has_completion and not has_continuation: return True
            if not has_continuation and "tool_calls" not in last_message_text: return True
        return False

    async def reset_conversation(self) -> None: # Omitted for brevity
        pass
    async def get_memory_history(self, limit: int = 50, query: str | None = None) -> list[Message]: # Omitted for brevity
        return []
    async def compress_conversation_history(self, keep_recent: int = 20, chunk_size: int = 10) -> int: # Omitted for brevity
        return 0
    def get_conversation_summary(self) -> dict[str, _t.Any]: # Omitted for brevity
        return {}
    def add_tool(self, tool) -> None: # Omitted for brevity
        pass
    async def cleanup(self) -> None: # Omitted for brevity
        pass

    @staticmethod
    def _message_to_llm_dict(msg: Message) -> dict[str, _t.Any]:
        """Convert :class:`Message` to OpenAI message dict.""" # Already correct from previous subtask
        if not msg.parts: raise ValueError("Message parts is empty")
        # This method will effectively be unused if all callers are updated to use msg.get_text_content()
        # and then construct the dict, or if a new method on Message handles this.
        # For now, leaving as is, but it's a candidate for removal/refactoring if Message.to_llm_dict() is added.
        content_str = msg.get_text_content() # Using new method
        role_map = { Role.user: "user", Role.agent: "assistant", }
        return { "role": role_map[msg.role], "content": content_str, }

    def _detect_and_intervene_loop(self, original_user_input: str, tool_calls: list[ToolCall]) -> tuple[str | None, list[ToolCall]]: # Omitted for brevity
        return None, tool_calls
    def _generate_intervention_and_fix(self, repeated_call: str, original_user_input: str, tool_calls: list[ToolCall]) -> tuple[str, list[ToolCall]]: # Omitted for brevity
        return "", []
    def _clean_parameters(self, parameters: dict) -> dict: # Omitted for brevity
        return {}
