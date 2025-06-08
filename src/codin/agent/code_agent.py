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
* A2A protocol types from :pymod:`a2a.types`

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

# Use a2a types directly
from a2a.types import Role
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
from ..utils.message import (
    extract_text_from_message,
    format_history_for_prompt,
    format_tool_results_for_conversation,
)
from .types import Message, TextPart

__all__: list[str] = [
    'AgentEvent',
    'CodeAgent',
    'ContinueDecision',
]

logger = logging.getLogger('codin.agent.code_agent')


class ContinueDecision(Enum):
    """Decision on whether to continue execution."""

    CONTINUE = 'continue'
    NEED_APPROVAL = 'need_approval'
    CANCEL = 'cancel'


class AgentEvent(BaseModel):
    """Event emitted during agent execution."""

    event_type: str
    event_id: str
    data: dict[str, _t.Any]
    timestamp: datetime


class CodeAgent(Agent):
    """Enhanced iterative agent following codex.rs loop logic with A2A protocol support."""

    def __init__(
        self,
        *,
        name: str = 'CodeAgent',
        description: str = 'Python agent with sandbox + tools support and iterative execution',
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

        logger.info(f'Initialized CodeAgent with {len(self.tool_registry.get_tools())} tools')

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

        # Future extensions could include:
        # - Token usage limits
        # - Time-based limits (e.g., > 5 minutes needs approval)
        # - Cost-based limits
        # - Tool call frequency limits
        # - User-defined stopping conditions

        # Example of time-based approval (commented for now):
        # if elapsed_time > 300:  # 5 minutes
        #     return ContinueDecision.NEED_APPROVAL

        # Example of tool call limit (commented for now):
        # if tool_call_count > 50:
        #     return ContinueDecision.NEED_APPROVAL

        return ContinueDecision.CONTINUE

    async def _emit_event(self, event_type: str, data: dict[str, _t.Any]) -> None:
        """Emit an event to all registered callbacks."""
        event = AgentEvent(event_type=event_type, event_id=str(uuid.uuid4()), data=data, timestamp=datetime.now())

        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f'Error in event callback: {e}')

    def _create_user_message(self, content: str, task_id: str | None = None) -> Message:
        """Create a user message for conversation history."""
        return Message(
            messageId=str(uuid.uuid4()),
            role=Role.user,
            parts=[TextPart(text=content)],
            contextId=self._context_id,
            taskId=task_id,
            kind='message',
        )

    def _create_agent_message(self, content: str, task_id: str | None = None) -> Message:
        """Create an agent message for conversation history."""
        return Message(
            messageId=str(uuid.uuid4()),
            role=Role.agent,
            parts=[TextPart(text=content)],
            contextId=self._context_id,
            taskId=task_id,
            kind='message',
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
        if isinstance(content, dict) and 'tool_calls' in content:
            response_dict = content
        elif isinstance(content, str):
            # First try to parse as Python dict string (common format from LLM responses)
            if content.strip().startswith('{') and 'tool_calls' in content:
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
            if 'tool_calls' in response_dict:
                openai_tool_calls = response_dict['tool_calls']
                for call in openai_tool_calls:
                    if isinstance(call, dict):
                        # Handle both OpenAI format and Claude format
                        if 'function' in call:
                            # OpenAI format
                            func = call['function']
                            call_id = call.get('id', str(uuid.uuid4()))
                            tool_name = func.get('name', '')
                            arguments_str = func.get('arguments', '{}')

                            try:
                                if isinstance(arguments_str, str):
                                    arguments = json.loads(arguments_str)
                                else:
                                    arguments = arguments_str
                                tool_calls.append(ToolCall(call_id=call_id, name=tool_name, arguments=arguments))
                            except json.JSONDecodeError:
                                logger.error(f'Failed to parse tool arguments: {arguments_str}')
                        elif 'type' in call and call['type'] == 'function':
                            # Claude format
                            func = call.get('function', {})
                            call_id = call.get('id', str(uuid.uuid4()))
                            tool_name = func.get('name', '')
                            arguments_str = func.get('arguments', '{}')

                            try:
                                if isinstance(arguments_str, str):
                                    arguments = json.loads(arguments_str)
                                else:
                                    arguments = arguments_str
                                tool_calls.append(ToolCall(call_id=call_id, name=tool_name, arguments=arguments))
                            except json.JSONDecodeError:
                                logger.error(f'Failed to parse Claude tool arguments: {arguments_str}')

                if tool_calls:
                    return tool_calls

        # Pattern 1: Claude-style multiple function calls in a single block
        # <function_calls>
        # [{"name": "tool_name", "arguments": {...}}, ...]
        # </function_calls>
        multi_pattern = r'<function_calls>\s*(\[.*?\])\s*</function_calls>'
        multi_matches = re.findall(multi_pattern, content, re.DOTALL)

        for match in multi_matches:
            try:
                calls_data = json.loads(match)
                if isinstance(calls_data, list):
                    for call_data in calls_data:
                        if isinstance(call_data, dict) and 'name' in call_data:
                            call_id = str(uuid.uuid4())
                            tool_calls.append(
                                ToolCall(
                                    call_id=call_id, name=call_data['name'], arguments=call_data.get('arguments', {})
                                )
                            )
            except json.JSONDecodeError:
                logger.error(f'Failed to parse multi-function calls: {match}')

        # Pattern 2: Individual function call tags
        # <function_call name="tool_name">{"arg": "value"}</function_call>
        single_pattern = r'<function_call name="([^"]+)">\s*({[^}]*}|\{.*?\})\s*</function_call>'
        single_matches = re.findall(single_pattern, content, re.DOTALL)

        for tool_name, args_str in single_matches:
            try:
                call_id = str(uuid.uuid4())
                arguments = json.loads(args_str)
                tool_calls.append(ToolCall(call_id=call_id, name=tool_name, arguments=arguments))
            except json.JSONDecodeError:
                logger.error(f'Failed to parse arguments for tool {tool_name}: {args_str}')

        # Pattern 3: Claude-style thinking with tool calls
        # Looking for patterns like:
        # I need to use the following tools:
        # <function_call name="tool_name">...</function_call>
        thinking_pattern = r'<function_call name="([^"]+)">\s*(\{.*?\})\s*</function_call>'
        thinking_matches = re.findall(thinking_pattern, content, re.DOTALL)

        for tool_name, args_str in thinking_matches:
            # Avoid duplicates from single_pattern
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
                    logger.error(f'Failed to parse thinking arguments for tool {tool_name}: {args_str}')

        return tool_calls

    def _parse_structured_response(self, content: str) -> dict[str, _t.Any]:
        """Parse the structured JSON response from the new prompt format.

        Expected format:
        ```json
        {
            "thinking": "Analysis and reasoning",
            "task_list": {
                "completed": ["Task 1", "Task 2"],
                "pending": ["Task 3", "Task 4"]
            },
            "tool_calls": [
                {
                    "name": "tool_name",
                    "arguments": {"param": "value"}
                }
            ],
            "message": "Clear explanation to user",
            "should_continue": true
        }
        ```

        Returns:
            Parsed response dict with extracted fields
        """
        # Initialize default response structure
        parsed_response = {
            'thinking': '',
            'task_list': {'completed': [], 'pending': []},
            'tool_calls': [],
            'message': '',
            'should_continue': True,
            'raw_content': content,
        }

        # Try to extract JSON from markdown code blocks first
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, content, re.DOTALL)

        response_data = None
        if json_matches:
            # Use the first JSON block found
            try:
                response_data = json.loads(json_matches[0])
            except json.JSONDecodeError as e:
                logger.warning(f'Failed to parse JSON from markdown block: {e}')

        # If no JSON block found, try parsing the entire content as JSON
        if not response_data:
            try:
                response_data = json.loads(content.strip())
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract information from text
                logger.debug('No valid JSON found, extracting from text content')
                parsed_response['message'] = content
                parsed_response['tool_calls'] = self._parse_tool_calls(content)
                return parsed_response

        # Extract fields from parsed JSON
        if response_data and isinstance(response_data, dict):
            parsed_response['thinking'] = response_data.get('thinking', '')
            parsed_response['message'] = response_data.get('message', '')
            parsed_response['should_continue'] = response_data.get('should_continue', True)

            # Extract task list
            task_list = response_data.get('task_list', {})
            if isinstance(task_list, dict):
                parsed_response['task_list']['completed'] = task_list.get('completed', [])
                parsed_response['task_list']['pending'] = task_list.get('pending', [])

            # Extract tool calls and convert to ToolCall objects
            tool_calls_data = response_data.get('tool_calls', [])
            if isinstance(tool_calls_data, list):
                for call_data in tool_calls_data:
                    if isinstance(call_data, dict) and 'name' in call_data:
                        call_id = str(uuid.uuid4())
                        tool_call = ToolCall(
                            call_id=call_id, name=call_data['name'], arguments=call_data.get('arguments', {})
                        )
                        parsed_response['tool_calls'].append(tool_call)

        return parsed_response

    async def _execute_tool_call(self, tool_call: ToolCall, task_id: str = 'default') -> ToolCallResult:
        await self._emit_event(
            'tool_call_start',
            {'call_id': tool_call.call_id, 'tool_name': tool_call.name, 'arguments': tool_call.arguments},
        )

        try:
            # Check if tool exists
            tool = self.tool_registry.get_tool(tool_call.name)
            if tool is None:
                return ToolCallResult(
                    call_id=tool_call.call_id, success=False, output='', error=f'Unknown tool: {tool_call.name}'
                )

            # Check approval for potentially dangerous commands
            if await self._needs_approval(tool_call):
                if not await self._request_approval(tool_call):
                    return ToolCallResult(
                        call_id=tool_call.call_id, success=False, output='', error='Tool execution rejected by user'
                    )

            # Execute the tool
            tool_context = ToolContext(tool_name=tool_call.name, arguments=tool_call.arguments, session_id=task_id)

            result = await self.tool_executor.execute(tool_call.name, tool_call.arguments, tool_context)

            output = str(result) if result is not None else ''

            await self._emit_event(
                'tool_call_end',
                {
                    'call_id': tool_call.call_id,
                    'tool_name': tool_call.name,
                    'success': True,
                    'output': output,  # No truncation - show full output
                },
            )

            return ToolCallResult(
                call_id=tool_call.call_id,
                success=True,
                output=output,  # No truncation - show full output
            )

        except Exception as e:
            error_msg = f'Tool execution error: {e!s}'
            logger.error(f'Error executing tool {tool_call.name}: {e}', exc_info=True)

            await self._emit_event(
                'tool_call_end',
                {'call_id': tool_call.call_id, 'tool_name': tool_call.name, 'success': False, 'error': error_msg},
            )

            return ToolCallResult(call_id=tool_call.call_id, success=False, output='', error=error_msg)

    async def _needs_approval(self, tool_call: ToolCall) -> bool:
        """Check if a tool call needs user approval."""
        if self.approval_mode == ApprovalMode.NEVER:
            return False
        if self.approval_mode == ApprovalMode.ALWAYS:
            return True
        # UNSAFE_ONLY
        # Check if this is a potentially unsafe operation
        dangerous_tools = {'shell', 'container_exec', 'file_write', 'exec'}
        if tool_call.name in dangerous_tools:
            # Check if command was already approved
            if tool_call.name == 'shell' or tool_call.name == 'container_exec':
                command = tool_call.arguments.get('command', [])
                if isinstance(command, str):
                    command = [command]
                command_tuple = tuple(command)
                return command_tuple not in self._approved_commands
            return True
        return False

    async def _request_approval(self, tool_call: ToolCall) -> bool:
        """Request user approval for a tool call."""
        # In a real implementation, this would show a UI prompt
        # For now, we'll auto-approve to keep the demo working
        logger.info(f'Auto-approving tool call: {tool_call.name} with args {tool_call.arguments}')

        # Remember this approval
        if tool_call.name in {'shell', 'container_exec'}:
            command = tool_call.arguments.get('command', [])
            if isinstance(command, str):
                command = [command]
            self._approved_commands.add(tuple(command))

        await self._emit_event(
            'approval_requested',
            {
                'call_id': tool_call.call_id,
                'tool_name': tool_call.name,
                'arguments': tool_call.arguments,
                'auto_approved': True,
            },
        )

        return True

    async def _run_turn(
        self, turn_input: list[Message], run_context: dict[str, _t.Any]
    ) -> tuple[list[Message], list[ToolCallResult]]:
        """Run a single turn of the conversation using prompt_run."""
        self._turn_count += 1

        await self._emit_event('turn_start', {'turn': self._turn_count, 'input_messages': len(turn_input)})

        try:
            # Get pre-converted tool definitions from run_context and convert to dict format
            tool_definitions_objects = run_context.get('tool_definitions', [])
            tool_definitions = [
                {'name': td.name, 'description': td.description, 'parameters': self._clean_parameters(td.parameters)}
                for td in tool_definitions_objects
            ]

            # Prepare conversation history for prompt
            history_messages = []
            for msg in turn_input[:-1]:  # All but the last message
                history_messages.append(
                    {
                        'role': 'user' if msg.role == Role.user else 'assistant',
                        'content': extract_text_from_message(msg),
                    }
                )

            # Get the current user input
            current_input = extract_text_from_message(turn_input[-1]) if turn_input else ''

            # Format tool results from previous turn if any
            tool_results_text = ''
            if hasattr(self, '_last_tool_results') and self._last_tool_results:
                tool_results_text = format_tool_results_for_conversation(self._last_tool_results)

            # Get task_id and task_list from run_context
            task_id = run_context.get('task_id', 'default')
            task_list = run_context.get('task_list', {'completed': [], 'pending': []})

            # Prepare variables for prompt_run with task list support
            variables = {
                'agent_name': self.name,
                'task_id': task_id,
                'turn_count': self._turn_count,
                'has_tools': len(tool_definitions) > 0,
                'tools': tool_definitions,
                'has_history': len(history_messages) > 0,
                'history_text': format_history_for_prompt(history_messages),
                'user_input': current_input if current_input else None,
                'tool_results': bool(tool_results_text),
                'tool_results_text': tool_results_text,
                'task_list': task_list,  # Use task_list from run_context
                'rules': self.rules,
            }

            logger.debug('RunTurn, task_id: %s, turn: %s, variables: %s', task_id, self._turn_count, variables)

            # Debug mode: Print detailed information about what's being sent to LLM
            if self.debug:
                print('\n' + '=' * 80)
                print(f'🐛 DEBUG MODE - Turn {self._turn_count}')
                print('=' * 80)
                print('📝 Variables being sent to LLM:')
                for key, value in variables.items():
                    if key == 'tools':
                        print(f'  {key}: [{len(value)} tools] {[tool.get("name", "unknown") for tool in value]}')
                    elif key == 'history_text':
                        print(f'  {key}: {len(str(value))} characters')
                        if value:
                            print(f'    Preview: {str(value)[:200]}...')
                    elif key == 'user_input':
                        print(f'  {key}: {value!r}')
                    elif key == 'tool_results_text':
                        if value:
                            print(f'  {key}: {len(str(value))} characters')
                            print(f'    Preview: {str(value)[:200]}...')
                        else:
                            print(f'  {key}: None')
                    else:
                        print(f'  {key}: {value!r}')

                print('\n🔧 Available Tools:')
                for i, tool_def in enumerate(tool_definitions):
                    print(
                        f'  {i + 1}. {tool_def.get("name", "unknown")}: {tool_def.get("description", "no description")}'
                    )

                print('\n📊 Context Summary:')
                print(f'  - Turn: {self._turn_count}/{self.max_turns}')
                print(f'  - Has previous results: {bool(tool_results_text)}')
                print(f'  - Has conversation history: {len(history_messages) > 0}')
                print(f'  - Current user input: {bool(current_input)}')
                print(f'  - Task ID: {task_id}')
                print('=' * 80 + '\n')

            # Use prompt_run to get LLM response with timeout handling
            try:
                response = await asyncio.wait_for(
                    prompt_run(
                        'code_agent_loop',
                        variables=variables,
                        tools=tool_definitions,  # Also pass to prompt_run for function calling
                        stream=False,  # Disable streaming for now to get complete response
                    ),
                    timeout=120.0,  # 2 minute timeout
                )
            except TimeoutError:
                logger.error('LLM request timed out after 2 minutes')
                error_message = self._create_agent_message(
                    "I'm sorry, but my request to the language model timed out. This might be due to "
                    'network issues or high server load. Please try again.',
                    task_id,
                )
                return [error_message], []
            except Exception as e:
                logger.error(f'LLM request failed: {e}')
                # Try to provide a helpful fallback response
                if 'list tools' in current_input.lower():
                    # Special case for tool listing
                    tools_list = []
                    for tool in self.tool_registry.get_tools():
                        tools_list.append(f'- {tool.name}: {tool.description}')

                    fallback_content = f'I have access to {len(self.tool_registry.get_tools())} tools:\n\n' + '\n'.join(
                        tools_list
                    )
                    fallback_message = self._create_agent_message(fallback_content, task_id)
                    return [fallback_message], []
                error_message = self._create_agent_message(
                    f'I encountered an error while processing your request: {e!s}. '
                    f'Please try rephrasing your request or try again later.',
                    task_id,
                )
                return [error_message], []

            # Extract content from response
            content = ''
            if hasattr(response, 'message') and response.message:
                # Extract text from A2A message parts
                for part in response.message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        content += part.root.text
                    elif hasattr(part, 'text'):
                        content += part.text
            elif hasattr(response, 'content'):
                if asyncio.iscoroutine(response.content):
                    content = str(await response.content)
                else:
                    content = str(response.content)
            else:
                content = str(response)

            # Parse the structured response using the new parser
            parsed_response = self._parse_structured_response(content)

            # Emit debug event with comprehensive debug information
            if self.debug:
                debug_info = {
                    'turn_count': self._turn_count,
                    'raw_content_length': len(content),
                    'thinking': parsed_response['thinking'],
                    'message': parsed_response['message'],
                    'should_continue': parsed_response['should_continue'],
                    'task_list': {
                        'completed_count': len(parsed_response['task_list']['completed']),
                        'pending_count': len(parsed_response['task_list']['pending']),
                        'completed': parsed_response['task_list']['completed'],
                        'pending': parsed_response['task_list']['pending'],
                    },
                    'tool_calls': [],
                }

                if parsed_response['tool_calls']:
                    debug_info['tool_calls'] = [
                        {
                            'name': tool_call.name,
                            'arguments_keys': list(tool_call.arguments.keys()),
                            'call_id': tool_call.call_id,
                        }
                        for tool_call in parsed_response['tool_calls']
                    ]

                await self._emit_event('debug_llm_response', debug_info)

            # Update task list in run_context from response
            if parsed_response['task_list']['completed'] or parsed_response['task_list']['pending']:
                run_context['task_list'] = parsed_response['task_list']
                logger.info(
                    f'Updated task list: completed={len(parsed_response["task_list"]["completed"])}, '
                    f'pending={len(parsed_response["task_list"]["pending"])}'
                )

            # Emit the LLM response with structured data
            await self._emit_event(
                'llm_response',
                {
                    'content': content,
                    'thinking': parsed_response['thinking'],
                    'message': parsed_response['message'],
                    'task_list': parsed_response['task_list'],
                    'should_continue': parsed_response['should_continue'],
                    'has_tool_calls': len(parsed_response['tool_calls']) > 0,
                },
            )

            # Get tool calls from parsed response
            tool_calls = parsed_response['tool_calls']

            # Execute tool calls if any
            tool_results = []
            if tool_calls:
                await self._emit_event('tool_results_start', {'num_tools': len(tool_calls)})

                for tool_call in tool_calls:
                    try:
                        # Emit tool call start event
                        await self._emit_event(
                            'tool_call_start',
                            {
                                'tool_name': tool_call.name,
                                'arguments': tool_call.arguments,
                                'call_id': tool_call.call_id,
                            },
                        )

                        # Execute the tool call with timeout
                        result = await asyncio.wait_for(
                            self._execute_tool_call(tool_call, task_id),
                            timeout=60.0,  # 1 minute timeout per tool call
                        )
                        tool_results.append(result)

                        # Emit tool call end event
                        await self._emit_event(
                            'tool_call_end',
                            {
                                'tool_name': tool_call.name,
                                'call_id': tool_call.call_id,
                                'success': result.success,
                                'output': result.output,
                                'error': result.error,
                            },
                        )

                    except TimeoutError:
                        logger.error(f'Tool call {tool_call.name} timed out after 1 minute')
                        error_result = ToolCallResult(
                            call_id=tool_call.call_id,
                            success=False,
                            output='',
                            error='Tool execution timed out after 1 minute',
                        )
                        tool_results.append(error_result)

                        await self._emit_event(
                            'tool_call_end',
                            {
                                'tool_name': tool_call.name,
                                'call_id': tool_call.call_id,
                                'success': False,
                                'output': '',
                                'error': 'Tool execution timed out',
                            },
                        )

                    except Exception as e:
                        logger.error(f'Error executing tool call {tool_call.name}: {e}')
                        error_result = ToolCallResult(call_id=tool_call.call_id, success=False, output='', error=str(e))
                        tool_results.append(error_result)

                        await self._emit_event(
                            'tool_call_end',
                            {
                                'tool_name': tool_call.name,
                                'call_id': tool_call.call_id,
                                'success': False,
                                'output': '',
                                'error': str(e),
                            },
                        )

                await self._emit_event(
                    'tool_results',
                    {
                        'num_tools_executed': len(tool_results),
                        'successful_calls': sum(1 for r in tool_results if r.success),
                        'failed_calls': sum(1 for r in tool_results if not r.success),
                    },
                )

            # Store tool results for next turn
            self._last_tool_results = tool_results

            # Create response message using the structured message field
            response_message_content = parsed_response['message'] or content
            response_message = self._create_agent_message(response_message_content, task_id)

            # Store the should_continue flag for task completion checking
            run_context['should_continue'] = parsed_response['should_continue']

            # Add tool results to conversation if any
            if tool_results:
                # Create a tool results message
                tool_results_text = format_tool_results_for_conversation(tool_results)
                tool_results_message = self._create_agent_message(
                    f'Tool execution results:\n{tool_results_text}', task_id
                )
                return [response_message, tool_results_message], tool_results
            return [response_message], tool_results

        except Exception as e:
            logger.error(f'Error in turn execution: {e}')
            await self._emit_event('turn_error', {'error': str(e), 'turn': self._turn_count})

            # Create error response
            error_message = self._create_agent_message(f'I encountered an error: {e!s}', task_id)
            return [error_message], []

    def _parse_json_tool_calls(self, content: str) -> list[ToolCall]:
        """Parse tool calls from JSON format in the LLM response."""
        tool_calls = []

        try:
            # Look for JSON blocks in the content
            import re

            json_pattern = r'```json\s*(\{.*?\})\s*```'
            matches = re.findall(json_pattern, content, re.DOTALL)

            for match in matches:
                try:
                    data = json.loads(match)
                    if 'tool_calls' in data and isinstance(data['tool_calls'], list):
                        for tc_data in data['tool_calls']:
                            if isinstance(tc_data, dict) and 'name' in tc_data:
                                tool_call = ToolCall(
                                    call_id=str(uuid.uuid4()),
                                    name=tc_data['name'],
                                    arguments=tc_data.get('arguments', {}),
                                )
                                tool_calls.append(tool_call)
                except json.JSONDecodeError as e:
                    logger.warning(f'Failed to parse JSON tool call: {e}')
                    continue

        except Exception as e:
            logger.warning(f'Error parsing tool calls from content: {e}')

        return tool_calls


    async def run(self, input_data: AgentRunInput) -> AgentRunOutput:
        """Run the agent with iterative execution until task completion."""
        self._start_time = time.time()
        task_id = str(uuid.uuid4())

        # Convert tools to definitions once at the start
        tool_definitions = to_tool_definitions(self.tool_registry.get_tools())

        # Initialize run context with task-specific data
        run_context = {
            'task_id': task_id,
            'task_list': {'completed': [], 'pending': []},
            'tool_definitions': tool_definitions,
            'should_continue': True,
        }

        # Initialize conversation with user input
        conversation_history = [input_data.message]

        # Track execution state
        total_tool_calls = 0
        max_iterations = self.max_turns
        iteration = 0

        await self._emit_event(
            'task_start',
            {
                'task_id': task_id,
                'session_id': self._context_id,
                'iteration': 0,
                'elapsed_time': 0.0,
                'user_input': extract_text_from_message(input_data.message),
            },
        )

        try:
            # Main execution loop - like Codex pattern
            while iteration < max_iterations:
                iteration += 1
                elapsed_time = time.time() - self._start_time

                # Check if we should continue execution
                continue_decision = self._should_continue_execution(iteration, elapsed_time, total_tool_calls)

                if continue_decision != ContinueDecision.CONTINUE:
                    logger.info(f'Stopping execution: {continue_decision.value}')
                    break

                # Run a single turn
                response_messages, tool_results = await self._run_turn(conversation_history, run_context)

                # Add response messages to conversation history
                conversation_history.extend(response_messages)

                # Update tool call count
                total_tool_calls += len(tool_results)

                # Check if the task is complete
                if self._is_task_complete(response_messages, tool_results, run_context):
                    logger.info('Task completed successfully')
                    await self._emit_event(
                        'task_complete',
                        {
                            'task_id': task_id,
                            'iterations': iteration,
                            'total_tool_calls': total_tool_calls,
                            'elapsed_time': elapsed_time,
                        },
                    )
                    break

                # If there were tool calls, continue the loop to let the LLM process results
                if tool_results:
                    # Add tool results to conversation and continue
                    continue
                # No tool calls, task is likely complete
                logger.info('No tool calls in response, task appears complete')
                await self._emit_event(
                    'task_complete',
                    {
                        'task_id': task_id,
                        'iterations': iteration,
                        'total_tool_calls': total_tool_calls,
                        'elapsed_time': elapsed_time,
                    },
                )
                break

            # If we hit max iterations
            if iteration >= max_iterations:
                logger.warning(f'Reached maximum iterations ({max_iterations})')
                await self._emit_event(
                    'task_timeout',
                    {'task_id': task_id, 'max_iterations': max_iterations, 'total_tool_calls': total_tool_calls},
                )

            # Get the final response (last assistant message)
            final_response = None
            for msg in reversed(conversation_history):
                if msg.role == Role.agent:
                    final_response = msg
                    break

            if not final_response:
                # Create a default response if none found
                final_response = self._create_agent_message('Task completed.', task_id)

            # Store conversation in memory
            if self.memory_system:
                try:
                    for msg in conversation_history:
                        await self.memory_system.add_message(msg)
                except Exception as e:
                    logger.warning(f'Failed to store conversation in memory: {e}')

            await self._emit_event(
                'task_end',
                {
                    'task_id': task_id,
                    'session_id': self._context_id,
                    'iteration': iteration,
                    'elapsed_time': time.time() - self._start_time,
                },
            )

            return AgentRunOutput(
                result=final_response,
                metadata={
                    'task_id': task_id,
                    'iterations': iteration,
                    'total_tool_calls': total_tool_calls,
                    'elapsed_time': time.time() - self._start_time,
                    'conversation_length': len(conversation_history),
                },
            )

        except Exception as e:
            logger.error(f'Error in agent execution: {e}', exc_info=True)
            await self._emit_event('task_error', {'task_id': task_id, 'error': str(e), 'iteration': iteration})

            # Return error response
            error_response = self._create_agent_message(f'I encountered an error: {e!s}', task_id)
            await self._emit_event(
                'task_end',
                {
                    'task_id': task_id,
                    'session_id': self._context_id,
                    'iteration': iteration,
                    'elapsed_time': time.time() - self._start_time,
                },
            )
            return AgentRunOutput(
                result=error_response,
                metadata={
                    'task_id': task_id,
                    'error': str(e),
                    'iterations': iteration,
                    'total_tool_calls': total_tool_calls,
                    'elapsed_time': time.time() - self._start_time,
                },
            )

    def _is_task_complete(
        self, response_messages: list[Message], tool_results: list[ToolCallResult], run_context: dict[str, _t.Any]
    ) -> bool:
        """Determine if the task is complete based on the response."""
        # Check if we have the should_continue flag from the structured response
        if 'should_continue' in run_context:
            # Use the explicit should_continue flag from the LLM response
            task_complete = not run_context['should_continue']
            logger.info(f'Task completion based on should_continue flag: {task_complete}')
            return task_complete

        # Fallback to original logic if no should_continue field found
        # If there are no tool calls and the response doesn't indicate more work needed
        if not tool_results:
            # Check if the last message indicates completion
            if response_messages:
                last_message_text = extract_text_from_message(response_messages[-1]).lower()

                # Look for completion indicators
                completion_indicators = [
                    'task completed',
                    'finished',
                    'done',
                    'complete',
                    'successfully created',
                    'successfully executed',
                    'no further',
                    "that's all",
                    'task is complete',
                ]

                # Look for continuation indicators
                continuation_indicators = [
                    'let me',
                    "i'll",
                    'next i',
                    'now i',
                    'first i',
                    'tool_calls',
                    'need to',
                    'should',
                    'will',
                ]

                has_completion = any(indicator in last_message_text for indicator in completion_indicators)
                has_continuation = any(indicator in last_message_text for indicator in continuation_indicators)

                # If it has completion indicators and no continuation indicators, consider it complete
                if has_completion and not has_continuation:
                    return True

                # If it has no continuation indicators and no tool calls, likely complete
                if not has_continuation and 'tool_calls' not in last_message_text:
                    return True

        return False

    async def reset_conversation(self) -> None:
        """Reset the conversation history and start fresh."""
        self._conversation_history.clear()
        self._context_id = str(uuid.uuid4())
        self._approved_commands.clear()
        self._completed_actions.clear()
        self._last_tool_calls.clear()

        await self._emit_event('conversation_reset', {'new_context_id': self._context_id})

        logger.info('Conversation history reset')

    async def get_memory_history(self, limit: int = 50, query: str | None = None) -> list[Message]:
        """Get conversation history from memory system with optional search.

        Args:
            limit: Maximum number of recent messages to return
            query: Optional search query to include relevant memory chunks

        Returns:
            List of Messages including recent messages and relevant memory chunks
        """
        return await self.memory_system.get_history(self._context_id, limit, query)

    async def compress_conversation_history(self, keep_recent: int = 20, chunk_size: int = 10) -> int:
        """Compress old conversation history into memory chunks.

        Args:
            keep_recent: Number of recent messages to keep uncompressed
            chunk_size: Number of messages per chunk

        Returns:
            Number of chunks created
        """

        # Create LLM summarizer function
        async def llm_summarizer(text: str) -> dict[str, _t.Any]:
            """Use the agent's LLM to create intelligent summaries."""
            try:
                # Use conversation_summary template
                from ..prompt import prompt_run

                response = await prompt_run('conversation_summary', variables={'conversation_text': text})

                # Extract content from response
                content = ''
                if hasattr(response, 'message') and response.message:
                    # Extract text from A2A message parts
                    for part in response.message.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            content += part.root.text
                        elif hasattr(part, 'text'):
                            content += part.text
                elif hasattr(response, 'content'):
                    content = str(response.content)
                else:
                    content = str(response)

                # Try to parse JSON response
                import json

                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Fallback if LLM doesn't return valid JSON
                    return {
                        'summary': content,  # Full content, no truncation
                        'entities': {},
                        'id_mappings': {},
                    }
            except Exception:
                # Fallback to simple summary
                return {
                    'summary': text,  # Full text, no truncation
                    'entities': {},
                    'id_mappings': {},
                }

        # Use memory system's compression with LLM summarizer
        if hasattr(self.memory_system, 'compress_old_messages'):
            return await self.memory_system.compress_old_messages(
                self._context_id, keep_recent, chunk_size, llm_summarizer
            )
        return 0

    def get_conversation_summary(self) -> dict[str, _t.Any]:
        """Get a summary of the current conversation state."""
        return {
            'context_id': self._context_id,
            'message_count': len(self._conversation_history),
            'user_messages': len([msg for msg in self._conversation_history if msg.role == Role.user]),
            'agent_messages': len([msg for msg in self._conversation_history if msg.role == Role.agent]),
            'approved_commands': len(self._approved_commands),
            'last_activity': (
                self._conversation_history[-1].metadata.get('timestamp') if self._conversation_history else None
            ),
        }

    def add_tool(self, tool) -> None:
        """Add a tool or toolset to the agent."""
        if hasattr(tool, 'tools'):  # It's a toolset
            self.tool_registry.register_toolset(tool)
        else:  # It's a single tool
            self.tool_registry.register_tool(tool)

    async def cleanup(self) -> None:
        """Cleanup the agent and all its resources."""
        logger.debug('Cleaning up CodeAgent resources...')

        # Cleanup all toolsets
        for toolset in self.tool_registry.get_toolsets():
            try:
                if hasattr(toolset, 'cleanup'):
                    await toolset.cleanup()
                elif hasattr(toolset, 'close'):
                    await toolset.close()
                logger.debug(f'Cleaned up toolset: {toolset.name}')
            except Exception as e:
                logger.warning(f'Error cleaning up toolset {toolset.name}: {e}')

        # Cleanup individual tools
        for tool in self.tool_registry.get_tools():
            try:
                if hasattr(tool, 'cleanup'):
                    await tool.cleanup()
                elif hasattr(tool, 'close'):
                    await tool.close()
            except Exception as e:
                logger.warning(f'Error cleaning up tool {tool.name}: {e}')

        # Cleanup sandbox if it's managed by this agent
        if hasattr(self.sandbox, 'down'):
            try:
                await self.sandbox.down()
                logger.debug('Cleaned up sandbox')
            except Exception as e:
                logger.warning(f'Error cleaning up sandbox: {e}')

        logger.debug('CodeAgent cleanup completed')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _message_to_llm_dict(msg: Message) -> dict[str, _t.Any]:
        """Convert A2A :class:`Message` to OpenAI message dict."""
        if not msg.parts:
            raise ValueError('Message parts is empty')

        # Concatenate text parts for now – richer multipart support can be added later
        text_parts = []
        for p in msg.parts:
            # Handle Part objects that contain TextPart objects
            if hasattr(p, 'root') and hasattr(p.root, 'text'):
                text_parts.append(p.root.text)
            # Handle direct TextPart objects
            elif hasattr(p, 'text'):
                text_parts.append(p.text)

        content_str = '\n'.join(text_parts)
        role_map = {
            Role.user: 'user',
            Role.agent: 'assistant',
        }
        return {
            'role': role_map[msg.role],
            'content': content_str,
        }

    def _detect_and_intervene_loop(
        self, original_user_input: str, tool_calls: list[ToolCall]
    ) -> tuple[str | None, list[ToolCall]]:
        """Detect if the agent is stuck in a loop and intervene by modifying tool calls.

        Returns:
            Tuple of (intervention_message, modified_tool_calls)
        """
        # Check if we've repeated similar tool calls multiple times
        if len(self._last_tool_calls) >= 3:
            recent_calls = self._last_tool_calls[-3:]

            # Check for exact duplicates
            if len(set(recent_calls)) == 1:
                repeated_call = recent_calls[0]
                return self._generate_intervention_and_fix(repeated_call, original_user_input, tool_calls)

            # Check for similar calls (same tool and file path, but different content)
            if all(call.startswith('sandbox_write_file:') for call in recent_calls):
                # Extract file paths
                file_paths = [call.split(':', 1)[1] for call in recent_calls]
                if len(set(file_paths)) == 1:  # Same file being written multiple times
                    file_path = file_paths[0]
                    if f'sandbox_write_file:{file_path}' in self._completed_actions:
                        return self._generate_intervention_and_fix(
                            f'sandbox_write_file:{file_path}', original_user_input, tool_calls
                        )

        return None, tool_calls

    def _generate_intervention_and_fix(
        self, repeated_call: str, original_user_input: str, tool_calls: list[ToolCall]
    ) -> tuple[str, list[ToolCall]]:
        """Generate intervention message and modify tool calls to force progression."""
        if repeated_call.startswith('sandbox_write_file:'):
            # Extract the file path
            file_path = repeated_call.split(':', 1)[1]

            # Check if this is a "create and run" scenario
            if any(keyword in original_user_input.lower() for keyword in ['run', 'execute', 'and run']):
                intervention_msg = (
                    f'SYSTEM INTERVENTION: The file {file_path} has been successfully created multiple times. '
                    f'Proceeding to execute it instead of recreating it.'
                )

                # Force the correct tool call - replace any sandbox_write_file calls with sandbox_exec
                modified_calls = []
                for call in tool_calls:
                    if call.name == 'sandbox_write_file' and call.arguments.get('path') == file_path:
                        # Replace with execution call
                        if file_path.endswith('.py'):
                            exec_call = ToolCall(
                                call_id=call.call_id, name='sandbox_exec', arguments={'cmd': f'python {file_path}'}
                            )
                        else:
                            exec_call = ToolCall(
                                call_id=call.call_id, name='sandbox_exec', arguments={'cmd': file_path}
                            )
                        modified_calls.append(exec_call)
                    else:
                        modified_calls.append(call)

                # If no write calls were found, add the execution call
                if not any(call.name == 'sandbox_write_file' for call in tool_calls):
                    if file_path.endswith('.py'):
                        exec_call = ToolCall(
                            call_id=str(uuid.uuid4()), name='sandbox_exec', arguments={'cmd': f'python {file_path}'}
                        )
                    else:
                        exec_call = ToolCall(
                            call_id=str(uuid.uuid4()), name='sandbox_exec', arguments={'cmd': file_path}
                        )
                    modified_calls.append(exec_call)

                return intervention_msg, modified_calls
            intervention_msg = (
                f'SYSTEM INTERVENTION: The file {file_path} has been successfully created.'
                ' Task is complete.'
            )
            return intervention_msg, []  # No more tool calls needed

        return (
            'SYSTEM INTERVENTION: You appear to be repeating the same action. Please proceed to the next step.',
            tool_calls,
        )

    def _clean_parameters(self, parameters: dict) -> dict:
        """Clean parameters to remove Undefined values and make them JSON serializable."""
        import json

        def clean_value(value):
            """Recursively clean a value to make it JSON serializable."""
            if hasattr(value, '__class__') and value.__class__.__name__ == 'Undefined':
                return None
            if isinstance(value, dict):
                return {
                    k: clean_value(v)
                    for k, v in value.items()
                    if not (hasattr(v, '__class__') and v.__class__.__name__ == 'Undefined')
                }
            if isinstance(value, list | tuple):
                return [
                    clean_value(item)
                    for item in value
                    if not (hasattr(item, '__class__') and item.__class__.__name__ == 'Undefined')
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
