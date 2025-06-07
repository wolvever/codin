"""Base agent implementation using codin architecture components.

This module provides a simplified agent implementation that uses the codin
architecture including prompt_run() for LLM interactions, Memory for conversation
history, Tools for capabilities, and bidirectional mailbox for communication.
"""

import asyncio
import logging
import time
import typing as _t
import uuid

from datetime import datetime
from enum import Enum

# Import a2a types that we need
from a2a.types import Role, TextPart

from ..actor.mailbox import LocalMailbox, Mailbox
from ..memory.base import MemMemoryService, Memory
from ..model.base import BaseLLM
from ..tool.base import Tool, ToolContext
from .base import Agent, Planner
from .types import (
    AgentRunInput,
    AgentRunOutput,
    ControlSignal,
    EventStep,
    FinishStep,
    Message,  # Use Message from codin.agent.types
    MessageStep,
    Metrics,
    RunConfig,
    RunnerControl,
    State,
    Step,
    StepType,
    Task,
    ThinkStep,
    ToolCallStep,
    ToolUsePart,
)


__all__ = ['BaseAgent', 'ContinueDecision']
logger = logging.getLogger('codin.agent.base_agent')


class ContinueDecision:
    """Continue decision for agent."""
    CONTINUE, NEED_APPROVAL, FINISH, CANCEL = 'continue', 'need_approval', 'finish', 'cancel'


class BaseAgent(Agent):
    """Simplified agent using codin architecture components.

    Uses:
    - prompt_run() for LLM interactions (via Planner)
    - Memory class for conversation history
    - Tools from tool system
    - Bidirectional mailbox for event communication and control
    - Sandbox through tools for file I/O
    """

    def __init__(
        self,
        *,
        agent_id: str | None = None,
        name: str = 'BaseAgent',
        description: str = 'Agent using codin architecture',
        version: str | None = None,
        planner: Planner,
        memory: Memory | None = None,
        tools: list[Tool] | None = None,
        llm: BaseLLM | None = None,
        mailbox: Mailbox | None = None,
        default_config: RunConfig | None = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(
            id=agent_id, name=name, description=description, version=version or '1.0.0', tools=tools or [], **kwargs
        )
        self.planner = planner
        self.memory = memory or MemMemoryService()
        self.tools = tools or []
        self.llm = llm
        self.mailbox = mailbox or LocalMailbox()
        self.default_config = default_config
        self.debug = debug

        # State for control handling
        self._paused = False
        self._cancelled = False

        logger.info(f'Initialized {self.id} with {len(self.tools)} tools')

    @property
    def agent_id(self) -> str:
        """Compatibility property for agent_id."""
        return self.id

    def get_tool_by_name(self, tool_name: str) -> Tool | None:
        """Get a tool by name from the tools list."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    async def handle_control(self, control: RunnerControl) -> None:
        """Handle control signals from dispatcher or other agents."""
        logger.info(f'Agent {self.id} received control signal: {control.signal}')

        if control.signal == ControlSignal.PAUSE:
            self._paused = True
            await self.mailbox.put_outbox(
                Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[TextPart(text=f'Agent {self.id} paused')],
                    contextId=self.id,
                    kind='message',
                    metadata={'control_response': 'paused'},
                )
            )

        elif control.signal == ControlSignal.RESUME:
            self._paused = False
            await self.mailbox.put_outbox(
                Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[TextPart(text=f'Agent {self.id} resumed')],
                    contextId=self.id,
                    kind='message',
                    metadata={'control_response': 'resumed'},
                )
            )

        elif control.signal in (ControlSignal.CANCEL, ControlSignal.STOP):
            self._cancelled = True
            await self.mailbox.put_outbox(
                Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[TextPart(text=f'Agent {self.id} cancelled')],
                    contextId=self.id,
                    kind='message',
                    metadata={'control_response': 'cancelled'},
                )
            )

        elif control.signal == ControlSignal.RESET:
            self._paused = False
            self._cancelled = False
            # Clear memory if possible
            if hasattr(self.memory, 'clear_all'):
                await self.memory.clear_all()
            await self.mailbox.put_outbox(
                Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[TextPart(text=f'Agent {self.id} reset')],
                    contextId=self.id,
                    kind='message',
                    metadata={'control_response': 'reset'},
                )
            )

    async def check_inbox_for_control(self) -> bool:
        """Check inbox for control messages and handle them. Returns True if should continue."""
        try:
            # Non-blocking check for control messages
            message = await self.mailbox.get_inbox(timeout=0.1)

            # Check if this is a control message
            if message.metadata and message.metadata.get('control'):
                control_signal = ControlSignal(message.metadata['control'])
                control = RunnerControl(signal=control_signal, metadata=message.metadata)
                await self.handle_control(control)

            return not self._cancelled

        except TimeoutError:
            # No messages, continue
            return not self._cancelled
        except Exception as e:
            logger.warning(f'Error checking inbox: {e}')
            return not self._cancelled

    async def wait_while_paused(self) -> None:
        """Wait while paused, checking for control messages."""
        while self._paused and not self._cancelled:
            await asyncio.sleep(0.1)
            await self.check_inbox_for_control()

    async def run(self, input_data: AgentRunInput) -> _t.AsyncGenerator[AgentRunOutput]:
        """Execute agent logic using simplified architecture."""
        start_time = time.time()
        session_id = input_data.session_id or f'session_{uuid.uuid4().hex[:8]}'

        try:
            # Build state with simple components
            state = await self._build_state(session_id, input_data)

            # Emit start event via mailbox
            await self._emit_event(
                'run_start',
                {
                    'agent_id': self.id,
                    'session_id': session_id,
                    'input_message_id': input_data.message.messageId if input_data.message else 'N/A',
                },
            )

            async for output in self._execute_planning_loop(state, session_id, start_time):
                yield output

        except Exception as e:
            logger.error(f'Error in agent run: {e}', exc_info=True)
            await self._emit_event('run_error', {'agent_id': self.id, 'session_id': session_id, 'error': str(e)})

            error_message = Message(
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                parts=[TextPart(text=f'I encountered an error: {e!s}')],
                contextId=session_id,
                kind='message',
            )
            yield AgentRunOutput(
                id=str(uuid.uuid4()), result=error_message, metadata={'error': str(e), 'agent_id': self.id}
            )

        finally:
            await self.cleanup()
            await self._emit_event(
                'run_end',
                {
                    'agent_id': self.id,
                    'session_id': session_id,
                    'elapsed_time': time.time() - start_time,
                },
            )

    async def _build_state(self, session_id: str, input_data: AgentRunInput) -> State:
        """Build state using simple memory and components."""
        # Set the session ID in memory if it's MemMemoryService
        if isinstance(self.memory, MemMemoryService):
            self.memory.set_session_id(session_id)

        # Add input message to memory if provided
        if input_data.message:
            # Ensure message has correct contextId for this session
            if input_data.message.contextId != session_id:
                # Create a copy with the correct contextId
                from a2a.types import Message

                corrected_message = Message(
                    messageId=input_data.message.messageId,
                    role=input_data.message.role,
                    parts=input_data.message.parts,
                    contextId=session_id,  # Use our session_id
                    kind=input_data.message.kind,
                    metadata=input_data.message.metadata,
                    taskId=input_data.message.taskId,
                    referenceTaskIds=input_data.message.referenceTaskIds,
                )
                await self.memory.add_message(corrected_message)
            else:
                await self.memory.add_message(input_data.message)

        # Get conversation history from memory (no session_id parameter needed)
        history = await self.memory.get_history()

        # Simple metrics tracking
        metrics = Metrics(iterations=0, input_token_used=0, output_token_used=0, cost_used=0.0, time_used=0.0)

        # Extract config from input
        config_data = input_data.options.get('config') if input_data.options else None
        current_config = self.default_config
        if isinstance(config_data, RunConfig):
            current_config = config_data
        elif isinstance(config_data, dict):
            current_config = RunConfig(**config_data)

        # Create task from input message
        task = None
        if input_data.message:
            query = ''
            if input_data.message.parts:
                for part in input_data.message.parts:
                    if isinstance(part, TextPart) and hasattr(part, 'text'):
                        query = part.text
                        break
            if query:
                task = Task(
                    id=input_data.message.messageId or str(uuid.uuid4()),
                    query=query,
                    metadata={'session_id': session_id, 'agent_id': self.id},
                    parts=[],
                )

        return State(
            session_id=session_id,
            agent_id=self.id,
            created_at=datetime.now(),
            iteration=0,
            history=list(history),
            artifact_ref=None,  # File operations through tools/sandbox
            tools=self.tools,
            metrics=metrics,
            config=current_config,
            task=task,
            metadata={
                'agent_id': self.id,
                'input_options': input_data.options or {},
                'memory': self.memory,  # Provide memory access to planner
                'mailbox': self.mailbox,  # Provide mailbox access to planner
            },
        )

    async def _execute_planning_loop(
        self, state: State, session_id: str, start_time: float
    ) -> _t.AsyncGenerator[AgentRunOutput]:
        """Execute planning loop with budget constraints and control handling."""

        await self._emit_event(
            'task_start',
            {
                'session_id': session_id,
                'iteration': state.iteration,
                'elapsed_time': 0.0,
            },
        )
        while state.iteration < (state.config.turn_budget or 100):
            # Check for control messages and handle pause/cancel states
            should_continue = await self.check_inbox_for_control()
            if not should_continue:
                logger.info(f'Agent {self.id} cancelled by control signal')
                break

            # Wait while paused
            if self._paused:
                await self.wait_while_paused()
                if self._cancelled:
                    break

            elapsed_time = time.time() - start_time
            state.metrics.time_used = elapsed_time

            # Check budget constraints
            is_exceeded, reason = state.config.is_budget_exceeded(state.metrics, elapsed_time)
            if is_exceeded:
                logger.warning(f'Budget constraint exceeded: {reason}')
                finish_reason = f'Budget exceeded: {reason}'
                finish_msg = Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[TextPart(text=finish_reason)],
                    contextId=session_id,
                    kind='message',
                )
                finish_step = FinishStep(step_id=str(uuid.uuid4()), reason=finish_reason, final_message=finish_msg)

                # Send output to mailbox as well as yielding
                await self.mailbox.put_outbox(finish_msg)

                async for output in self._execute_step(finish_step, state, session_id):
                    yield output
                break

            await self._emit_event(
                'turn_start',
                {'session_id': session_id, 'iteration': state.iteration, 'metrics': state.metrics.__dict__},
            )

            steps_executed = 0
            task_finished_by_step = False

            try:
                # Generate steps from planner
                async for step in self.planner.next(state):
                    # Check for control messages between steps
                    should_continue = await self.check_inbox_for_control()
                    if not should_continue:
                        break

                    if self._paused:
                        await self.wait_while_paused()
                        if self._cancelled:
                            break

                    steps_executed += 1
                    if self.debug:
                        logger.debug(
                            f'Executing step {step.step_id}: '
                            f'{step.step_type.value if isinstance(step.step_type, Enum) else step.step_type}'
                        )

                    # If planner yields a message, add it to memory
                    if step.step_type == StepType.MESSAGE and isinstance(step, MessageStep) and step.message:
                        if not any(
                            h.messageId == step.message.messageId
                            for h in state.history
                            if h.messageId and step.message.messageId
                        ):
                            await self.memory.add_message(step.message)
                            state.history.append(step.message)

                    async for output in self._execute_step(step, state, session_id):
                        yield output

                    if step.step_type == StepType.FINISH:
                        task_finished_by_step = True
                        await self._emit_event(
                            'task_complete',
                            {
                                'session_id': session_id,
                                'iteration': state.iteration,
                                'reason': step.reason if isinstance(step, FinishStep) and step.reason else 'Finished',
                            },
                        )
                        break

                    if steps_executed >= 10:
                        logger.warning('Maximum steps per turn reached from planner')
                        break

                if task_finished_by_step or self._cancelled:
                    break

                state.iteration += 1

                await self._emit_event(
                    'turn_end',
                    {'session_id': session_id, 'iteration': state.iteration, 'steps_executed': steps_executed},
                )

            except Exception as e:
                logger.error(f'Error in planning loop: {e}', exc_info=True)
                state.metrics.increment_errors()
                await self._emit_event(
                    'turn_error', {'session_id': session_id, 'iteration': state.iteration, 'error': str(e)}
                )

                error_msg = Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[TextPart(text=f'Error during planning: {e!s}')],
                    contextId=session_id,
                    kind='message',
                )

                # Send error to mailbox as well
                await self.mailbox.put_outbox(error_msg)

                yield AgentRunOutput(
                    id=str(uuid.uuid4()), result=error_msg, metadata={'error': str(e), 'agent_id': self.id}
                )
                break

        await self._emit_event(
            'task_end',
            {
                'session_id': session_id,
                'iteration': state.iteration,
                'elapsed_time': time.time() - start_time,
            },
        )

    async def _execute_step(self, step: Step, state: State, session_id: str) -> _t.AsyncGenerator[AgentRunOutput]:
        """Execute step using codin components and send outputs to mailbox."""
        step_output_metadata_base = {'agent_id': self.id, 'step_id': step.step_id}

        try:
            if step.step_type == StepType.MESSAGE and isinstance(step, MessageStep):
                if step.is_streaming and step.message_stream is not None:
                    collected: list[str] = []
                    async for chunk in step.stream_content():
                        collected.append(chunk)
                        stream_msg = Message(
                            messageId=f"{step.step_id}-stream",
                            role=Role.agent,
                            parts=[TextPart(text=chunk)],
                            contextId=session_id,
                            kind='message',
                            metadata={'stream': True, 'step_id': step.step_id},
                        )
                        await self.mailbox.put_outbox(stream_msg)
                        yield AgentRunOutput(
                            id=step.step_id,
                            result=stream_msg,
                            metadata={**step_output_metadata_base, 'step_type': 'message', 'stream': True},
                        )
                    if step.message is None:
                        step.message = Message(
                            messageId=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=[TextPart(text="".join(collected))],
                            contextId=session_id,
                            kind='message',
                        )
                    await self.memory.add_message(step.message)
                    await self.mailbox.put_outbox(step.message)
                    yield AgentRunOutput(
                        id=step.step_id,
                        result=step.message,
                        metadata={**step_output_metadata_base, 'step_type': 'message', 'is_streaming': True},
                    )
                elif step.message:
                    await self.memory.add_message(step.message)
                    await self.mailbox.put_outbox(step.message)
                    yield AgentRunOutput(
                        id=step.step_id,
                        result=step.message,
                        metadata={
                            **step_output_metadata_base,
                            'step_type': 'message',
                            'is_streaming': step.is_streaming,
                        },
                    )

            elif step.step_type == StepType.EVENT and isinstance(step, EventStep) and step.event:
                # Event step - emit via mailbox and internal event system
                await self._emit_event(
                    'agent_event', {'session_id': session_id, 'step_id': step.step_id, 'event': step.event}
                )

                # Create a message representation of the event for outbox
                event_msg = Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[TextPart(text=f'Event: {step.event}')],
                    contextId=session_id,
                    kind='message',
                    metadata={'event_type': 'agent_event', 'step_id': step.step_id},
                )
                await self.mailbox.put_outbox(event_msg)

                yield AgentRunOutput(
                    id=step.step_id, result=step.event, metadata={**step_output_metadata_base, 'step_type': 'event'}
                )

            elif step.step_type == StepType.TOOL_CALL and isinstance(step, ToolCallStep) and step.tool_call:
                # Tool call step - execute tool and create result message
                result_tool_use_part = await self._execute_tool_call(step.tool_call, state)
                step.tool_call_result = result_tool_use_part
                state.metrics.increment_tool_calls()

                # Create tool interaction message
                tool_interaction_parts: list[TextPart | ToolUsePart] = []
                tool_interaction_parts.append(step.tool_call)
                if result_tool_use_part:
                    tool_interaction_parts.append(result_tool_use_part)

                tool_interaction_message = Message(
                    messageId=f'toolmsg_{step.tool_call.id if step.tool_call.id else str(uuid.uuid4())}',
                    role=Role.agent,
                    parts=tool_interaction_parts,
                    contextId=session_id,
                    kind='message',
                )

                # Add to memory, send to outbox, and update state
                await self.memory.add_message(tool_interaction_message)
                await self.mailbox.put_outbox(tool_interaction_message)

                if not any(
                    h.messageId == tool_interaction_message.messageId
                    for h in state.history
                    if h.messageId and tool_interaction_message.messageId
                ):
                    state.history.append(tool_interaction_message)

                success_from_meta = (
                    result_tool_use_part.metadata.get('success', False) if result_tool_use_part.metadata else False
                )
                yield AgentRunOutput(
                    id=step.step_id,
                    result=tool_interaction_message,
                    metadata={
                        **step_output_metadata_base,
                        'step_type': 'tool_call',
                        'tool_name': step.tool_call.name,
                        'success': success_from_meta,
                    },
                )

            elif step.step_type == StepType.THINK and isinstance(step, ThinkStep) and self.debug:
                # Think step - emit thinking event
                await self._emit_event(
                    'agent_thinking', {'session_id': session_id, 'step_id': step.step_id, 'thinking': step.thinking}
                )

            elif step.step_type == StepType.FINISH and isinstance(step, FinishStep) and step.final_message:
                # Finish step - add final message to memory
                if not any(
                    h.messageId == step.final_message.messageId
                    for h in state.history
                    if h.messageId and step.final_message.messageId
                ):
                    await self.memory.add_message(step.final_message)
                    state.history.append(step.final_message)

                yield AgentRunOutput(
                    id=step.step_id,
                    result=step.final_message,
                    metadata={
                        **step_output_metadata_base,
                        'step_type': 'finish',
                        'reason': step.reason or 'Finished',
                        'success': step.success,
                    },
                )

        except Exception as e:
            logger.error(f'Error executing step {step.step_id}: {e}', exc_info=True)
            state.metrics.increment_errors()
            step_type_val = step.step_type.value if isinstance(step.step_type, Enum) else step.step_type
            await self._emit_event(
                'step_error',
                {'session_id': session_id, 'step_id': step.step_id, 'step_type': step_type_val, 'error': str(e)},
            )

            error_text = f'Error executing step {step.step_id} ({step_type_val}): {e!s}'
            error_msg = Message(
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                parts=[TextPart(text=error_text)],
                contextId=session_id,
                kind='message',
            )
            yield AgentRunOutput(
                id=step.step_id,
                result=error_msg,
                metadata={**step_output_metadata_base, 'error': str(e), 'agent_id': self.id},
            )

    async def _execute_tool_call(self, tool_call_part: ToolUsePart, state: State) -> ToolUsePart:
        """Execute a tool call using the tool system."""
        if not isinstance(tool_call_part, ToolUsePart) or tool_call_part.type != 'call':
            error_msg = "_execute_tool_call expects a ToolUsePart with type='call'"
            logger.error(f'{error_msg}. Received: {type(tool_call_part)} with data: {tool_call_part}')
            call_id_for_error = getattr(tool_call_part, 'id', f'error_call_{uuid.uuid4().hex[:8]}')
            tool_name_for_error = getattr(tool_call_part, 'name', 'unknown_tool')
            return ToolUsePart(
                kind='tool-use',
                type='result',
                id=call_id_for_error,
                name=tool_name_for_error,
                output='',
                metadata={'success': False, 'error': error_msg},
            )

        tool_name = tool_call_part.name
        arguments = tool_call_part.input if isinstance(tool_call_part.input, dict) else {}
        call_id = tool_call_part.id

        try:
            tool = self.get_tool_by_name(tool_name)
            if not tool:
                raise ValueError(f'Tool not found: {tool_name}')

            context = ToolContext(
                agent_id=self.id, session_id=state.session_id, tool_name=tool_name, arguments=arguments
            )

            # Validate input using the tool's schema
            validated_args = tool.validate_input(arguments)

            # Execute the tool
            result = await tool.run(validated_args, context)

            return ToolUsePart(
                kind='tool-use',
                type='result',
                id=call_id,
                name=tool_name,
                output=str(result),
                metadata={'success': True},
            )
        except Exception as e:
            logger.error(f'Tool execution error for {tool_name} (call_id: {call_id}): {e}', exc_info=True)
            return ToolUsePart(
                kind='tool-use',
                type='result',
                id=call_id,
                name=tool_name,
                output='',
                metadata={'success': False, 'error': str(e)},
            )

    async def _emit_event(self, event_type: str, data: dict) -> None:
        """Emit event via mailbox system."""
        try:
            event_message = Message(
                messageId=f'event_{uuid.uuid4().hex[:8]}',
                role=Role.agent,
                parts=[TextPart(text=f'Event: {event_type}')],
                contextId=data.get('session_id', 'system'),
                kind='message',  # Use "message" kind for events
                metadata={'event_type': event_type, 'timestamp': datetime.now().isoformat(), **data},
            )

            # Send event via outbox for streaming to clients/dispatcher
            await self.mailbox.put_outbox(event_message)

        except Exception as e:
            if self.debug:
                logger.warning(f'Failed to emit event {event_type}: {e}')

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self.planner, 'cleanup'):
                await self.planner.cleanup()
            if hasattr(self.llm, 'cleanup'):
                await self.llm.cleanup()
            if hasattr(self.memory, 'cleanup'):
                await self.memory.cleanup()
            if hasattr(self.mailbox, 'cleanup'):
                await self.mailbox.cleanup()
        except Exception as e:
            if self.debug:
                logger.warning(f'Error during cleanup: {e}')
