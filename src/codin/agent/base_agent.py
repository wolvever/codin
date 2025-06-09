"""BaseAgent: A concrete actor implementation for A2A message processing.

This module defines `BaseAgent`, which inherits from `AgentActor` and specializes
in handling A2A (Agent-to-Agent) messages. It uses a planner-driven execution
loop to process these messages.
"""

import asyncio
import logging
import time
import typing as _t
import uuid
from datetime import datetime
from enum import Enum
from pydantic import ValidationError # For input validation

# Actor system imports
from ..actor.types import ActorRunInput, ActorRunOutput
from ..actor.actors import AgentActor # Inherit from AgentActor
from ..actor.envelope_types import EnvelopeKind

# Agent-specific data structures and control types (used for internal logic)
from .types import (
    ControlSignal, EventStep, FinishStep, Message as A2AMessage, # Renamed Message to A2AMessage for clarity
    MessageStep, Metrics, Role, RunConfig, RunnerControl, State, Step, StepType, Task,
    TaskState, TaskStatus, TextPart, ThinkStep, ToolCallStep, ToolUsePart,
)

from ..actor.mailbox import LocalMailbox, Mailbox
from ..memory.base import MemMemoryService, Memory
from ..model.base import BaseLLM
from ..tool.base import Tool, ToolContext
from .base import Planner # BaseAgent uses a Planner
from ..id import new_id # For generating default agent ID

__all__ = ["BaseAgent", "ContinueDecision"] # ContinueDecision might be unused now or refactored
logger = logging.getLogger("codin.agent.base_agent")


class ContinueDecision: # This might be part of the planner's specific step outputs, review if needed.
    """Enumerates decisions for an agent's continuation logic."""
    CONTINUE, NEED_APPROVAL, FINISH, CANCEL = "continue", "need_approval", "finish", "cancel"


class BaseAgent(AgentActor):
    """A specialized `AgentActor` for handling A2A messages via a planner-driven loop.

    `BaseAgent` implements the `run` method to process `ActorRunInput`. If the
    input's `envelope_kind` is `A2A_MESSAGE`, it deserializes the payload into
    an `A2AMessage`, builds an internal `State`, and then uses its `Planner`
    to execute a series of `Step`s. Other envelope kinds like `A2A_TASK` or
    `CONTROL` are handled with placeholder logic.

    It inherits `capability` (AGENT_CAP) from `AgentActor`.
    """

    def __init__(
        self,
        *,
        agent_id: str | None = None, # This will be the actor_id
        name: str = "BaseAgent",
        description: str = "A planner-driven agent for A2A messages.",
        version: str = "1.0.0",
        planner: Planner,
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        llm: BaseLLM | None = None,
        mailbox: Mailbox | None = None,
        default_config: RunConfig | None = None,
        debug: bool = False,
        # **kwargs: Any # From AgentActor's __init__ if it were to take more
    ):
        """Initializes the `BaseAgent`.

        Args:
            agent_id: Unique ID for this agent/actor. If None, an ID is generated.
            name: Name of the agent.
            description: Description of the agent.
            version: Agent version.
            planner: The `Planner` instance for decision-making.
            tools: List of `Tool` instances available to the agent.
            memory: `Memory` instance for history and context. Defaults to `MemMemoryService`.
            llm: Optional `BaseLLM` instance, may be used by the planner.
            mailbox: `Mailbox` for communication. Defaults to `LocalMailbox`.
            default_config: Default `RunConfig` for execution parameters.
            debug: Enable debug logging.
        """
        # Initialize attributes specific to BaseAgent's operation
        self.id = agent_id or new_id(prefix=name) # Used for actor_id
        self.name: str = name
        self.description: str = description
        self.version: str = version
        self.tools: list[Tool] = tools or []

        # Call super().__init__ from AgentActor, providing the actor_id
        super().__init__(actor_id=self.id) # Removed **kwargs as AgentActor doesn't use them meaningfully

        self.planner: Planner = planner
        self.memory: Memory = memory or MemMemoryService()
        self.llm: BaseLLM | None = llm
        self.mailbox: Mailbox = mailbox or LocalMailbox()
        self.default_config: RunConfig = default_config or RunConfig()
        self.debug: bool = debug

        self._paused: bool = False
        self._cancelled: bool = False

        logger.info(f"Initialized BaseAgent {self.id} (actor_id: {self.actor_id}) with {len(self.tools)} tools.")


    async def run(self, input_data: ActorRunInput) -> _t.AsyncIterator[ActorRunOutput]:
        """Processes an `ActorRunInput` based on its `envelope_kind`.

        If `A2A_MESSAGE`, it deserializes the payload, builds state, and runs a planning loop.
        Other kinds (`A2A_TASK`, `CONTROL`) have placeholder handlers.

        Args:
            input_data: The `ActorRunInput` from the dispatcher.

        Yields:
            `ActorRunOutput`: Results or status messages.
        """
        logger.info(f"BaseAgent {self.actor_id} received run input for request {input_data.request_id}, kind: {input_data.envelope_kind}.")
        session_id = input_data.context_id or f"session_{input_data.request_id}_{uuid.uuid4().hex[:4]}"
        start_time = time.time()

        if input_data.envelope_kind == EnvelopeKind.A2A_MESSAGE:
            if not isinstance(input_data.payload, dict):
                logger.error(f"BaseAgent {self.actor_id}: A2A_MESSAGE payload is not a dict for request {input_data.request_id}.")
                yield {"error": "A2A_MESSAGE payload must be a dictionary", "request_id": input_data.request_id}
                return
            try:
                a2a_message = A2AMessage.model_validate(input_data.payload)
                logger.info(f"BaseAgent {self.actor_id}: Successfully parsed A2A_MESSAGE for request {input_data.request_id} (session: {session_id}).")

                state = await self._build_state_for_a2a_message(session_id, a2a_message, input_data.metadata, input_data.request_id)

                await self._emit_event("run_start", {
                    "agent_id": self.actor_id, "session_id": session_id,
                    "input_message_id": a2a_message.messageId, "request_id": input_data.request_id
                })

                async for output_item in self._execute_planning_loop(state, session_id, start_time):
                    yield output_item

            except ValidationError as e:
                logger.error(f"BaseAgent {self.actor_id}: A2A_MESSAGE payload validation failed for request {input_data.request_id}: {e}", exc_info=True)
                yield {"error": "A2A_MESSAGE payload validation failed", "details": e.errors(), "request_id": input_data.request_id}
            except Exception as e: # Catch other errors during A2A_MESSAGE processing
                logger.error(f"BaseAgent {self.actor_id}: Error processing A2A_MESSAGE for request {input_data.request_id}: {e}", exc_info=True)
                yield {"error": f"Error processing A2A_MESSAGE: {str(e)}", "request_id": input_data.request_id}

        elif input_data.envelope_kind == EnvelopeKind.A2A_TASK:
            logger.info(f"BaseAgent {self.actor_id}: Received A2A_TASK (placeholder) for request {input_data.request_id}")
            # TODO: Implement A2A_TASK deserialization and processing logic.
            #       This might involve a different state-building and execution flow.
            yield {
                "status": f"BaseAgent {self.actor_id} received A2A_TASK (placeholder processing)",
                "request_id": input_data.request_id,
                "task_payload_preview": str(input_data.payload)[:200]
            }

        elif input_data.envelope_kind == EnvelopeKind.CONTROL:
            logger.info(f"BaseAgent {self.actor_id}: Received CONTROL in 'run' (placeholder) for request {input_data.request_id}, payload: {input_data.payload}")
            # Note: Control signals are typically handled by dispatcher or dedicated actor methods,
            # but an actor's 'run' could be invoked with CONTROL if part of a workflow.
            # TODO: Define how BaseAgent specifically reacts to CONTROL signals via its run method, if at all.
            yield {
                "status": f"BaseAgent {self.actor_id} acknowledged CONTROL signal in run method (placeholder)",
                "request_id": input_data.request_id,
                "control_info": input_data.payload
            }
        else:
            logger.warning(f"BaseAgent {self.actor_id}: Received unhandled envelope kind '{input_data.envelope_kind}' in run method for request {input_data.request_id}")
            yield { "error": f"BaseAgent {self.actor_id} cannot handle kind '{input_data.envelope_kind}' in its run method", "request_id": input_data.request_id}

        # Ensure cleanup is called if this run method was the main entry point for a self-contained execution.
        # However, cleanup is usually managed by the supervisor when the actor is released.
        # If this 'run' represents a single, complete task after which the agent might be idle,
        # then a specific kind of "task_end" event might be emitted here.
        # The existing _execute_planning_loop already emits task_end.
        # The superclass AgentActor.cleanup() will be called by supervisor.
        logger.info(f"BaseAgent {self.actor_id} finished run method for request {input_data.request_id}, session {session_id}.")
        # Final _emit_event for run_end is handled by the original AgentRunInput logic, which is now inside the A2A_MESSAGE block.
        # This might need to be generalized if other kinds also have a similar "run_end" concept.

    async def _build_state_for_a2a_message(
        self, session_id: str, input_message: A2AMessage, actor_run_metadata: dict, request_id: str
    ) -> State:
        """Builds the initial `State` for an agent run specifically for an `A2AMessage`.

        Args:
            session_id: The unique ID for the current session.
            input_message: The deserialized `A2AMessage` object from the input payload.
            actor_run_metadata: Metadata from `ActorRunInput.metadata`. Used for `RunConfig`.
            request_id: The `request_id` from `ActorRunInput`.

        Returns:
            An initialized `State` object.
        """
        if isinstance(self.memory, MemMemoryService):
            self.memory.set_session_id(session_id)

        if input_message:
            # Ensure message has correct contextId for this session
            # This check might be redundant if contextId is already session_id from dispatcher
            current_message = input_message
            if input_message.contextId != session_id:
                logger.warning(f"BaseAgent {self.actor_id}: Input message contextId '{input_message.contextId}' differs from session_id '{session_id}'. Overwriting message contextId.")
                current_message = input_message.model_copy(update={"contextId": session_id})
            await self.memory.add_message(current_message)
            logger.debug(f"BaseAgent {self.actor_id}: Added input message to memory for request {request_id}.")
        else: # Should not happen if called for A2A_MESSAGE kind with validated message
            logger.warning(f"BaseAgent {self.actor_id}: _build_state_for_a2a_message called with no input_message for request {request_id}.")


        history = await self.memory.get_history()
        metrics = Metrics()

        config_from_metadata = actor_run_metadata.get("config")
        current_config = self.default_config
        if isinstance(config_from_metadata, RunConfig):
            current_config = config_from_metadata
        elif isinstance(config_from_metadata, dict):
            current_config = RunConfig(**config_from_metadata)

        task = None
        if input_message: # input_message is now guaranteed to be an A2AMessage
            query = next((part.text for part in input_message.parts if isinstance(part, TextPart) and hasattr(part, "text")), "")
            if query:
                task = Task(
                    id=input_message.messageId or str(uuid.uuid4()), query=query, contextId=session_id,
                    status=TaskStatus(state=TaskState.submitted),
                    metadata={"session_id": session_id, "agent_id": self.actor_id, "request_id": request_id}, parts=[]
                )

        initial_metadata = {
            "agent_id": self.actor_id, "request_id": request_id,
            "memory_type": type(self.memory).__name__, "mailbox_type": type(self.mailbox).__name__,
            **(actor_run_metadata or {}),
        }
        if current_config != self.default_config: initial_metadata['config_source'] = 'runtime_metadata'
        else: initial_metadata['config_source'] = 'agent_default'
        initial_metadata.pop('config', None)

        return State(
            session_id=session_id, agent_id=self.actor_id, created_at=datetime.now(),
            history=list(history), tools=self.tools, metrics=metrics, config=current_config,
            task=task, metadata=initial_metadata
        )

    # Methods like _execute_planning_loop, _execute_step, _execute_tool_call, _emit_event
    # are kept from the original BaseAgent as they define its core A2A message processing logic.
    # Their docstrings and logging were updated in the previous step.
    # Their yield types are compatible with ActorRunOutput = Any.

    async def _execute_planning_loop(self, state: State, session_id: str, start_time: float) -> _t.AsyncIterator[ActorRunOutput]:
        # ... (implementation from previous version, with yields of Message/dict)
        await self._emit_event("task_start", {"session_id": session_id, "iteration": state.iteration, "elapsed_time": 0.0, "request_id": state.metadata.get("request_id")})
        while state.iteration < (state.config.turn_budget or 100):
            if not await self.check_inbox_for_control() or self._cancelled: logger.info(f"Agent {self.id} ({self.name}): Planning loop interrupted for session {session_id}."); break
            if self._paused: await self.wait_while_paused()
            if self._cancelled: logger.info(f"Agent {self.id} ({self.name}): Planning loop broken post-pause for session {session_id}."); break
            elapsed_time = time.time() - start_time; state.metrics.time_used = elapsed_time
            is_exceeded, reason = state.config.is_budget_exceeded(state.metrics, elapsed_time)
            if is_exceeded:
                logger.warning(f"Agent {self.id} ({self.name}): Budget constraint exceeded for session {session_id}: {reason}")
                finish_msg = A2AMessage(messageId=str(uuid.uuid4()),role=Role.agent,parts=[TextPart(text=f"Budget exceeded: {reason}")],contextId=session_id,kind="message")
                finish_step = FinishStep(step_id=str(uuid.uuid4()), reason=f"Budget exceeded: {reason}", final_message=finish_msg)
                await self.mailbox.put_outbox(finish_msg); async for output in self._execute_step(finish_step, state, session_id): yield output
                break
            await self._emit_event("turn_start", {"session_id": session_id, "iteration": state.iteration, "metrics": state.metrics.model_dump(), "request_id": state.metadata.get("request_id")})
            steps_executed = 0; task_finished_by_step = False
            try:
                async for step in self.planner.next(state):
                    if not await self.check_inbox_for_control() or self._cancelled: break
                    if self._paused: await self.wait_while_paused()
                    if self._cancelled: break
                    steps_executed += 1
                    if self.debug: logger.debug(f"Agent {self.id} ({self.name}): Executing step {step.step_id} ({step.step_type.value if isinstance(step.step_type, Enum) else step.step_type}) for session {session_id}")
                    if step.step_type == StepType.MESSAGE and isinstance(step, MessageStep) and step.message:
                        if not any(h.messageId == step.message.messageId for h in state.history if h.messageId and step.message.messageId):
                            await self.memory.add_message(step.message); state.history.append(step.message)
                    async for output in self._execute_step(step, state, session_id): yield output
                    if step.step_type == StepType.FINISH:
                        task_finished_by_step = True
                        await self._emit_event("task_complete", {"session_id": session_id, "iteration": state.iteration, "reason": step.reason if isinstance(step, FinishStep) and step.reason else "Finished", "request_id": state.metadata.get("request_id")})
                        break
                    if steps_executed >= (state.config.max_steps_per_turn or 10): logger.warning(f"Agent {self.id} ({self.name}): Max steps per turn ({state.config.max_steps_per_turn or 10}) reached for session {session_id}."); break
                if task_finished_by_step or self._cancelled: break
                state.iteration += 1
                await self._emit_event("turn_end", {"session_id": session_id, "iteration": state.iteration, "steps_executed": steps_executed, "request_id": state.metadata.get("request_id")})
            except Exception as e:
                logger.error(f"Agent {self.id} ({self.name}): Error in planning loop for session {session_id}: {e}", exc_info=True); state.metrics.increment_errors()
                await self._emit_event("turn_error", {"session_id": session_id, "iteration": state.iteration, "error": str(e), "request_id": state.metadata.get("request_id")})
                error_msg = A2AMessage(messageId=str(uuid.uuid4()),role=Role.agent,parts=[TextPart(text=f"Error during planning: {e!s}")],contextId=session_id,kind="message")
                await self.mailbox.put_outbox(error_msg); yield error_msg
                break
        await self._emit_event("task_end", {"session_id": session_id, "iteration": state.iteration, "elapsed_time": time.time() - start_time, "request_id": state.metadata.get("request_id")})

    async def _execute_step(self, step: Step, state: State, session_id: str) -> _t.AsyncIterator[ActorRunOutput]:
        # ... (implementation from previous version, with yields of Message/dict)
        step_output_metadata_base = {"agent_id": self.id, "step_id": step.step_id, "request_id": state.metadata.get("request_id")}
        try:
            if step.step_type == StepType.MESSAGE and isinstance(step, MessageStep):
                if step.is_streaming and step.message_stream is not None:
                    collected_chunks: list[str] = []
                    async for chunk in step.stream_content():
                        collected_chunks.append(chunk)
                        stream_msg = A2AMessage(messageId=f"{step.step_id}-stream-chunk-{len(collected_chunks)}",role=Role.agent,parts=[TextPart(text=chunk)],contextId=session_id,kind="message",metadata={"stream_chunk": True, "step_id": step.step_id})
                        await self.mailbox.put_outbox(stream_msg); yield stream_msg
                    if step.message is None and collected_chunks: step.message = A2AMessage(messageId=step.step_id, role=Role.agent, parts=[TextPart(text="".join(collected_chunks))], contextId=session_id, kind="message", metadata={"aggregated_stream": True})
                    if step.message: await self.memory.add_message(step.message); await self.mailbox.put_outbox(step.message); yield step.message
                elif step.message: await self.memory.add_message(step.message); await self.mailbox.put_outbox(step.message); yield step.message
            elif step.step_type == StepType.EVENT and isinstance(step, EventStep) and step.event:
                event_data_for_emit = {"session_id": session_id, "step_id": step.step_id, "event": step.event, "request_id": state.metadata.get("request_id")}
                await self._emit_event("agent_event", event_data_for_emit)
                event_msg_text = f"Event: {step.event.event_type if hasattr(step.event, 'event_type') else str(step.event)}"
                event_msg = A2AMessage(messageId=str(uuid.uuid4()),role=Role.agent,parts=[TextPart(text=event_msg_text)],contextId=session_id,kind="message",metadata={"event_details": event_data_for_emit, "step_id": step.step_id})
                await self.mailbox.put_outbox(event_msg); yield {"event_type": "agent_event", "details": step.event.model_dump() if hasattr(step.event, 'model_dump') else str(step.event), **step_output_metadata_base}
            elif step.step_type == StepType.TOOL_CALL and isinstance(step, ToolCallStep) and step.tool_call:
                result_tool_use_part = await self._execute_tool_call(step.tool_call, state); step.tool_call_result = result_tool_use_part; state.metrics.increment_tool_calls()
                tool_interaction_parts: list[_t.Union[TextPart, ToolUsePart]] = [step.tool_call]
                if result_tool_use_part: tool_interaction_parts.append(result_tool_use_part)
                tool_interaction_message = A2AMessage(messageId=f"toolmsg_{step.tool_call.id if step.tool_call.id else str(uuid.uuid4())}",role=Role.agent,parts=tool_interaction_parts,contextId=session_id,kind="message")
                await self.memory.add_message(tool_interaction_message); await self.mailbox.put_outbox(tool_interaction_message)
                if not any(h.messageId == tool_interaction_message.messageId for h in state.history if h.messageId and tool_interaction_message.messageId): state.history.append(tool_interaction_message)
                yield tool_interaction_message
            elif step.step_type == StepType.THINK and isinstance(step, ThinkStep) and self.debug:
                await self._emit_event("agent_thinking", {"session_id": session_id, "step_id": step.step_id, "thinking": step.thinking, "request_id": state.metadata.get("request_id")})
            elif step.step_type == StepType.FINISH and isinstance(step, FinishStep) and step.final_message:
                if not any(h.messageId == step.final_message.messageId for h in state.history if h.messageId and step.final_message.messageId):
                    await self.memory.add_message(step.final_message); state.history.append(step.final_message)
                await self.mailbox.put_outbox(step.final_message); yield step.final_message
        except Exception as e:
            logger.error(f"Agent {self.id} ({self.name}): Error executing step {step.step_id} for session {session_id}: {e}", exc_info=True); state.metrics.increment_errors()
            step_type_val = step.step_type.value if isinstance(step.step_type, Enum) else str(step.step_type)
            await self._emit_event("step_error", {"session_id": session_id, "step_id": step.step_id, "step_type": step_type_val, "error": str(e), "request_id": state.metadata.get("request_id")})
            error_text = f"Error executing step {step.step_id} ({step_type_val}): {e!s}"
            error_msg = A2AMessage(messageId=str(uuid.uuid4()),role=Role.agent,parts=[TextPart(text=error_text)],contextId=session_id,kind="message")
            yield error_msg

    async def _execute_tool_call(self, tool_call_part: ToolUsePart, state: State) -> ToolUsePart:
        # ... (implementation from previous version)
        if not isinstance(tool_call_part, ToolUsePart) or tool_call_part.type != "call":
            error_msg = "_execute_tool_call expects a ToolUsePart with type='call'"
            logger.error(f"Agent {self.id}: {error_msg}. Received: {type(tool_call_part)} with data: {tool_call_part}")
            call_id_for_error = getattr(tool_call_part, "id", f"error_call_{uuid.uuid4().hex[:8]}")
            tool_name_for_error = getattr(tool_call_part, "name", "unknown_tool")
            return ToolUsePart(kind="tool-use",type="result",id=call_id_for_error,name=tool_name_for_error,output=f"Error: {error_msg}",metadata={"success": False, "error": error_msg})
        tool_name = tool_call_part.name
        arguments = tool_call_part.input if isinstance(tool_call_part.input, dict) else {}
        call_id = tool_call_part.id
        try:
            tool = self.get_tool_by_name(tool_name)
            if not tool: raise ValueError(f"Tool not found: {tool_name}")
            context = ToolContext(agent_id=self.id, session_id=state.session_id, tool_name=tool_name, arguments=arguments, request_id=state.metadata.get("request_id"))
            validated_args = tool.validate_input(arguments)
            result_output = await tool.run(validated_args, context)
            return ToolUsePart(kind="tool-use",type="result",id=call_id,name=tool_name,output=str(result_output),metadata={"success": True})
        except Exception as e:
            logger.error(f"Agent {self.id}: Tool execution error for {tool_name} (call_id: {call_id}, session: {state.session_id}): {e}", exc_info=True)
            return ToolUsePart(kind="tool-use",type="result",id=call_id,name=tool_name,output=f"Error: {str(e)}",metadata={"success": False, "error": str(e)})

    async def _emit_event(self, event_type: str, data: dict) -> None:
        # ... (implementation from previous version)
        try:
            serializable_data = {k: (v if isinstance(v, (str, int, float, bool, list, dict)) else str(v)) for k, v in data.items()}
            event_message = A2AMessage(messageId=f"event_{uuid.uuid4().hex[:4]}_{event_type}",role=Role.agent,parts=[TextPart(text=f"Event: {event_type}")],contextId=data.get("session_id", self.id),kind="event",metadata={"event_type": event_type, "timestamp": datetime.now().isoformat(), **serializable_data})
            await self.mailbox.put_outbox(event_message)
        except Exception as e:
            logger.warning(f"Agent {self.id}: Failed to emit event {event_type} for session {data.get('session_id', 'N/A')}: {e}", exc_info=self.debug)

    # cleanup method is inherited from AgentActor if not overridden.
    # If BaseAgent has specific cleanup beyond AgentActor, it can override it.
    # For now, assuming AgentActor.cleanup() is sufficient or BaseAgent implements its own.
    # The Agent ABC required it, so BaseAgent should have it.
    async def cleanup(self) -> None:
        """Cleans up resources used by the `BaseAgent` instance."""
        logger.info(f"BaseAgent {self.id} ({self.name}) starting cleanup (actor_id: {self.actor_id}).")
        try:
            if hasattr(self.planner, "cleanup") and callable(self.planner.cleanup):
                await self.planner.cleanup()
            if hasattr(self.memory, "cleanup") and callable(self.memory.cleanup):
                await self.memory.cleanup()
            if hasattr(self.mailbox, "cleanup") and callable(self.mailbox.cleanup):
                await self.mailbox.cleanup()
            # Call superclass cleanup if AgentActor has one and it's not just pass
            # await super().cleanup() # If AgentActor had a meaningful cleanup
            logger.info(f"BaseAgent {self.id} ({self.name}) finished cleanup successfully.")
        except Exception as e:
            logger.error(f"Error during BaseAgent {self.id} ({self.name}) cleanup: {e}", exc_info=True)
            if self.debug: raise

