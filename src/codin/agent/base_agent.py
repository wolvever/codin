import asyncio
import logging
import time
import uuid
import typing as _t
from datetime import datetime
from enum import Enum

from a2a.types import Message, Role, TextPart

from .base import Agent, Planner
from .types import (
    AgentRunInput,
    AgentRunOutput,
    State,
    Step, 
    StepType, 
    MessageStep,
    EventStep,
    ToolCallStep, 
    ThinkStep, 
    FinishStep,
    Task,
    Metrics,
    RunConfig,
    ToolUsePart,
)
from ..memory.base import MemoryService, MemMemoryService
from ..tool.base import Tool, ToolContext
from ..actor.mailbox import Mailbox
from ..model.base import BaseLLM

__all__ = ["BaseAgent", "ContinueDecision"]
logger = logging.getLogger("codin.agent.base_agent")

class ContinueDecision:
    CONTINUE, NEED_APPROVAL, FINISH, CANCEL = "continue", "need_approval", "finish", "cancel"

class BaseAgent(Agent):
    """Simplified agent using codin architecture components.
    
    Uses:
    - prompt_run() for LLM interactions (via Planner)
    - Memory class for conversation history
    - Tools from tool system
    - Mailbox for event communication
    - Sandbox through tools for file I/O
    """
    
    def __init__(
        self,
        *,
        name: str = "BaseAgent",
        description: str = "Agent using codin architecture",
        agent_id: str | None = None,
        planner: Planner,
        memory: MemoryService | None = None,
        tools: list[Tool] | None = None,
        llm: BaseLLM | None = None,
        mailbox: Mailbox | None = None,
        default_config: RunConfig | None = None,
        debug: bool = False,
        **kwargs
    ):
        super().__init__(name=name, description=description, tools=tools or [], **kwargs)
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.planner = planner
        
        # Simple memory - default to in-memory implementation
        self.memory = memory or MemMemoryService()
        
        # Tools from tool system
        self.tools = tools or []
        
        # Optional LLM for direct model calls (alternative to prompt_run)
        self.llm = llm
        
        # Mailbox for events and inter-agent communication
        self.mailbox = mailbox or Mailbox(self.agent_id)
        
        self.default_config = default_config or RunConfig(
            turn_budget=100, token_budget=100000, cost_budget=10.0, time_budget=300.0
        )
        self.debug = debug
        
        logger.info(f"Initialized {self.agent_id} with {len(self.tools)} tools")

    def get_tool_by_name(self, tool_name: str) -> Tool | None:
        """Get a tool by name from the tools list."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    async def run(self, input_data: AgentRunInput) -> _t.AsyncGenerator[AgentRunOutput, None]:
        """Execute agent logic using simplified architecture."""
        start_time = time.time()
        session_id = input_data.session_id or f"session_{uuid.uuid4().hex[:8]}"
        
        try:
            # Build state with simple components
            state = await self._build_state(session_id, input_data)
            
            # Emit start event via mailbox
            await self._emit_event("run_start", {
                "agent_id": self.agent_id, 
                "session_id": session_id,
                "input_message_id": input_data.message.messageId if input_data.message else "N/A"
            })
            
            async for output in self._execute_planning_loop(state, session_id, start_time):
                yield output
                
        except Exception as e:
            logger.error(f"Error in agent run: {e}", exc_info=True)
            await self._emit_event("run_error", {
                "agent_id": self.agent_id, 
                "session_id": session_id, 
                "error": str(e)
            })
            
            error_message = Message(
                messageId=str(uuid.uuid4()), 
                role=Role.agent, 
                parts=[TextPart(text=f"I encountered an error: {str(e)}")],
                contextId=session_id, 
                kind="message"
            )
            yield AgentRunOutput(
                id=str(uuid.uuid4()), 
                result=error_message, 
                metadata={"error": str(e), "agent_id": self.agent_id}
            )
    
    async def _build_state(self, session_id: str, input_data: AgentRunInput) -> State:
        """Build state using simple memory and components."""
        
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
                    referenceTaskIds=input_data.message.referenceTaskIds
                )
                await self.memory.add_message(corrected_message)
            else:
                await self.memory.add_message(input_data.message)
        
        # Get conversation history from memory using session_id
        history = await self.memory.get_history(session_id)
        
        # Simple metrics tracking
        metrics = Metrics(
            iterations=0,
            input_token_used=0,
            output_token_used=0,
            cost_used=0.0,
            time_used=0.0
        )
        
        # Extract config from input
        config_data = input_data.options.get("config") if input_data.options else None
        current_config = self.default_config
        if isinstance(config_data, RunConfig): 
            current_config = config_data
        elif isinstance(config_data, dict): 
            current_config = RunConfig(**config_data)

        # Create task from input message
        task = None
        if input_data.message:
            query = ""
            if input_data.message.parts:
                for part in input_data.message.parts:
                    if isinstance(part, TextPart) and hasattr(part, 'text'): 
                        query = part.text
                        break
            if query:
                task = Task(
                    id=input_data.message.messageId or str(uuid.uuid4()),
                    query=query,
                    metadata={"session_id": session_id, "agent_id": self.agent_id},
                    parts=[]
                )

        return State(
            session_id=session_id, 
            agent_id=self.agent_id, 
            created_at=datetime.now(),
            iteration=0, 
            history=list(history),
            artifact_ref=None,  # File operations through tools/sandbox
            tools=self.tools, 
            metrics=metrics, 
            config=current_config, 
            task=task,
            metadata={
                "agent_id": self.agent_id, 
                "input_options": input_data.options or {},
                "memory": self.memory,  # Provide memory access to planner
                "mailbox": self.mailbox  # Provide mailbox access to planner
            }
        )
    
    async def _execute_planning_loop(self, state: State, session_id: str, start_time: float) -> _t.AsyncGenerator[AgentRunOutput, None]:
        """Execute planning loop with budget constraints."""
        
        while state.iteration < (state.config.turn_budget or 100):
            elapsed_time = time.time() - start_time
            state.metrics.time_used = elapsed_time

            # Check budget constraints
            is_exceeded, reason = state.config.is_budget_exceeded(state.metrics, elapsed_time)
            if is_exceeded:
                logger.warning(f"Budget constraint exceeded: {reason}")
                finish_reason = f"Budget exceeded: {reason}"
                finish_msg = Message(
                    messageId=str(uuid.uuid4()), 
                    role=Role.agent, 
                    parts=[TextPart(text=finish_reason)], 
                    contextId=session_id, 
                    kind="message"
                )
                finish_step = FinishStep(
                    step_id=str(uuid.uuid4()), 
                    reason=finish_reason, 
                    success=False, 
                    final_message=finish_msg
                )
                async for output in self._execute_step(finish_step, state, session_id): 
                    yield output
                break 
            
            await self._emit_event("turn_start", {
                "session_id": session_id, 
                "iteration": state.iteration, 
                "metrics": state.metrics.__dict__
            })
            
            steps_executed = 0
            task_finished_by_step = False
            
            try:
                # Generate steps from planner
                async for step in self.planner.next(state):
                    steps_executed += 1
                    if self.debug: 
                        logger.debug(f"Executing step {step.step_id}: {step.step_type.value if isinstance(step.step_type, Enum) else step.step_type}")
                    
                    # If planner yields a message, add it to memory
                    if step.step_type == StepType.MESSAGE and isinstance(step, MessageStep) and step.message:
                        if not any(h.messageId == step.message.messageId for h in state.history if h.messageId and step.message.messageId):
                            await self.memory.add_message(step.message)
                            state.history.append(step.message)

                    async for output in self._execute_step(step, state, session_id): 
                        yield output
                    
                    if step.step_type == StepType.FINISH:
                        task_finished_by_step = True
                        await self._emit_event("task_complete", {
                            "session_id": session_id, 
                            "iteration": state.iteration, 
                            "reason": step.reason if isinstance(step, FinishStep) and step.reason else "Finished", 
                            "success": isinstance(step, FinishStep) and step.success
                        })
                        break
                    
                    if steps_executed >= 10: 
                        logger.warning("Maximum steps per turn reached from planner")
                        break
                
                if task_finished_by_step: 
                    break
                    
                state.iteration += 1
                
                await self._emit_event("turn_end", {
                    "session_id": session_id, 
                    "iteration": state.iteration, 
                    "steps_executed": steps_executed
                })
                
            except Exception as e:
                logger.error(f"Error in planning loop: {e}", exc_info=True)
                state.metrics.increment_errors()
                await self._emit_event("turn_error", {
                    "session_id": session_id, 
                    "iteration": state.iteration, 
                    "error": str(e)
                })
                
                error_msg = Message(
                    messageId=str(uuid.uuid4()), 
                    role=Role.agent, 
                    parts=[TextPart(text=f"Error during planning: {str(e)}")], 
                    contextId=session_id, 
                    kind="message"
                )
                yield AgentRunOutput(
                    id=str(uuid.uuid4()), 
                    result=error_msg, 
                    metadata={"error": str(e), "agent_id": self.agent_id}
                )
                break
    
    async def _execute_step(self, step: Step, state: State, session_id: str) -> _t.AsyncGenerator[AgentRunOutput, None]:
        """Execute step using codin components."""
        step_output_metadata_base = {"agent_id": self.agent_id, "step_id": step.step_id}
        
        try:
            if step.step_type == StepType.MESSAGE and isinstance(step, MessageStep) and step.message:
                # Message step - add to memory and yield
                await self.memory.add_message(step.message)
                yield AgentRunOutput(
                    id=step.step_id, 
                    result=step.message, 
                    metadata={**step_output_metadata_base, "step_type": "message", "is_streaming": step.is_streaming}
                )
            
            elif step.step_type == StepType.EVENT and isinstance(step, EventStep) and step.event:
                # Event step - emit via mailbox
                await self._emit_event("agent_event", {
                    "session_id": session_id,
                    "step_id": step.step_id,
                    "event": step.event
                })
                yield AgentRunOutput(
                    id=step.step_id, 
                    result=step.event, 
                    metadata={**step_output_metadata_base, "step_type": "event"}
                )
            
            elif step.step_type == StepType.TOOL_CALL and isinstance(step, ToolCallStep) and step.tool_call:
                # Tool call step - execute tool and create result message
                result_tool_use_part = await self._execute_tool_call(step.tool_call, state)
                step.add_result(result_tool_use_part)
                state.metrics.increment_tool_calls()
                
                # Create tool interaction message
                call_part, result_part = step.to_message_parts()
                
                tool_interaction_parts: _t.List[_t.Union[TextPart, ToolUsePart]] = []
                if call_part: 
                    tool_interaction_parts.append(call_part)
                if result_part: 
                    tool_interaction_parts.append(result_part)

                tool_interaction_message = Message(
                    messageId=f"toolmsg_{step.tool_call.id if step.tool_call.id else str(uuid.uuid4())}", 
                    role=Role.agent,
                    parts=tool_interaction_parts,
                    contextId=session_id,
                    kind="message" 
                )
                
                # Add to memory and state
                await self.memory.add_message(tool_interaction_message)
                if not any(h.messageId == tool_interaction_message.messageId for h in state.history if h.messageId and tool_interaction_message.messageId):
                    state.history.append(tool_interaction_message)

                success_from_meta = result_tool_use_part.metadata.get('success', False) if result_tool_use_part.metadata else False
                yield AgentRunOutput(
                    id=step.step_id, 
                    result=tool_interaction_message, 
                    metadata={
                        **step_output_metadata_base, 
                        "step_type": "tool_call", 
                        "tool_name": step.tool_call.name, 
                        "success": success_from_meta
                    }
                )
            
            elif step.step_type == StepType.THINK and isinstance(step, ThinkStep) and self.debug:
                # Think step - emit thinking event
                await self._emit_event("agent_thinking", {
                    "session_id": session_id, 
                    "step_id": step.step_id, 
                    "thinking": step.thinking
                })
            
            elif step.step_type == StepType.FINISH and isinstance(step, FinishStep) and step.final_message:
                # Finish step - add final message to memory
                if not any(h.messageId == step.final_message.messageId for h in state.history if h.messageId and step.final_message.messageId):
                    await self.memory.add_message(step.final_message)
                    state.history.append(step.final_message)
                    
                yield AgentRunOutput(
                    id=step.step_id, 
                    result=step.final_message, 
                    metadata={
                        **step_output_metadata_base, 
                        "step_type": "finish", 
                        "reason": step.reason or "Finished", 
                        "success": step.success
                    }
                )
        
        except Exception as e:
            logger.error(f"Error executing step {step.step_id}: {e}", exc_info=True)
            state.metrics.increment_errors()
            step_type_val = step.step_type.value if isinstance(step.step_type, Enum) else step.step_type
            await self._emit_event("step_error", {
                "session_id": session_id, 
                "step_id": step.step_id, 
                "step_type": step_type_val, 
                "error": str(e)
            })
            
            error_text = f"Error executing step {step.step_id} ({step_type_val}): {str(e)}"
            error_msg = Message(
                messageId=str(uuid.uuid4()), 
                role=Role.agent, 
                parts=[TextPart(text=error_text)], 
                contextId=session_id, 
                kind="message"
            )
            yield AgentRunOutput(
                id=step.step_id, 
                result=error_msg, 
                metadata={**step_output_metadata_base, "error": str(e), "agent_id": self.agent_id}
            )

    async def _execute_tool_call(self, tool_call_part: ToolUsePart, state: State) -> ToolUsePart:
        """Execute a tool call using the tool system."""
        if not isinstance(tool_call_part, ToolUsePart) or tool_call_part.type != 'call':
            error_msg = "_execute_tool_call expects a ToolUsePart with type='call'"
            logger.error(f"{error_msg}. Received: {type(tool_call_part)} with data: {tool_call_part}")
            call_id_for_error = getattr(tool_call_part, 'id', f"error_call_{uuid.uuid4().hex[:8]}")
            tool_name_for_error = getattr(tool_call_part, 'name', 'unknown_tool')
            return ToolUsePart(
                kind='tool-use', type='result', id=call_id_for_error,
                name=tool_name_for_error, output="",
                metadata={'success': False, 'error': error_msg}
            )

        tool_name = tool_call_part.name
        arguments = tool_call_part.input if isinstance(tool_call_part.input, dict) else {}
        call_id = tool_call_part.id

        try:
            tool = self.get_tool_by_name(tool_name)
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")
            
            context = ToolContext(
                agent_id=self.agent_id, 
                session_id=state.session_id,
                tool_name=tool_name, 
                arguments=arguments 
            )
            
            # Validate input using the tool's schema
            validated_args = tool.validate_input(arguments)
            
            # Execute the tool
            result = await tool.run(validated_args, context)
            
            return ToolUsePart(
                kind='tool-use', type='result', id=call_id, name=tool_name,
                output=str(result), metadata={'success': True}
            )
        except Exception as e:
            logger.error(f"Tool execution error for {tool_name} (call_id: {call_id}): {e}", exc_info=True)
            return ToolUsePart(
                kind='tool-use', type='result', id=call_id, name=tool_name, output="",
                metadata={'success': False, 'error': str(e)}
            )

    async def _emit_event(self, event_type: str, data: dict) -> None:
        """Emit event via mailbox system."""
        try:
            event_message = Message(
                messageId=f"event_{uuid.uuid4().hex[:8]}",
                role=Role.agent,
                parts=[TextPart(text=f"Event: {event_type}")],
                contextId=data.get("session_id", "system"),
                kind="message",  # Use "message" kind for events
                metadata={
                    "event_type": event_type,
                    "timestamp": datetime.now().isoformat(),
                    **data
                }
            )
            
            # Send event via mailbox (for inter-agent communication)
            await self.mailbox.send_message(
                event_message,
                recipient_id="system",  # System events
                metadata={"event_type": event_type}
            )
            
        except Exception as e:
            if self.debug:
                logger.warning(f"Failed to emit event {event_type}: {e}") 