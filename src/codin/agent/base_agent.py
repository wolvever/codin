import asyncio
import logging
import time
import uuid
import typing as _t
from datetime import datetime

from a2a.types import Message, Role, TextPart

from .base import Agent
from .types import (
    AgentRunInput,
    AgentRunOutput,
    ToolCall,
    ToolCallResult,
    State,
    Step, 
    StepType, 
    MessageStep,
    EventStep,
    ToolCallStep, 
    ThinkStep, 
    FinishStep,
    Task,
    TaskStatus,
    Metrics,
    RunConfig,
    # Backward compatibility
    AgentConfig,
    EventType,
    InternalEvent,
)
from .base import Planner
from ..memory import Memory, MemoryWriter, InMemoryService
from ..artifact import ArtifactService, InMemoryArtifactService
from ..session import SessionService, ReplayService, TaskService
from ..tool.base import Tool, ToolContext
from ..tool.executor import ToolExecutor
from ..tool.registry import ToolRegistry

__all__ = [
    "BaseAgent",
    "ContinueDecision",
]

logger = logging.getLogger("codin.agent.base_agent")


class ContinueDecision:
    """Decision values for continuation logic."""
    CONTINUE = "continue"
    NEED_APPROVAL = "need_approval" 
    FINISH = "finish"
    CANCEL = "cancel"


class BaseAgent(Agent):
    """Stateful agent that orchestrates Planner and services.
    
    This agent follows the new architecture where:
    1. Agent receives AgentRunInput
    2. Gets or creates Session from SessionService  
    3. Starts new Task if needed using TaskService
    4. Builds comprehensive State from Session, Task, and service references
    5. Calls Planner to get Steps (Planner only reads State)
    6. Executes Steps in a loop (messages, tool calls, events)
    7. Updates State based on Step results
    8. Continues until FinishStep or budget constraints
    9. Records to ReplayService and updates Session
    """
    
    def __init__(
        self,
        *,
        name: str = "BaseAgent",
        description: str = "Base agent with comprehensive service orchestration",
        agent_id: str | None = None,
        planner: Planner,
        memory: Memory | None = None,
        memory_writer: MemoryWriter | None = None,
        artifact_service: ArtifactService | None = None,
        session_service: SessionService | None = None,
        replay_service: ReplayService | None = None,
        task_service: TaskService | None = None,
        tool_registry: ToolRegistry | None = None,
        mailbox: _t.Any = None,  # Mailbox interface TBD
        default_config: RunConfig | None = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize the BaseAgent with service injection.
        
        Args:
            name: Agent name
            description: Agent description
            agent_id: Unique agent identifier
            planner: Planner instance for generating execution steps
            memory: Read-only memory service for retrieving history
            memory_writer: Memory writer service for adding messages and creating chunks
            artifact_service: Service for code artifacts and files
            session_service: Service for session lifecycle
            replay_service: Service for execution replay logging
            task_service: Service for task lifecycle management
            tool_registry: Registry of available tools
            mailbox: Inter-agent communication (TBD)
            default_config: Default budget constraints and configuration
            debug: Enable debug logging
        """
        super().__init__(name=name, description=description, **kwargs)
        
        # Core components
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.planner = planner
        
        # Service injection with defaults
        # Use single InMemoryService for both read and write if no specific services provided
        if memory is None and memory_writer is None:
            combined_service = InMemoryService()
            self.memory = combined_service
            self.memory_writer = combined_service
        else:
            self.memory = memory or InMemoryService()
            self.memory_writer = memory_writer or InMemoryService()
        
        self.artifact_service = artifact_service or InMemoryArtifactService()
        self.session_service = session_service or SessionService()
        self.replay_service = replay_service or ReplayService()
        self.task_service = task_service or TaskService()
        self.tool_registry = tool_registry or ToolRegistry()
        self.mailbox = mailbox  # TBD
        
        # Tool execution
        self.tool_executor = ToolExecutor(self.tool_registry)
        
        # Configuration
        self.default_config = default_config or RunConfig(
            turn_budget=100,
            token_budget=100000,
            cost_budget=10.0,
            time_budget=300.0
        )
        self.debug = debug
        
        # Event callbacks for monitoring
        self._event_callbacks: list[_t.Callable[[dict], _t.Awaitable[None]]] = []
        
        logger.info(f"Initialized {self.agent_id} with {len(self.tool_registry.get_tools())} tools")
    
    def add_event_callback(self, callback: _t.Callable[[dict], _t.Awaitable[None]]) -> None:
        """Add event callback for monitoring."""
        self._event_callbacks.append(callback)
    
    async def _emit_event(self, event_type: str, data: dict) -> None:
        """Emit event to all callbacks."""
        event = {"type": event_type, "timestamp": datetime.now().isoformat(), **data}
        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.warning(f"Event callback error: {e}")
    
    async def run(self, input_data: AgentRunInput) -> _t.AsyncGenerator[AgentRunOutput, None]:
        """Main execution method implementing the new architecture."""
        start_time = time.time()
        
        try:
            # 1. Get or create session
            session = await self.session_service.get_or_create_session(input_data.session_id)
            session_id = session["session_id"]
            
            await self._emit_event("run_start", {
                "agent_id": self.agent_id,
                "session_id": session_id,
                "input_message_id": input_data.message.messageId
            })
            
            # 2. Start new task if needed  
            task = await self._start_task_if_needed(input_data, session_id)
            
            # 3. Build comprehensive State
            state = await self._build_state(session, task, input_data)
            
            # 4. Execute planning loop until completion
            async for output in self._execute_planning_loop(state, session_id, start_time):
                yield output
                
        except Exception as e:
            logger.error(f"Error in agent run: {e}", exc_info=True)
            await self._emit_event("run_error", {
                "agent_id": self.agent_id,
                "session_id": input_data.session_id,
                "error": str(e)
            })
            
            # Yield error message
            error_message = Message(
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                parts=[TextPart(text=f"I encountered an error: {str(e)}")],
                contextId=input_data.session_id or "unknown",
                kind="message"
            )
            
            yield AgentRunOutput(
                id=str(uuid.uuid4()),
                result=error_message,
                metadata={"error": str(e), "agent_id": self.agent_id}
            )
    
    async def _start_task_if_needed(
        self, 
        input_data: AgentRunInput, 
        session_id: str
    ) -> Task | None:
        """Start new task if this is a new query."""
        
        # If task_id provided, get existing task
        if input_data.task_id:
            task = await self.task_service.get_task(input_data.task_id)
            if task:
                await self.task_service.start_task(task.id)
                return task
        
        # Extract query from message
        query = ""
        for part in input_data.message.parts:
            if hasattr(part, 'text'):
                query = part.text
                break
        
        if query:
            # Create new task
            task = await self.task_service.create_task(
                query=query,
                metadata={
                    "session_id": session_id,
                    "agent_id": self.agent_id,
                    "input_message_id": input_data.message.messageId
                }
            )
            await self.task_service.start_task(task.id)
            return task
        
        return None
    
    async def _build_state(
        self, 
        session: dict, 
        task: Task | None, 
        input_data: AgentRunInput
    ) -> State:
        """Build comprehensive State from session and services."""
        
        session_id = session["session_id"]
        
        # Get chat history from Memory service
        history = await self.memory.get_history(session_id)
        
        # Add input message to history
        await self.memory_writer.add_message(session_id, input_data.message)
        
        # Get available tools
        tools = self.tool_registry.get_tools()
        
        # Build metrics from session
        metrics = Metrics(
            iterations=session.get("iteration_count", 0),
            tokens_used=session.get("total_tokens", 0),
            cost_used=session.get("total_cost", 0.0),
            elapsed_seconds=session.get("elapsed_time", 0.0)
        )
        
        # Use provided config or default
        config = input_data.options.get("config") if input_data.options else None
        if not isinstance(config, RunConfig):
            config = self.default_config
        
        return State(
            session_id=session_id,
            task_id=task.id if task else None,
            parent_task_id=task.parent_id if task else None,
            agent_id=self.agent_id,
            created_at=session.get("created_at", datetime.now()),
            iteration=session.get("iteration_count", 0),
            memory=self.memory,  # Readonly reference
            memory_writer=self.memory_writer,  # Write reference 
            artifact_ref=self.artifact_service,  # Readonly reference
            tools=tools,
            metrics=metrics,
            config=config,
            current_task=task,
            context=input_data.metadata or {},
            metadata={
                "agent_id": self.agent_id,
                "input_options": input_data.options or {}
            }
        )
    
    async def _execute_planning_loop(
        self, 
        state: State, 
        session_id: str, 
        start_time: float
    ) -> _t.AsyncGenerator[AgentRunOutput, None]:
        """Execute the main planning and execution loop."""
        
        while state.iteration < (state.config.turn_budget or 100):
            elapsed_time = time.time() - start_time
            
            # Check budget constraints
            is_exceeded, reason = state.config.is_budget_exceeded(state.metrics, elapsed_time)
            if is_exceeded:
                logger.warning(f"Budget constraint exceeded: {reason}")
                break
            
            await self._emit_event("turn_start", {
                "session_id": session_id,
                "iteration": state.iteration,
                "metrics": {
                    "tokens_used": state.metrics.tokens_used,
                    "cost_used": state.metrics.cost_used,
                    "elapsed_seconds": elapsed_time
                }
            })
            
            # Get steps from planner (READONLY access to state)
            steps_executed = 0
            task_finished = False
            
            try:
                async for step in self.planner.next(state):
                    steps_executed += 1
                    
                    if self.debug:
                        logger.debug(f"Executing step {step.step_id}: {step.step_type.value}")
                    
                    # Execute the step and yield outputs
                    async for output in self._execute_step(step, state, session_id):
                        yield output
                    
                    # Check if this is a finish step
                    if step.step_type == StepType.FINISH:
                        task_finished = True
                        if state.current_task:
                            success = isinstance(step, FinishStep) and step.success
                            if success:
                                await self.task_service.complete_task(
                                    state.current_task.id,
                                    {"finish_reason": step.reason}
                                )
                            else:
                                await self.task_service.fail_task(
                                    state.current_task.id,
                                    step.reason,
                                    {"finish_reason": step.reason}
                                )
                        
                        await self._emit_event("task_complete", {
                            "session_id": session_id,
                            "task_id": state.task_id,
                            "iteration": state.iteration,
                            "reason": step.reason if isinstance(step, FinishStep) else "Finished",
                            "success": isinstance(step, FinishStep) and step.success
                        })
                        break
                    
                    # Record step for replay
                    await self.replay_service.record_step(session_id, step, "executed")
                    
                    # Limit steps per turn to prevent runaway planners
                    if steps_executed >= 10:
                        logger.warning("Maximum steps per turn reached")
                        break
                
                if task_finished:
                    break
                    
                # Update iteration counter
                state.iteration += 1
                state.metrics.iterations = state.iteration
                
                # Update session with metrics
                await self.session_service.update_session(session_id, {
                    "iteration_count": state.iteration,
                    "total_tokens": state.metrics.tokens_used,
                    "total_cost": state.metrics.cost_used,
                    "elapsed_time": elapsed_time
                })
                
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
                break
    
    async def _execute_step(
        self, 
        step: Step, 
        state: State, 
        session_id: str
    ) -> _t.AsyncGenerator[AgentRunOutput, None]:
        """Execute a single step and yield outputs."""
        
        try:
            if step.step_type == StepType.MESSAGE:
                # Handle message step
                if isinstance(step, MessageStep):
                    # Add message to memory
                    await self.memory_writer.add_message(session_id, step.message)
                    
                    yield AgentRunOutput(
                        id=step.step_id,
                        result=step.message,
                        metadata={
                            "step_type": "message",
                            "is_streaming": step.is_streaming,
                            "agent_id": self.agent_id
                        }
                    )
            
            elif step.step_type == StepType.EVENT:
                # Handle event step
                if isinstance(step, EventStep):
                    yield AgentRunOutput(
                        id=step.step_id,
                        result=step.event,
                        metadata={
                            "step_type": "event", 
                            "event_type": step.event_type.value,
                            "agent_id": self.agent_id
                        }
                    )
            
            elif step.step_type == StepType.TOOL_CALL:
                # Handle tool call step
                if isinstance(step, ToolCallStep):
                    result = await self._execute_tool_call(step.tool_call, state)
                    state.tool_call_results.append(result)
                    state.metrics.increment_tool_calls()
                    
                    # Create result message
                    result_text = f"Tool '{step.tool_call.name}' {'succeeded' if result.success else 'failed'}"
                    if result.output:
                        result_text += f":\n{result.output}"
                    if result.error:
                        result_text += f"\nError: {result.error}"
                    
                    result_message = Message(
                        messageId=str(uuid.uuid4()),
                        role=Role.agent,
                        parts=[TextPart(text=result_text)],
                        contextId=session_id,
                        kind="tool_result"
                    )
                    
                    await self.memory_writer.add_message(session_id, result_message)
                    
                    yield AgentRunOutput(
                        id=step.step_id,
                        result=result_message,
                        metadata={
                            "step_type": "tool_call",
                            "tool_name": step.tool_call.name,
                            "success": result.success,
                            "agent_id": self.agent_id
                        }
                    )
            
            elif step.step_type == StepType.THINK:
                # Handle thinking step (internal, may or may not be yielded)
                if isinstance(step, ThinkStep) and self.debug:
                    # Only emit thinking events in debug mode
                    await self._emit_event("agent_thinking", {
                        "session_id": session_id,
                        "step_id": step.step_id,
                        "thinking": step.thinking
                    })
            
            elif step.step_type == StepType.FINISH:
                # Handle finish step
                if isinstance(step, FinishStep) and step.final_message:
                    await self.memory_writer.add_message(session_id, step.final_message)
                    
                    yield AgentRunOutput(
                        id=step.step_id,
                        result=step.final_message,
                        metadata={
                            "step_type": "finish",
                            "reason": step.reason,
                            "success": step.success,
                            "agent_id": self.agent_id
                        }
                    )
        
        except Exception as e:
            logger.error(f"Error executing step {step.step_id}: {e}", exc_info=True)
            state.metrics.increment_errors()
            
            # Emit error event
            await self._emit_event("step_error", {
                "session_id": session_id,
                "step_id": step.step_id,
                "step_type": step.step_type.value,
                "error": str(e)
            })
    
    async def _execute_tool_call(self, tool_call, state: State) -> ToolCallResult:
        """Execute a tool call using the tool executor."""
        try:
            # Create tool context
            context = ToolContext(
                agent_id=self.agent_id,
                session_id=state.session_id,
                metadata=state.metadata
            )
            
            # Execute tool
            result = await self.tool_executor.execute(
                tool_call.name,
                tool_call.arguments,
                context
            )
            
            return ToolCallResult(
                call_id=tool_call.call_id,
                success=True,
                output=str(result)
            )
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return ToolCallResult(
                call_id=tool_call.call_id,
                success=False,
                output="",
                error=str(e)
            ) 