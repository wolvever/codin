import asyncio
import logging
import time
import uuid
import typing as _t
from datetime import datetime

from a2a.types import Message, Role, TextPart

from .base import Agent
from .types import AgentRunInput, AgentRunOutput, ToolCallResult
from .planner import Planner
from .types import Step, StepType, ThinkStep, MessageStep, ToolCallStep, FinishStep, State
from .session import Session, SessionManager
from ..tool.base import Tool, ToolContext
from ..tool.executor import ToolExecutor
from ..tool.registry import ToolRegistry
from ..memory.base import MemorySystem, InMemoryStore

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
    """Base agent that orchestrates Session, State, Planner and execution loop.
    
    This agent follows the new architecture where:
    1. Agent receives AgentRunInput
    2. Creates or gets Session from SessionManager
    3. Builds State from Session
    4. Calls Planner to get Steps
    5. Executes steps in a loop (messages, tool calls, etc.)
    6. Continues until FinishStep or stopping conditions
    """
    
    def __init__(
        self,
        *,
        name: str = "BaseAgent",
        description: str = "Base agent with session management and planner execution",
        planner: Planner,
        session_manager: SessionManager | None = None,
        tool_registry: ToolRegistry | None = None,
        max_turns: int = 100,
        max_execution_time: float = 300.0,  # 5 minutes
        debug: bool = False,
        **kwargs
    ):
        """Initialize the BaseAgent.
        
        Args:
            name: Agent name
            description: Agent description  
            planner: Planner instance for generating execution steps
            session_manager: Manager for session lifecycle
            tool_registry: Registry of available tools
            max_turns: Maximum number of planner turns
            max_execution_time: Maximum execution time in seconds
            debug: Enable debug logging
        """
        super().__init__(name=name, description=description, **kwargs)
        
        # Core components
        self.planner = planner
        self.session_manager = session_manager or SessionManager()
        self.tool_registry = tool_registry or ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)
        
        # Configuration
        self.max_turns = max_turns
        self.max_execution_time = max_execution_time
        self.debug = debug
        
        # Event callbacks for monitoring
        self._event_callbacks: list[_t.Callable[[dict], _t.Awaitable[None]]] = []
        
        logger.info(f"Initialized {name} with {len(self.tool_registry.get_tools())} tools")
    
    def add_event_callback(self, callback: _t.Callable[[dict], _t.Awaitable[None]]) -> None:
        """Add an event callback for monitoring agent execution."""
        self._event_callbacks.append(callback)
    
    async def _emit_event(self, event_type: str, data: dict[str, _t.Any]) -> None:
        """Emit an event to all registered callbacks."""
        event = {
            "event_type": event_type,
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(),
            "data": data
        }
        
        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    async def run(self, input_data: AgentRunInput) -> AgentRunOutput:
        """Execute the agent with the new session/planner architecture."""
        start_time = time.time()
        
        # Get or create session
        session_id = input_data.session_id or str(uuid.uuid4())
        session = await self.session_manager.get_or_create_session(session_id)
        
        # Record the input message
        await session.record(input_data.message)
        
        await self._emit_event("task_start", {
            "session_id": session_id,
            "input_message": self._extract_text_from_message(input_data.message)
        })
        
        try:
            # Main execution loop
            turn_count = 0
            final_message = None
            
            while turn_count < self.max_turns:
                elapsed_time = time.time() - start_time
                
                # Check execution time limit
                if elapsed_time > self.max_execution_time:
                    logger.warning(f"Execution time limit reached: {elapsed_time:.1f}s")
                    break
                
                # Build current state from session
                state = session.build_state()
                state.turn_count = turn_count
                
                await self._emit_event("turn_start", {
                    "session_id": session_id,
                    "turn": turn_count,
                    "state_summary": {
                        "message_count": len(state.history),
                        "task_completed": len(state.task_list.get("completed", [])),
                        "task_pending": len(state.task_list.get("pending", []))
                    }
                })
                
                # Get steps from planner
                steps_executed = 0
                task_finished = False
                
                try:
                    async for step in self.planner.next(state):
                        steps_executed += 1
                        
                        if self.debug:
                            logger.debug(f"Executing step {step.step_id}: {step.step_type.value}")
                        
                        # Execute the step
                        step_result = await self._execute_step(step, session, state)
                        
                        # Check if this is a finish step
                        if step.step_type == StepType.FINISH:
                            final_message = step.final_message
                            task_finished = True
                            await self._emit_event("task_complete", {
                                "session_id": session_id,
                                "turn": turn_count,
                                "reason": step.reason if isinstance(step, FinishStep) else "Finished"
                            })
                            break
                        
                        # Update session from any state changes
                        session.update_from_state(state)
                        
                        # Limit steps per turn to prevent runaway planners
                        if steps_executed >= 10:
                            logger.warning("Maximum steps per turn reached")
                            break
                
                except Exception as e:
                    logger.error(f"Error in planner execution: {e}")
                    await self._emit_event("turn_error", {
                        "session_id": session_id,
                        "turn": turn_count,
                        "error": str(e)
                    })
                    break
                
                await self._emit_event("turn_end", {
                    "session_id": session_id,
                    "turn": turn_count,
                    "steps_executed": steps_executed,
                    "task_finished": task_finished
                })
                
                # Exit if task is finished
                if task_finished:
                    break
                
                turn_count += 1
                session.turn_count = turn_count
            
            # Get final response message
            if not final_message:
                # Get the last assistant message from session
                for msg in reversed(session.messages):
                    if msg.role == Role.agent:
                        final_message = msg
                        break
                
                if not final_message:
                    # Create a default completion message
                    final_message = Message(
                        messageId=str(uuid.uuid4()),
                        role=Role.agent,
                        parts=[TextPart(text="Task completed.")],
                        contextId=session_id,
                        kind="message"
                    )
            
            # Update final metrics
            elapsed_time = time.time() - start_time
            session.metrics.update({
                "total_turns": turn_count,
                "execution_time": elapsed_time,
                "completed_at": datetime.now().timestamp()
            })
            
            return AgentRunOutput(
                result=final_message,
                metadata={
                    "session_id": session_id,
                    "turns": turn_count,
                    "execution_time": elapsed_time,
                    "session_metrics": session.get_metrics_summary()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in agent execution: {e}", exc_info=True)
            await self._emit_event("task_error", {
                "session_id": session_id,
                "error": str(e)
            })
            
            # Create error response
            error_message = Message(
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                parts=[TextPart(text=f"I encountered an error: {str(e)}")],
                contextId=session_id,
                kind="message"
            )
            
            return AgentRunOutput(
                result=error_message,
                metadata={
                    "session_id": session_id,
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }
            )
    
    async def _execute_step(self, step: Step, session: Session, state: State) -> _t.Any:
        """Execute a single step and update session/state accordingly."""
        
        if step.step_type == StepType.THINK:
            # Handle thinking step
            think_step = _t.cast(ThinkStep, step)
            await self._emit_event("agent_thinking", {
                "session_id": session.session_id,
                "thinking": think_step.thinking
            })
            # Thinking doesn't produce messages, just internal processing
            return None
            
        elif step.step_type == StepType.MESSAGE:
            # Handle message step
            message_step = _t.cast(MessageStep, step)
            await session.record(message_step.message)
            state.add_message(message_step.message)
            
            await self._emit_event("agent_message", {
                "session_id": session.session_id,
                "message_id": message_step.message.messageId,
                "is_streaming": message_step.is_streaming
            })
            return message_step.message
            
        elif step.step_type == StepType.TOOL_CALL:
            # Handle tool call step
            tool_step = _t.cast(ToolCallStep, step)
            return await self._execute_tool_call(tool_step, session, state)
            
        elif step.step_type == StepType.FINISH:
            # Handle finish step
            finish_step = _t.cast(FinishStep, step)
            if finish_step.final_message:
                await session.record(finish_step.final_message)
                state.add_message(finish_step.final_message)
            return finish_step
            
        else:
            logger.warning(f"Unknown step type: {step.step_type}")
            return None
    
    async def _execute_tool_call(self, tool_step: ToolCallStep, session: Session, state: State) -> ToolCallResult:
        """Execute a tool call and record results."""
        tool_call = tool_step.tool_call
        
        await self._emit_event("tool_call_start", {
            "session_id": session.session_id,
            "call_id": tool_call.call_id,
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments
        })
        
        try:
            # Check if tool exists
            tool = self.tool_registry.get_tool(tool_call.name)
            if tool is None:
                result = ToolCallResult(
                    call_id=tool_call.call_id,
                    success=False,
                    output="",
                    error=f"Unknown tool: {tool_call.name}"
                )
            else:
                # Execute the tool
                tool_context = ToolContext(
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    session_id=session.session_id
                )
                
                result_data = await self.tool_executor.execute(
                    tool_call.name, 
                    tool_call.arguments, 
                    tool_context
                )
                
                output = str(result_data) if result_data is not None else ""
                
                result = ToolCallResult(
                    call_id=tool_call.call_id,
                    success=True,
                    output=output
                )
            
            # Update metrics
            session.metrics["total_tool_calls"] = session.metrics.get("total_tool_calls", 0) + 1
            state.update_metrics("total_tool_calls", session.metrics["total_tool_calls"])
            
            # Store last tool results in context
            if not state.last_tool_results:
                state.last_tool_results = []
            state.last_tool_results.append(result)
            state.context["last_tool_results"] = state.last_tool_results
            
            await self._emit_event("tool_call_end", {
                "session_id": session.session_id,
                "call_id": tool_call.call_id,
                "tool_name": tool_call.name,
                "success": result.success,
                "output": result.output,
                "error": result.error
            })
            
            return result
            
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            logger.error(f"Error executing tool {tool_call.name}: {e}", exc_info=True)
            
            result = ToolCallResult(
                call_id=tool_call.call_id,
                success=False,
                output="",
                error=error_msg
            )
            
            await self._emit_event("tool_call_end", {
                "session_id": session.session_id,
                "call_id": tool_call.call_id,
                "tool_name": tool_call.name,
                "success": False,
                "error": error_msg
            })
            
            return result
    
    def _extract_text_from_message(self, message: Message) -> str:
        """Extract text content from a Message object."""
        text_parts = []
        for part in message.parts:
            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                text_parts.append(part.root.text)
            elif hasattr(part, 'text'):
                text_parts.append(part.text)
        return "\n".join(text_parts)
    
    async def cleanup(self) -> None:
        """Cleanup the agent and all its resources."""
        logger.debug("Cleaning up BaseAgent resources...")
        
        # Cleanup planner
        if hasattr(self.planner, 'cleanup'):
            try:
                await self.planner.cleanup()
                logger.debug("Cleaned up planner")
            except Exception as e:
                logger.warning(f"Error cleaning up planner: {e}")
        
        # Cleanup session manager (which cleans up all sessions)
        try:
            await self.session_manager.cleanup()
            logger.debug("Cleaned up session manager")
        except Exception as e:
            logger.warning(f"Error cleaning up session manager: {e}")
        
        # Cleanup tool registry and tools
        for toolset in self.tool_registry.get_toolsets():
            try:
                if hasattr(toolset, 'cleanup'):
                    await toolset.cleanup()
                elif hasattr(toolset, 'close'):
                    await toolset.close()
            except Exception as e:
                logger.warning(f"Error cleaning up toolset {toolset.name}: {e}")
        
        logger.debug("BaseAgent cleanup completed") 