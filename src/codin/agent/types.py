import typing as _t
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

import pydantic as _pyd
from a2a.types import Message, Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent

__all__ = [
    # Agent types from base.py
    "AgentRunInput",
    "AgentRunOutput", 
    "ToolCall",
    "ToolCallResult",
    
    # Planner types from planner.py
    "Step",
    "StepType",
    "ThinkStep",
    "MessageStep", 
    "ToolCallStep",
    "FinishStep",
    "State",
]


# =============================================================================
# Agent Types (from base.py)
# =============================================================================

class AgentRunInput(_pyd.BaseModel):
    """Input for agent execution."""
    id: str | int | None = None
    message: Message
    metadata: dict[str, _t.Any] | None = None
    options: dict[str, _t.Any] | None = None
    session_id: str | None = None
        
    class Config:
        arbitrary_types_allowed = True


class AgentRunOutput(_pyd.BaseModel):
    """Output from agent execution."""
    id: str | int | None = None
    result: Task | Message | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
    metadata: dict[str, _t.Any] | None = None
    
    class Config:
        arbitrary_types_allowed = True


class ToolCall(_pyd.BaseModel):
    """Represents a tool call from the LLM."""
    call_id: str
    name: str
    arguments: dict[str, _t.Any]


class ToolCallResult(_pyd.BaseModel):
    """Result of a tool execution."""
    call_id: str
    success: bool
    output: str
    error: str | None = None


# =============================================================================
# Planner Types (from planner.py)
# =============================================================================

class StepType(Enum):
    """Types of steps a planner can emit."""
    THINK = "think"           # Internal reasoning step
    MESSAGE = "message"       # Send message to user
    TOOL_CALL = "tool_call"   # Execute a tool
    FINISH = "finish"         # Task completion


@dataclass
class Step:
    """Base class for all planner steps."""
    step_id: str
    step_type: StepType
    timestamp: datetime
    metadata: dict[str, _t.Any] | None = None


@dataclass 
class ThinkStep(Step):
    """Step for internal agent thinking/reasoning."""
    thinking: str
    step_type: StepType = StepType.THINK


@dataclass
class MessageStep(Step):
    """Step for sending a message response."""
    message: Message
    is_streaming: bool = False
    step_type: StepType = StepType.MESSAGE


@dataclass
class ToolCallStep(Step):
    """Step for executing a tool call."""
    tool_call: ToolCall
    step_type: StepType = StepType.TOOL_CALL


@dataclass
class FinishStep(Step):
    """Step indicating task completion."""
    final_message: Message | None = None
    reason: str = "Task completed"
    step_type: StepType = StepType.FINISH


@dataclass
class State:
    """Current conversation and execution state."""
    session_id: str
    turn_count: int
    history: list[Message]
    task_list: dict[str, list[str]]  # {"completed": [...], "pending": [...]}
    context: dict[str, _t.Any]       # Additional context for planner
    metrics: dict[str, _t.Any]       # Performance metrics
    last_tool_results: list[_t.Any] | None = None
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        self.history.append(message)
    
    def update_metrics(self, key: str, value: _t.Any) -> None:
        """Update performance metrics."""
        self.metrics[key] = value
    
    def update_context(self, updates: dict[str, _t.Any]) -> None:
        """Update context with new information."""
        self.context.update(updates) 