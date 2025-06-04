import typing as _t
import abc
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

import pydantic as _pyd
from a2a.types import Message as A2AMessage, Task as A2ATask, TaskStatusUpdateEvent, TaskArtifactUpdateEvent, TaskState, TextPart

# Import ArtifactService and Memory from new locations - using TYPE_CHECKING to avoid circular imports
if _t.TYPE_CHECKING:
    from ..artifact.base import ArtifactService
    from ..memory.base import Memory, MemoryWriter
    from ..tool.base import Tool

__all__ = [
    # Agent types from base.py
    "AgentRunInput",
    "AgentRunOutput", 
    "ToolCall",
    "ToolCallResult",
    
    # Core architecture types
    "State",
    "Step",
    "StepType",
    "MessageStep",
    "EventStep", 
    "ToolCallStep",
    "ThinkStep",
    "FinishStep",
    
    # A2A Compatible types
    "Message",
    "Task",
    "TaskStatus",
    "TextPart",
    
    # Services and configuration
    "Metrics",
    "RunConfig",
    
    # Event types
    "EventType",
    "InternalEvent",
    "Event",
]


# =============================================================================
# A2A Compatible Types
# =============================================================================

class Message(A2AMessage):
    """Extended A2A Message with additional functionality."""
    
    def add_text_part(self, text: str, metadata: dict[str, _t.Any] | None = None) -> None:
        """Add a text part to the message."""
        from a2a.types import TextPart
        text_part = TextPart(text=text, metadata=metadata)
        self.parts.append(text_part)
    
    def add_file_part(self, file_data: dict[str, _t.Any], metadata: dict[str, _t.Any] | None = None) -> None:
        """Add a file part to the message."""
        from a2a.types import FilePart
        file_part = FilePart(file=file_data, metadata=metadata)
        self.parts.append(file_part)
    
    def add_data_part(self, data: dict[str, _t.Any], metadata: dict[str, _t.Any] | None = None) -> None:
        """Add a data part to the message."""
        from a2a.types import DataPart
        data_part = DataPart(data=data, metadata=metadata)
        self.parts.append(data_part)


class Task(A2ATask):
    """Extended A2A Task with additional functionality."""
    
    def add_message(self, message: Message) -> None:
        """Add a message to the task history."""
        if self.history is None:
            self.history = []
        self.history.append(message)
    
    def update_status(self, state: TaskState, message: Message | None = None, timestamp: str | None = None) -> None:
        """Update the task status."""
        from a2a.types import TaskStatus
        self.status = TaskStatus(
            state=state,
            message=message,
            timestamp=timestamp or datetime.now().isoformat()
        )


# Task status alias for compatibility
TaskStatus = TaskState


# =============================================================================
# Event Types
# =============================================================================

class EventType(Enum):
    """Types of events for EventStep."""
    # A2A Events
    TASK_STATUS_UPDATE = "task_status_update"
    TASK_ARTIFACT_UPDATE = "task_artifact_update"
    
    # Internal Events
    TASK_START = "task_start"
    TASK_END = "task_end" 
    THINK = "think"
    TOOL_CALL_BEGIN = "tool_call_begin"
    TOOL_CALL_END = "tool_call_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    ERROR = "error"


@dataclass
class InternalEvent:
    """Internal event type for non-A2A events."""
    event_type: str
    data: dict[str, _t.Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, _t.Any] = field(default_factory=dict)


# Union type for all events
Event = TaskStatusUpdateEvent | TaskArtifactUpdateEvent | InternalEvent


# =============================================================================
# Agent Types (from base.py) - Updated for A2A compatibility
# =============================================================================

class AgentRunInput(_pyd.BaseModel):
    """Input for agent execution."""
    id: str | int | None = None
    message: Message
    metadata: dict[str, _t.Any] | None = None
    options: dict[str, _t.Any] | None = None
    session_id: str | None = None
    task_id: str | None = None  # Optional task ID for continuing existing task
        
    class Config:
        arbitrary_types_allowed = True


class AgentRunOutput(_pyd.BaseModel):
    """Output from agent execution."""
    id: str | int | None = None
    result: Task | Message | Event
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
# Metrics and Configuration
# =============================================================================

@dataclass
class Metrics:
    """Performance and usage metrics."""
    iterations: int = 0
    input_token_used: int = 0
    output_token_used: int = 0
    cost_used: float = 0.0
    time_used: float = 0.0
    tool_calls: int = 0
    llm_calls: int = 0
    errors: int = 0
    
    # Computed properties for backward compatibility
    @property
    def tokens_used(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_token_used + self.output_token_used
    
    @property
    def elapsed_seconds(self) -> float:
        """Alias for time_used."""
        return self.time_used
    
    @property
    def tool_calls_made(self) -> int:
        """Alias for tool_calls."""
        return self.tool_calls
    
    @property
    def errors_encountered(self) -> int:
        """Alias for errors."""
        return self.errors
    
    def add_tokens(self, input_tokens: int, output_tokens: int, cost: float = 0.0) -> None:
        """Add token usage to metrics."""
        self.input_token_used += input_tokens
        self.output_token_used += output_tokens
        self.cost_used += cost
    
    def increment_tool_calls(self) -> None:
        """Increment tool call counter."""
        self.tool_calls += 1
    
    def increment_llm_calls(self) -> None:
        """Increment LLM call counter."""
        self.llm_calls += 1
    
    def increment_errors(self) -> None:
        """Increment error counter."""
        self.errors += 1


@dataclass
class RunConfig:
    """Budget constraints and agent configuration."""
    turn_budget: int | None = None         # Maximum planning turns
    token_budget: int | None = None        # Maximum tokens
    cost_budget: float | None = None       # Maximum cost
    time_budget: float | None = None       # Maximum execution time in seconds
    deadline: datetime | None = None       # Absolute deadline
    
    def is_budget_exceeded(self, metrics: Metrics, elapsed_time: float) -> tuple[bool, str]:
        """Check if any budget constraints are exceeded."""
        if self.turn_budget and metrics.iterations >= self.turn_budget:
            return True, f"Turn budget exceeded: {metrics.iterations} >= {self.turn_budget}"
        
        if self.token_budget and metrics.tokens_used >= self.token_budget:
            return True, f"Token budget exceeded: {metrics.tokens_used} >= {self.token_budget}"
        
        if self.cost_budget and metrics.cost_used >= self.cost_budget:
            return True, f"Cost budget exceeded: {metrics.cost_used} >= {self.cost_budget}"
        
        if self.time_budget and elapsed_time >= self.time_budget:
            return True, f"Time budget exceeded: {elapsed_time:.1f}s >= {self.time_budget}s"
        
        if self.deadline and datetime.now() >= self.deadline:
            return True, f"Deadline exceeded: {datetime.now()} >= {self.deadline}"
        
        return False, ""


# Backward compatibility alias
AgentConfig = RunConfig


# =============================================================================
# Comprehensive State
# =============================================================================

@dataclass
class State:
    """Comprehensive state containing all context for planning and execution."""
    
    # Identity and hierarchy
    session_id: str
    task_id: str | None = None
    parent_task_id: str | None = None  
    agent_id: str = ""
    
    # Temporal context
    created_at: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    
    # Memory references (readonly)
    memory: "Memory | None" = None  # Read-only memory access
    memory_writer: "MemoryWriter | None" = None  # Write memory access
    artifact_ref: "ArtifactService | None" = None  # Readonly reference
    
    # Tools and execution context - using Tool from base.py
    tools: list["Tool"] = field(default_factory=list)  # Tool objects from base.py
    tool_call_results: list[ToolCallResult] = field(default_factory=list)
    
    # Performance metrics
    metrics: Metrics = field(default_factory=Metrics)
    
    # Budget constraints
    config: RunConfig = field(default_factory=RunConfig)
    
    # Current task (A2A compatible)
    current_task: Task | None = None
    
    # Additional context
    context: dict[str, _t.Any] = field(default_factory=dict)
    metadata: dict[str, _t.Any] = field(default_factory=dict)


# =============================================================================
# Step Types (A2A Compatible)
# =============================================================================

class StepType(Enum):
    """Types of steps a planner can emit."""
    MESSAGE = "message"      # A2A Message (streaming or non-streaming)
    EVENT = "event"          # A2A Event + internal events
    TOOL_CALL = "tool_call"  # Tool execution 
    THINK = "think"          # Internal reasoning
    FINISH = "finish"        # Task completion


@dataclass
class Step:
    """Base class for all planner steps - more general implementation."""
    step_id: str
    step_type: StepType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, _t.Any] = field(default_factory=dict)
    
    # General content fields
    message: Message | None = None
    event: Event | None = None
    tool_call: ToolCall | None = None
    thinking: str | None = None
    reason: str | None = None
    success: bool = True


@dataclass
class MessageStep(Step):
    """A2A compatible message step."""
    is_streaming: bool = False
    step_type: StepType = StepType.MESSAGE
    
    def __post_init__(self):
        if self.message is None:
            raise ValueError("message is required for MessageStep")
    
    async def stream_content(self) -> _t.AsyncGenerator[str, None]:
        """Stream message content if is_streaming=True."""
        if not self.is_streaming:
            return
        
        # Implementation depends on how streaming is handled
        # This is a placeholder for the streaming interface
        yield "streaming content would go here"


@dataclass  
class EventStep(Step):
    """A2A compatible event step with internal event support."""
    event_type: EventType = field(default=None)  # Will be set via constructor
    step_type: StepType = StepType.EVENT
    
    def __post_init__(self):
        if self.event is None:
            raise ValueError("event is required for EventStep")
        if self.event_type is None:
            raise ValueError("event_type is required for EventStep")


@dataclass
class ToolCallStep(Step):
    """Step for executing a tool call."""
    step_type: StepType = StepType.TOOL_CALL
    
    def __post_init__(self):
        if self.tool_call is None:
            raise ValueError("tool_call is required for ToolCallStep")


@dataclass 
class ThinkStep(Step):
    """Step for internal agent thinking/reasoning."""
    step_type: StepType = StepType.THINK
    
    def __post_init__(self):
        if self.thinking is None:
            self.thinking = ""


@dataclass
class FinishStep(Step):
    """Step indicating task completion."""
    step_type: StepType = StepType.FINISH
    
    def __post_init__(self):
        if self.reason is None:
            self.reason = "Task completed"
        if self.message is None and self.reason:
            # Create a simple message for the finish step
            self.message = Message(
                messageId=f"finish-{self.step_id}",
                role="agent",
                parts=[],
                taskId=None
            )
            self.message.add_text_part(self.reason) 