import typing as _t
import abc
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

import pydantic as _pyd
from a2a.types import Message as A2AMessage, Task as A2ATask, TaskStatusUpdateEvent, TaskArtifactUpdateEvent, TaskState, TextPart, Role

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
    "ToolUsePart",
    
    # Services and configuration
    "Metrics",
    "RunConfig",
    
    # Event types
    "EventType",
    "RunEvent",
    "Event",
]




# =============================================================================
# External
#  
# Agent.run(AgentRunInput) -> AgentRunOutput(Task|Message|Event)
# =============================================================================

class Task(A2ATask):
    """Extended A2A Task with additional functionality."""


class Message(A2AMessage):
    """Extended A2A Message with additional functionality."""


class ToolUsePart(_pyd.BaseModel):
    """Represents a tool use (call and/or result) segment within message parts."""
    
    kind: _t.Literal['tool-use'] = 'tool-use'
    """Part type - tool-use for ToolUseParts"""
    
    type: _t.Literal['call', 'result'] = 'call'
    """Whether this is a tool call or tool result"""
    
    id: str
    """Unique identifier for the tool use"""
    
    name: str
    """Name of the tool being called"""
    
    input: dict[str, _t.Any] | None = None
    """Tool input/arguments (for calls)"""
    
    output: _t.Any | None = None
    """Tool output/result (for results) - can be string, dict, or any serializable type"""
    
    metadata: dict[str, _t.Any] | None = None
    """Optional metadata associated with the part."""


class EventType(str, Enum):
    """Types of events for EventStep."""
    # A2A Events
    TASK_STATUS_UPDATE = "task_status_update"
    TASK_ARTIFACT_UPDATE = "task_artifact_update"

    TASK_START = "task_start"
    TASK_END = "task_end" 
    THINK = "think"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    ERROR = "error"


@dataclass
class RunEvent:
    """Internal event type for non-A2A events."""
    event_type: str
    data: dict[str, _t.Any]
    metadata: dict[str, _t.Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

# Union type for all events
Event = TaskStatusUpdateEvent | TaskArtifactUpdateEvent | RunEvent



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


@dataclass
class RunConfig:
    """Budget constraints and agent configuration."""
    turn_budget: int | None = None         # Maximum planning turns
    token_budget: int | None = None        # Maximum tokens
    cost_budget: float | None = None       # Maximum cost
    time_budget: float | None = None       # Maximum execution time in seconds
    deadline: datetime | None = None       # Absolute deadline
    
    def is_budget_exceeded(self, metrics: "Metrics", elapsed_time: float) -> tuple[bool, str]:
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



# =============================================================================
# Internal
#  
# Planner.next(State) -> Steps
# =============================================================================

@dataclass
class State:
    """Comprehensive state containing all context for planning and execution."""
    
    # Static states that are set once and never change duration the task
    session_id: str
    agent_id: str = ""
    config: RunConfig = field(default_factory=RunConfig)
    tools: list["Tool"] = field(default_factory=list)  # Tool objects from base.py
    created_at: datetime = field(default_factory=datetime.now)

    # Current task (A2A compatible)
    task: Task | None = None
    parent_task_id: str | None = None  
    iteration: int = 0
    
    # Memory references (readonly)
    history: list[Message] = field(default_factory=list)
    artifact_ref: "ArtifactService | None" = None  # Readonly reference
    
    # Performance metrics
    metrics: Metrics = field(default_factory=Metrics)

    # Additional context
    metadata: dict[str, _t.Any] = field(default_factory=dict)


# =============================================================================
# Enhanced Step Types (A2A Compatible)
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
    """Enhanced base class for all planner steps - supports both Message and Event content."""
    step_id: str
    step_type: StepType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, _t.Any] = field(default_factory=dict)
    
    # Content fields - can contain multiple types
    message: Message | None = None              # A2A Message content
    event: Event | None = None                  # A2A Event or Internal Event content
    thinking: str | None = None                 # Internal reasoning
    reason: str | None = None                   # Completion reason
    success: bool = True
    
    def has_message_content(self) -> bool:
        """Check if step contains message content."""
        return self.message is not None
    
    def has_event_content(self) -> bool:
        """Check if step contains event content."""
        return self.event is not None
    
    def has_tool_content(self) -> bool:
        """Check if step contains tool call/result content (implemented by subclasses)."""
        return False  # Base implementation, overridden by ToolCallStep
    
    def get_content_types(self) -> list[str]:
        """Get list of content types present in this step."""
        types = []
        if self.has_message_content():
            types.append("message")
        if self.has_event_content():
            types.append("event")
        if self.has_tool_content():
            types.append("tool")
        if self.thinking:
            types.append("thinking")
        return types


@dataclass
class MessageStep(Step):
    """A2A compatible message step with enhanced support for mixed content."""
    is_streaming: bool = False
    step_type: StepType = StepType.MESSAGE
    
    def __post_init__(self):
        if self.message is None:
            raise ValueError("message is required for MessageStep")
    
    async def stream_content(self) -> _t.AsyncGenerator[str, None]:
        """Stream message content if is_streaming=True."""
        if not self.is_streaming or not self.message:
            return
        
        # Stream text parts from the message
        for part in self.message.parts:
            if hasattr(part, 'text'):
                # Stream text content in chunks
                text = part.text
                chunk_size = 50  # Adjust as needed
                for i in range(0, len(text), chunk_size):
                    yield text[i:i + chunk_size]


@dataclass  
class EventStep(Step):
    """A2A compatible event step with support for both A2A and internal events."""
    event_type: EventType = field(default=None)  # Will be set via constructor
    step_type: StepType = StepType.EVENT
    
    def __post_init__(self):
        if self.event is None:
            raise ValueError("event is required for EventStep")
        if self.event_type is None:
            raise ValueError("event_type is required for EventStep")
    
    def is_a2a_event(self) -> bool:
        """Check if this is an A2A standard event."""
        return self.event_type in [EventType.TASK_STATUS_UPDATE, EventType.TASK_ARTIFACT_UPDATE]
    
    def is_internal_event(self) -> bool:
        """Check if this is an internal (non-A2A) event."""
        return not self.is_a2a_event()


@dataclass
class ToolCallStep(Step):
    """Step for executing a tool call with enhanced result handling."""
    step_type: StepType = StepType.TOOL_CALL
    tool_call: ToolUsePart | None = None
    tool_call_result: ToolUsePart | None = None
    
    def __post_init__(self):
        if self.tool_call is None:
            raise ValueError("tool_call (ToolUsePart type='call') is required for ToolCallStep")
        if self.tool_call.type != 'call':
            raise ValueError("ToolCallStep.tool_call must be a ToolUsePart with type='call'")
    
    def has_tool_content(self) -> bool:
        """Check if step contains tool call/result content."""
        return self.tool_call is not None or self.tool_call_result is not None
    
    def add_result(self, result: ToolUsePart) -> None:
        """Add tool call result (ToolUsePart type='result') to this step."""
        if result.type != 'result':
            raise ValueError("Result added to ToolCallStep must be a ToolUsePart with type='result'")
        self.tool_call_result = result
        # Extract success from metadata; default to False if not present or malformed
        success_val = result.metadata.get('success', False) if result.metadata else False
        self.success = bool(success_val)
    
    def to_message_parts(self) -> tuple[ToolUsePart, ToolUsePart | None]:
        """Convert tool call and result to A2A message parts.
        
        Returns the tool_call part and tool_call_result part directly.
        """
        # self.tool_call is already a ToolUsePart (type='call')
        # self.tool_call_result is already a ToolUsePart (type='result') or None
        if self.tool_call is None:
            # This case should ideally be prevented by __post_init__, but as a safeguard:
            raise ValueError("ToolCallStep.tool_call cannot be None when generating message parts")
            
        return self.tool_call, self.tool_call_result


@dataclass 
class ThinkStep(Step):
    """Step for internal agent thinking/reasoning."""
    step_type: StepType = StepType.THINK
    
    def __post_init__(self):
        if self.thinking is None:
            self.thinking = ""


@dataclass
class FinishStep(Step):
    """Step indicating task completion with enhanced content support."""
    step_type: StepType = StepType.FINISH
    final_message: Message | None = None
    
    def __post_init__(self):
        if self.reason is None:
            self.reason = "Task completed"
        # Only create a message if none is provided and we have a reason
        if self.message is None and self.final_message is None and self.reason:
            # Create a simple message for the finish step
            self.final_message = Message(
                messageId=f"finish-{self.step_id}",
                role=Role.agent,
                parts=[TextPart(text=self.reason)],
                contextId=None,
                kind="message"
            )
    
    def create_completion_event(self, task_id: str, context_id: str) -> TaskStatusUpdateEvent:
        """Create a task completion event."""
        from a2a.types import TaskStatus, TaskState
        completion_status = TaskStatus(
            state=TaskState.completed,
            message=self.message,
            timestamp=datetime.now().isoformat()
        )
        return TaskStatusUpdateEvent(
            contextId=context_id,
            taskId=task_id,
            status=completion_status,
            final=True,
            metadata=self.metadata
        ) 