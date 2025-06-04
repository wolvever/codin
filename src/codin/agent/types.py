import typing as _t
import abc
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

import pydantic as _pyd
from a2a.types import Message, Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent

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
    
    # Task management
    "TaskStatus",
    "TaskInfo",
    
    # Services and configuration
    "Metrics",
    "AgentConfig",
    "ChatHistory",
    "MemoryService",
    "ArtifactService",
    
    # Event types
    "EventType",
    "InternalEvent",
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
    task_id: str | None = None  # Optional task ID for continuing existing task
        
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
# Task Management
# =============================================================================

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TaskInfo:
    """Task information for State."""
    id: str
    parent_id: str | None = None
    query: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, _t.Any] = field(default_factory=dict)


# =============================================================================
# Metrics and Configuration
# =============================================================================

@dataclass
class Metrics:
    """Performance and usage metrics."""
    iterations: int = 0
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_used: float = 0.0
    elapsed_seconds: float = 0.0
    tool_calls_made: int = 0
    errors_encountered: int = 0
    
    def add_tokens(self, input_tokens: int, output_tokens: int, cost: float = 0.0) -> None:
        """Add token usage to metrics."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.tokens_used += input_tokens + output_tokens
        self.cost_used += cost
    
    def increment_tool_calls(self) -> None:
        """Increment tool call counter."""
        self.tool_calls_made += 1
    
    def increment_errors(self) -> None:
        """Increment error counter."""
        self.errors_encountered += 1


@dataclass
class AgentConfig:
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


# =============================================================================
# Service Interfaces
# =============================================================================

class ChatHistory(abc.ABC):
    """Readonly chat history interface."""
    
    @abc.abstractmethod
    def get_recent_messages(self, count: int = 10) -> list[Message]:
        """Get recent messages for context."""
        ...
    
    @abc.abstractmethod
    def search_messages(self, query: str) -> list[Message]:
        """Search through message history."""
        ...
    
    @abc.abstractmethod
    def get_all_messages(self) -> list[Message]:
        """Get all messages in history."""
        ...


class MemoryService(abc.ABC):
    """Service for managing conversation memory and chat history."""
    
    @abc.abstractmethod
    async def get_chat_history(self, session_id: str) -> ChatHistory:
        """Get chat history for session."""
        ...
    
    @abc.abstractmethod 
    async def add_message(self, session_id: str, message: Message) -> None:
        """Add message to chat history."""
        ...


class ArtifactService(abc.ABC):
    """Service for managing code artifacts and files."""
    
    @abc.abstractmethod
    async def get_artifact(self, artifact_id: str) -> _t.Any:
        """Get artifact by ID."""
        ...
    
    @abc.abstractmethod
    async def save_artifact(self, content: _t.Any, metadata: dict) -> str:
        """Save artifact and return ID."""
        ...


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
    
    # Conversation history (readonly reference)
    history: ChatHistory | None = None  # From MemoryService
    
    # Memory and artifact references (readonly)
    memory_ref: MemoryService | None = None  # Readonly reference 
    artifact_ref: ArtifactService | None = None  # Readonly reference
    
    # Tools and execution context
    tools: list[_t.Any] = field(default_factory=list)  # Tool objects
    tool_call_results: list[ToolCallResult] = field(default_factory=list)
    
    # Performance metrics
    metrics: Metrics = field(default_factory=Metrics)
    
    # Budget constraints
    config: AgentConfig = field(default_factory=AgentConfig)
    
    # Task management
    current_task: TaskInfo | None = None
    
    # Additional context
    context: dict[str, _t.Any] = field(default_factory=dict)
    metadata: dict[str, _t.Any] = field(default_factory=dict)


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
    """Base class for all planner steps."""
    step_id: str
    step_type: StepType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, _t.Any] = field(default_factory=dict)


@dataclass
class MessageStep(Step):
    """A2A compatible message step."""
    message: Message = field(default=None)  # Will be set via constructor
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
    event: TaskStatusUpdateEvent | TaskArtifactUpdateEvent | InternalEvent = field(default=None)  # Will be set via constructor
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
    tool_call: ToolCall = field(default=None)  # Will be set via constructor
    step_type: StepType = StepType.TOOL_CALL
    
    def __post_init__(self):
        if self.tool_call is None:
            raise ValueError("tool_call is required for ToolCallStep")


@dataclass 
class ThinkStep(Step):
    """Step for internal agent thinking/reasoning."""
    thinking: str = field(default="")  # Default empty thinking
    step_type: StepType = StepType.THINK


@dataclass
class FinishStep(Step):
    """Step indicating task completion."""
    reason: str = "Task completed"
    success: bool = True
    final_message: Message | None = None
    step_type: StepType = StepType.FINISH 