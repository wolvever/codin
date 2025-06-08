"""Type definitions for agent system."""

import typing as _t
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pydantic as _pyd
from pydantic import BaseModel, ConfigDict, Field, model_validator

Any = _t.Any


if _t.TYPE_CHECKING:
    from ..artifact.base import ArtifactService
    from ..tool.base import Tool

__all__ = [
    # Core types (A2A compatible)
    'Task',
    'Message',
    'Role',
    'TextPart',
    'DataPart',
    'FilePart',
    'TaskState',
    'TaskStatus',
    'TaskStatusUpdateEvent',
    'TaskArtifactUpdateEvent',
    'ToolUsePart',
    'ToolCall',
    'ToolCallResult',
    # Enhanced types for internal use
    "RunEvent",
    "Event",
    "ControlSignal",
    "RunnerControl",
    "RunnerInput",
    # Agent types
    "AgentRunInput",
    "AgentRunOutput",
    "RunConfig",
    "Metrics",
    "State",
    # Planning types
    "StepType",
    "Step",
    "MessageStep",
    "ToolCallStep",
    "EventStep",
    "ThinkStep",
    "FinishStep",
    "ErrorStep",
    # Base interfaces
    "Planner",
    "EventType",
]

class Role(str, Enum):
    """Simple role enumeration used across the codebase."""

    user = "user"
    agent = "agent"
    assistant = "assistant"


@dataclass
class TextPart:
    text: str
    kind: str = "text"
    metadata: dict[str, _t.Any] | None = None


@dataclass
class DataPart:
    data: dict[str, _t.Any]
    kind: str = "data"
    metadata: dict[str, _t.Any] | None = None


@dataclass
class FilePart:
    uri: str | None = None
    path: str | None = None
    kind: str = "file"
    metadata: dict[str, _t.Any] | None = None


@dataclass
class Message:
    messageId: str
    role: Role
    parts: list[_t.Any]
    contextId: str | None = None
    kind: str = "message"
    metadata: dict[str, _t.Any] = field(default_factory=dict)
    taskId: str | None = None
    referenceTaskIds: list[str] | None = None

    def add_text_part(self, text: str, metadata: dict[str, _t.Any] | None = None) -> None:
        """Append a TextPart to the message."""
        self.parts.append(TextPart(text=text, metadata=metadata))

    def add_data_part(self, data: dict[str, _t.Any], metadata: dict[str, _t.Any] | None = None) -> None:
        """Append a DataPart to the message."""
        self.parts.append(DataPart(data=data, metadata=metadata))

    def add_tool_call_part(self, call: ToolCall) -> None:
        """Append a tool call as a ToolUsePart."""
        self.parts.append(
            ToolUsePart(
                type="call",
                id=call.call_id,
                name=call.name,
                input=call.arguments,
            )
        )

    def add_tool_result_part(self, result: ToolCallResult, name: str) -> None:
        """Append a tool result as a ToolUsePart."""
        self.parts.append(
            ToolUsePart(
                type="result",
                id=result.call_id,
                name=name,
                output=result.output,
                metadata={"error": result.error} if result.error else None,
            )
        )


class TaskState(str, Enum):
    QUEUED = "queued"
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"

    queued = QUEUED
    submitted = SUBMITTED
    working = WORKING
    completed = COMPLETED
    failed = FAILED


@dataclass
class TaskStatus:
    state: TaskState
    message: Message | None = None
    timestamp: str | None = None


@dataclass
class TaskStatusUpdateEvent:
    contextId: str
    taskId: str
    status: TaskStatus
    final: bool = False
    metadata: dict[str, _t.Any] = field(default_factory=dict)


@dataclass
class TaskArtifactUpdateEvent:
    """Event representing artifact updates for a task."""

    contextId: str
    taskId: str
    artifact: _t.Any
    append: bool = False
    lastChunk: bool = False
    metadata: dict[str, _t.Any] = field(default_factory=dict)


@dataclass
class Task:
    id: str
    contextId: str
    status: TaskStatus
    query: str | None = None
    parts: list[_t.Any] = field(default_factory=list)
    message: Message | None = None
    metadata: dict[str, _t.Any] = field(default_factory=dict)

    def add_text_part(self, text: str, metadata: dict[str, _t.Any] | None = None) -> None:
        self.parts.append(TextPart(text=text, metadata=metadata))

    def add_data_part(self, data: dict[str, _t.Any], metadata: dict[str, _t.Any] | None = None) -> None:
        """Convenience helper to append a DataPart."""
        self.parts.append(DataPart(data=data, metadata=metadata))

    def add_tool_call_part(self, call: ToolCall) -> None:
        self.parts.append(ToolUsePart(type="call", id=call.call_id, name=call.name, input=call.arguments))

    def add_tool_result_part(self, result: ToolCallResult, name: str) -> None:
        self.parts.append(
            ToolUsePart(
                type="result",
                id=result.call_id,
                name=name,
                output=result.output,
                metadata={"error": result.error} if result.error else None,
            )
        )


class TaskState(str, Enum):
    submitted = "submitted"
    working = "working"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TaskStatus(BaseModel):
    state: TaskState


class TaskStatusUpdateEvent(BaseModel):
    contextId: str
    taskId: str
    status: TaskStatus
    final: bool = False


class TaskArtifactUpdateEvent(BaseModel):
    contextId: str
    taskId: str
    artifact: dict[str, _t.Any] | None = None
    final: bool = False


class Task(BaseModel):
    id: str
    contextId: str | None = None
    status: TaskStatus | None = None
    message: Message | None = None
    metadata: dict[str, _t.Any] | None = None


class Artifact(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    parts: list[Part] = Field(default_factory=list)
    metadata: dict[str, _t.Any] | None = None


class ToolCall(_pyd.BaseModel):
    """Represents a tool call request."""

    call_id: str
    name: str
    arguments: dict[str, _t.Any]


class ToolCallResult(_pyd.BaseModel):
    """Represents the result of a tool call."""

    call_id: str
    success: bool
    output: _t.Any = None
    error: str | None = None


class ToolUsePart(_pyd.BaseModel):
    """Represents a tool use (call and/or result) segment within message parts."""

    kind: _t.Literal["tool-use"] = "tool-use"
    """Part type - tool-use for ToolUseParts"""

    type: _t.Literal["call", "result"] = "call"
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
    """Additional metadata for this tool use"""


# =============================================================================
# Internal events and control types
# =============================================================================


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


class RunEvent(BaseModel):
    """Internal event type for non-A2A events."""

    event_type: str
    data: dict[str, _t.Any]
    metadata: dict[str, _t.Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# Union type for all events
Event = TaskStatusUpdateEvent | TaskArtifactUpdateEvent | RunEvent


# =============================================================================
# Control and Runner Types for Bidirectional Mailbox
# =============================================================================


class ControlSignal(str, Enum):
    """Control signals that can be sent through mailbox."""

    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    RESET = "reset"
    STOP = "stop"


class RunnerControl(BaseModel):
    """Control message for runner/agent management."""

    signal: ControlSignal
    metadata: dict[str, _t.Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class RunnerInput(BaseModel):
    """Enhanced input for agents through bidirectional mailbox."""

    message: Message | None = None
    control: RunnerControl | None = None
    metadata: dict[str, _t.Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    @classmethod
    def from_message(cls, message: Message) -> RunnerInput:
        """Create RunnerInput from a message."""
        return cls(message=message)

    @classmethod
    def from_control(cls, signal: ControlSignal, metadata: dict[str, _t.Any] | None = None) -> RunnerInput:
        """Create RunnerInput from a control signal."""
        control = RunnerControl(signal=signal, metadata=metadata or {})
        return cls(control=control)


# =============================================================================
# Agent Types (from base.py) - Updated for A2A compatibility
# =============================================================================


class AgentRunInput(_pyd.BaseModel):
    """Input for agent execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str | int | None = None
    message: Message
    metadata: dict[str, _t.Any] | None = None
    options: dict[str, _t.Any] | None = None
    session_id: str | None = None
    task_id: str | None = None  # Optional task ID for continuing existing task


class AgentRunOutput(_pyd.BaseModel):
    """Output from agent execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str | int | None = None
    result: Task | Message | Event
    metadata: dict[str, _t.Any] | None = None


class RunConfig(BaseModel):
    """Budget constraints and agent configuration."""

    turn_budget: int | None = None  # Maximum planning turns
    token_budget: int | None = None  # Maximum tokens
    cost_budget: float | None = None  # Maximum cost
    time_budget: float | None = None  # Maximum execution time in seconds
    deadline: datetime | None = None  # Absolute deadline

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


class Metrics(BaseModel):
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
# Internal - Planner.next(State) -> Steps
# =============================================================================


class State(BaseModel):
    """Comprehensive state containing all context for planning and execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    ## Static members that are set once and never change duration the task
    session_id: str
    agent_id: str = ""
    config: RunConfig = Field(default_factory=RunConfig)
    model_config_dict: dict[str, _t.Any] = Field(default_factory=dict, alias="model_config")
    tools: list[Tool] = Field(default_factory=list)  # Tool objects from base.py
    created_at: datetime = Field(default_factory=datetime.now)

    ## Dynamic updated members

    # Current task (A2A compatible)
    task: Task | None = None
    parent_task_id: str | None = None
    iteration: int = 0
    turn_count: int = 0  # Track number of planning turns

    # Memory references (readonly)
    pending: list[Message] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)
    artifact_ref: ArtifactService | None = None  # Readonly reference
    metadata: dict[str, _t.Any] = Field(default_factory=dict)

    # Performance metrics
    metrics: Metrics = Field(default_factory=Metrics)

    # Last tool results for context
    last_tool_results: list[_t.Any] = Field(default_factory=list)

    # Task list for structured planning (matches code_agent.py format)
    task_list: dict[str, list[str]] = Field(default_factory=lambda: {"completed": [], "pending": []})


class StepType(Enum):
    """Types of steps a planner can emit."""

    MESSAGE = "message"  # A2A Message (streaming or non-streaming)
    EVENT = "event"  # A2A Event + internal events
    TOOL_CALL = "tool_call"  # Tool execution
    THINK = "think"  # Internal reasoning
    FINISH = "finish"  # Task completion
    ERROR = "error"  # Error step


class Step(BaseModel):
    """Enhanced base class for all planner steps - supports both Message and Event content."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_id: str
    step_type: StepType
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, _t.Any] = Field(default_factory=dict)

    # Content fields - can contain multiple types
    message: Message | None = None  # A2A Message content
    event: Event | None = None  # A2A Event or Internal Event content
    tool_call: ToolUsePart | ToolCall | None = None
    tool_call_result: ToolUsePart | ToolCallResult | None = None
    thinking: str | None = None  # Internal reasoning

    # ------------------------------------------------------------------
    # Content helpers
    # ------------------------------------------------------------------

    def has_message_content(self) -> bool:
        """Return True if this step contains message content."""
        return self.message is not None or getattr(self, "final_message", None) is not None

    def has_event_content(self) -> bool:
        """Return True if this step contains an event."""
        return self.event is not None

    def has_tool_content(self) -> bool:
        """Return True if this step relates to tool usage."""
        if isinstance(self, ToolCallStep):
            return self.tool_call is not None or self.tool_call_result is not None
        if getattr(self, "tool_call", None) is not None or getattr(self, "tool_call_result", None) is not None:
            return True
        if self.message:
            return any(getattr(p, "kind", None) == "tool-use" for p in self.message.parts)
        return False

    def get_content_types(self) -> list[str]:
        """Return a list of content types present on this step."""
        types: list[str] = []
        if self.has_message_content():
            types.append("message")
        if self.has_event_content():
            types.append("event")
        if self.thinking:
            types.append("thinking")
        if self.has_tool_content():
            if "tool" not in types:
                types.append("tool")
        return types

    def is_a2a_event(self) -> bool:
        """Return True if the event is an A2A event."""
        return isinstance(self.event, TaskStatusUpdateEvent | TaskArtifactUpdateEvent)

    def is_internal_event(self) -> bool:
        """Return True if the event is an internal RunEvent."""
        return isinstance(self.event, RunEvent)


class MessageStep(Step):
    """A2A compatible message step with enhanced support for mixed content."""

    is_streaming: bool = False
    message_stream: _t.AsyncIterator[str] | None = None
    step_type: StepType = StepType.MESSAGE

    def __post_init__(self):
        if self.message is None and self.message_stream is None:
            raise ValueError("message or message_stream is required for MessageStep")

    async def stream_content(self) -> _t.AsyncGenerator[str]:
        """Stream message content if ``is_streaming`` is True."""
        if self.message_stream is not None:
            async for chunk in self.message_stream:
                yield chunk
            return

        if not self.is_streaming or not self.message:
            return

        for part in self.message.parts:
            if hasattr(part, "text"):
                text = part.text
                chunk_size = 50
                for i in range(0, len(text), chunk_size):
                    yield text[i : i + chunk_size]


class ToolCallStep(Step):
    """Step for executing a tool call with enhanced result handling."""

    step_type: StepType = StepType.TOOL_CALL
    tool_call: ToolUsePart | None = None
    tool_call_result: ToolUsePart | None = None
    success: bool = True

    @model_validator(mode="before")
    @classmethod
    def _convert_calls(cls, data: dict[str, _t.Any]) -> dict[str, _t.Any]:
        call = data.get("tool_call")
        if isinstance(call, ToolCall):
            data["tool_call"] = ToolUsePart(
                type="call",
                id=call.call_id,
                name=call.name,
                input=call.arguments,
            )
        result = data.get("tool_call_result")
        if isinstance(result, ToolCallResult):
            data["tool_call_result"] = ToolUsePart(
                type="result",
                id=result.call_id,
                name=call.name if call else "",
                output=result.output,
                metadata={"error": result.error} if result.error else None,
            )
        return data

    def model_post_init(self, __context: _t.Any) -> None:
        if self.tool_call is None:
            raise ValueError("tool_call (ToolUsePart type='call') is required for ToolCallStep")
        if self.tool_call.type != "call":
            raise ValueError("ToolCallStep.tool_call must be a ToolUsePart with type='call'")

    def add_result(self, result: ToolCallResult) -> None:
        """Attach a ToolCallResult to this step and update success."""
        self.tool_call_result = ToolUsePart(
            type="result",
            id=result.call_id,
            name=self.tool_call.name if self.tool_call else "",
            output=result.output,
            metadata={"error": result.error} if result.error else None,
        )
        self.success = result.success

    def to_message_parts(self) -> tuple[ToolUsePart, ToolUsePart | None]:
        """Return tool call and result parts for message conversion."""
        return self.tool_call, self.tool_call_result


class EventStep(Step):
    """Step for handling A2A events with enhanced content support."""

    step_type: StepType = StepType.EVENT

    def model_post_init(self, __context: _t.Any) -> None:
        if self.event is None:
            raise ValueError("event is required for EventStep")


class ThinkStep(Step):
    """Step for internal agent thinking/reasoning."""

    step_type: StepType = StepType.THINK

    def model_post_init(self, __context: _t.Any) -> None:
        if self.thinking is None:
            self.thinking = ""


class FinishStep(Step):
    """Step indicating task completion with enhanced content support."""

    step_type: StepType = StepType.FINISH
    final_message: Message | None = None
    reason: str | None = None
    success: bool = True
    task: Task | None = None

    def model_post_init(self, __context: _t.Any) -> None:
        if self.reason is None:
            self.reason = "Task completed"
        # Only create a message if none is provided and we have a reason
        if self.message is None and self.final_message is None and self.reason:
            # Create a simple message for the finish step
            msg = Message(
                messageId=f"finish-{self.step_id}",
                role=Role.agent,
                parts=[TextPart(text=self.reason)],
                contextId=None,
                kind="message",
            )
            self.message = msg
            self.final_message = msg

    def create_completion_event(self, task_id: str, context_id: str) -> TaskStatusUpdateEvent:
        """Create a task completion event."""
        completion_status = TaskStatus(
            state=TaskState.completed, message=self.message, timestamp=datetime.now().isoformat()
        )
        return TaskStatusUpdateEvent(
            contextId=context_id, taskId=task_id, status=completion_status, final=True, metadata=self.metadata
        )

class ErrorStep(Step):
    """Step emitted when planning fails."""

    step_type: StepType = StepType.ERROR
    error: str | None = None

