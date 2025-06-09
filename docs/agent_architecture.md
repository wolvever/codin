# Core Architecture: Planner, Agent, State, and Services

## Overview

This document clarifies the core design principles for the codin agent system, focusing on the separation of concerns between stateless **Planner**, stateful **Agent**, comprehensive **State**, and the orchestration of various services.

## Design Principles

1. **Planner is stateless** - Only reads from State, never modifies it
2. **Agent is stateful** - Manages State changes and service orchestration
3. **State is comprehensive** - Contains all context needed for planning and execution
4. **Steps are A2A compatible** - Support both Message and Event types with streaming
5. **Services are injected** - Memory, Tools, Replay, etc. are dependencies. Session management is typically handled by components like `SessionManager`.

## Core Components

### State

The `State` object contains all context needed for planning and execution:

```python
@dataclass
class State:
    """Comprehensive state containing all context for planning and execution."""

    # Identity and hierarchy
    session_id: str
    task_id: str | None = None
    parent_task_id: str | None = None
    agent_id: str

    # Temporal context
    created_at: datetime # Typically session creation time
    iteration: int = 0

    # Conversation history (readonly reference)
    history: ChatHistory  # From MemoryService, associated with session_id

    # Memory and artifact references (readonly)
    memory_ref: MemoryService  # Readonly reference
    artifact_ref: ArtifactService  # Readonly reference

    # Tools and execution context
    tools: list[Tool]
    tool_call_results: list[ToolCallResult] = field(default_factory=list)

    # Performance metrics
    metrics: Metrics # Agent-level metrics, potentially aggregated with session metrics

    # Budget constraints
    config: AgentConfig = field(default_factory=AgentConfig)

    # Task management
    current_task: Task | None = None
    task_status: TaskStatus = TaskStatus.PENDING

    # Additional context
    context: dict[str, Any] = field(default_factory=dict) # Can include session.context
```

### Planner Interface

The `Planner` is **stateless** and only reads from State:

```python
class Planner(abc.ABC):
    """Stateless planner that generates Steps from State."""

    @abc.abstractmethod
    async def next(self, state: State) -> AsyncGenerator[Step, None]:
        """Generate execution steps based on current state.

        The planner ONLY reads from state - it never modifies it.
        All state changes are handled by the Agent.
        """
        ...
```

### Step Types (A2A Compatible)

Steps support both A2A Messages and Events:

```python
class StepType(Enum):
    MESSAGE = "message"      # A2A Message (streaming or non-streaming)
    EVENT = "event"          # A2A Event + internal events
    TOOL_CALL = "tool_call"  # Tool execution
    THINK = "think"          # Internal reasoning
    FINISH = "finish"        # Task completion

@dataclass
class MessageStep(Step):
    """A2A compatible message step."""
    message: Message
    is_streaming: bool = False

    async def stream_content(self) -> AsyncGenerator[str, None]:
        """Stream message content if is_streaming=True."""
        ...

@dataclass
class EventStep(Step):
    """A2A compatible event step with internal event support."""
    event: Event  # A2A Event or internal event
    event_type: EventType  # TASK_START, TASK_END, THINK, TOOL_CALL_BEGIN, etc.
```

### Agent Responsibilities

The `Agent` is **stateful** and manages:

1. **Service orchestration** - MemoryService, ToolRegistry, ReplayService, etc.
2. **State management** - Creates, updates, and maintains State
3. **Execution loop** - Calls Planner and executes Steps
4. **Task lifecycle** - Start, pause, resume, cancel tasks

```python
class Agent: # Example structure
    """Stateful agent that orchestrates Planner and services."""

    def __init__(
        self,
        planner: Planner,
        memory_service: MemoryService, # Associated with agent/session
        tool_registry: ToolRegistry,
        replay_service: ReplayService, # Optional
        mailbox: Mailbox,
        **kwargs
    ):
        # Store service references
        self.planner = planner
        self.memory_service = memory_service
        # ... other services

    async def run(self, input: AgentRunInput) -> AsyncGenerator[AgentRunOutput, None]:
        """Main execution method.
        Assumes session_id is provided in input_data or managed by a runner.
        """
        session_id = input.session_id # Or retrieved via a SessionManager

        # Agent might retrieve a Session object or work directly with session_id
        # For example, session_object = await session_manager.get_or_create_session(session_id)
        # This session_object would be the simplified Session (ID, created_at, context, metrics)

        # 2. Start new task if needed
        task = await self._start_task_if_needed(input, session_id) # Pass session_id

        # 3. Build comprehensive State
        # The 'session' parameter here would be the simplified Session object if used,
        # or relevant parts like session_id and session.created_at.
        state = await self._build_state(session_id, task, input)

        # 4. Execute planning loop until completion
        async for output in self._execute_planning_loop(state):
            yield output

    async def _build_state(self, session_id: str, task, input) -> State:
        """Build comprehensive State from session_id, task, input and services."""

        # Get chat history from MemoryService using session_id
        history = await self.memory_service.get_chat_history(session_id)

        # Get available tools
        tools = self.tool_registry.get_tools()

        # Metrics might be sourced from a Session object's metrics if available,
        # or initialized/managed by the agent.
        # Example: session_metrics = session_object.get_metrics_summary()
        agent_metrics = Metrics(
            # iterations might come from agent's own loop or from session.metrics
            # tokens_used, cost_used from LLM calls managed by agent/planner
        )

        config = AgentConfig(
            turn_budget=100,
            token_budget=100000,
            cost_budget=10.0,
            time_budget=300.0
        )

        # session_created_at = session_object.created_at if session_object else datetime.now()
        # session_context = session_object.context if session_object else {}

        return State(
            session_id=session_id,
            task_id=task.id if task else None,
            agent_id=self.agent_id, # Assuming agent has an ID
            created_at=datetime.now(), # Or session_created_at
            iteration=0, # Agent's own iteration count for the current run
            history=history,
            memory_ref=self.memory_service,
            artifact_ref=self.artifact_service, # Assuming self.artifact_service
            tools=tools,
            metrics=agent_metrics,
            config=config,
            current_task=task,
            # context = session_context, # Include context from the session
        )
```

## Service Interfaces

### MemoryService

```python
class MemoryService(abc.ABC):
    """Service for managing conversation memory and chat history, associated with a session_id."""

    @abc.abstractmethod
    async def get_chat_history(self, session_id: str) -> ChatHistory:
        """Get chat history for session."""
        ...

    @abc.abstractmethod
    async def add_message(self, session_id: str, message: Message) -> None:
        """Add message to chat history for a given session."""
        ...

class ChatHistory: # Example structure
    """Readonly chat history interface."""

    def get_recent_messages(self, count: int = 10) -> list[Message]:
        """Get recent messages for context."""
        ...

    def search_messages(self, query: str) -> list[Message]:
        """Search through message history."""
        ...
```

### ReplayService

```python
class ReplayService(abc.ABC):
    """Service for recording execution replay logs."""

    @abc.abstractmethod
    async def record_step(self, step: Step, result: Any) -> None:
        """Record step execution for replay."""
        ...
```

**Note on Session Management:**
The `Session` object, managed by `SessionManager` (from `codin.session.base`), primarily holds `session_id`, `created_at` timestamp, a generic `context` dictionary, and a `metrics` dictionary for high-level session metrics. It does not directly contain conversation messages or detailed state like `task_list`. Agents, like `BaseAgent`, utilize a `MemoryService` (scoped by `session_id`) to manage conversation history and maintain their own comprehensive `State` object during execution.

## Task Management

Tasks support full lifecycle management:

```python
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class Task:
    """Task with full lifecycle support."""
    id: str
    parent_id: str | None
    query: str # Typically the main user request for the task
    status: TaskStatus
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    async def start(self) -> None:
        """Start task execution."""
        ...

    async def pause(self) -> None:
        """Pause task execution."""
        ...

    async def resume(self) -> None:
        """Resume paused task."""
        ...

    async def cancel(self) -> None:
        """Cancel task execution."""
        ...
```

## Execution Flow

1. **Agent receives AgentRunInput** with optional `session_id`, `task_id`.
2. **Session Context is Established**: Typically, a runner or host component ensures a session context is active, often using `SessionManager` to get or create a `Session` object (which holds `session_id`, `created_at`, `context`, `metrics`). The `session_id` is key.
3. **Agent starts Task** if this is a new query, associating it with the `session_id`.
4. **Agent builds State**: This `State` object includes the `session_id`, references to services (like `MemoryService` for history, scoped by `session_id`), and task details.
5. **Agent calls Planner.next(state)** to get Steps.
6. **Agent executes Steps** (messages, tool calls, events).
7. **Agent updates its State** based on Step results. `MemoryService` is used to record messages for the session.
8. **Agent continues loop** until FinishStep or budget constraints.
9. **Agent records to ReplayService** (if used) and potentially updates high-level session metrics in the `Session` object via `SessionManager`.

## Key Benefits

- **Clear separation of concerns** - Planner reads, Agent writes.
- **Comprehensive context** - Agent's `State` object contains everything needed for planning.
- **A2A compatibility** - Steps work with Agent-to-Agent protocol.
- **Service injection** - Easy testing and different implementations for memory, tools, etc.
- **Task lifecycle** - Full support for start/pause/resume/cancel.
- **Reproducible execution** - ReplayService enables debugging and analysis of runs.
