# Core Architecture: Planner, Agent, State, and Services

## Overview

This document clarifies the core design principles for the codin agent system, focusing on the separation of concerns between stateless **Planner**, stateful **Agent**, comprehensive **State**, and the orchestration of various services.

## Design Principles

1. **Planner is stateless** - Only reads from State, never modifies it
2. **Agent is stateful** - Manages State changes and service orchestration  
3. **State is comprehensive** - Contains all context needed for planning and execution
4. **Steps are A2A compatible** - Support both Message and Event types with streaming
5. **Services are injected** - Memory, Tools, Sessions, Replay, etc. are dependencies

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
    created_at: datetime
    iteration: int = 0
    
    # Conversation history (readonly reference)
    history: ChatHistory  # From MemoryService
    
    # Memory and artifact references (readonly)
    memory_ref: MemoryService  # Readonly reference 
    artifact_ref: ArtifactService  # Readonly reference
    
    # Tools and execution context
    tools: list[Tool]
    tool_call_results: list[ToolCallResult] = field(default_factory=list)
    
    # Performance metrics
    metrics: Metrics = field(default_factory=Metrics)
    
    # Budget constraints
    config: AgentConfig = field(default_factory=AgentConfig)
    
    # Task management
    current_task: Task | None = None
    task_status: TaskStatus = TaskStatus.PENDING
    
    # Additional context
    context: dict[str, Any] = field(default_factory=dict)
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

1. **Service orchestration** - MemoryService, ToolRegistry, SessionService, etc.
2. **State management** - Creates, updates, and maintains State 
3. **Execution loop** - Calls Planner and executes Steps
4. **Task lifecycle** - Start, pause, resume, cancel tasks

```python
class Agent:
    """Stateful agent that orchestrates Planner and services."""
    
    def __init__(
        self,
        planner: Planner,
        memory_service: MemoryService,
        tool_registry: ToolRegistry, 
        session_service: SessionService,
        replay_service: ReplayService,
        mailbox: Mailbox,
        **kwargs
    ):
        # Store service references
        self.planner = planner
        self.memory_service = memory_service
        # ... other services
    
    async def run(self, input: AgentRunInput) -> AsyncGenerator[AgentRunOutput, None]:
        """Main execution method."""
        
        # 1. Get or create session
        session = await self._get_or_create_session(input.session_id)
        
        # 2. Start new task if needed
        task = await self._start_task_if_needed(input, session)
        
        # 3. Build comprehensive State
        state = await self._build_state(session, task, input)
        
        # 4. Execute planning loop until completion
        async for output in self._execute_planning_loop(state):
            yield output
    
    async def _build_state(self, session, task, input) -> State:
        """Build comprehensive State from session and services."""
        
        # Get chat history from MemoryService
        history = await self.memory_service.get_chat_history(session.session_id)
        
        # Get available tools
        tools = self.tool_registry.get_tools()
        
        # Build metrics and config
        metrics = Metrics(
            iterations=session.iteration_count,
            tokens_used=session.total_tokens,
            cost_used=session.total_cost,
            elapsed_seconds=session.elapsed_time
        )
        
        config = AgentConfig(
            turn_budget=100,
            token_budget=100000,
            cost_budget=10.0,
            time_budget=300.0
        )
        
        return State(
            session_id=session.session_id,
            task_id=task.id if task else None,
            agent_id=self.agent_id,
            created_at=session.created_at,
            iteration=session.iteration_count,
            history=history,  # Readonly reference
            memory_ref=self.memory_service,  # Readonly reference
            artifact_ref=self.artifact_service,  # Readonly reference
            tools=tools,
            metrics=metrics,
            config=config,
            current_task=task
        )
```

## Service Interfaces

### MemoryService

```python
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

class ChatHistory:
    """Readonly chat history interface."""
    
    def get_recent_messages(self, count: int = 10) -> list[Message]:
        """Get recent messages for context."""
        ...
    
    def search_messages(self, query: str) -> list[Message]:
        """Search through message history."""
        ...
```

### SessionService & ReplayService

```python
class SessionService(abc.ABC):
    """Service for managing agent sessions."""
    
    @abc.abstractmethod
    async def get_or_create_session(self, session_id: str | None) -> Session:
        """Get existing or create new session."""
        ...

class ReplayService(abc.ABC):
    """Service for recording execution replay logs."""
    
    @abc.abstractmethod
    async def record_step(self, step: Step, result: Any) -> None:
        """Record step execution for replay."""
        ...
```

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
    query: str
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

1. **Agent receives AgentRunInput** with optional session_id, task_id
2. **Agent gets Session** from SessionService (create if needed)
3. **Agent starts Task** if this is a new query
4. **Agent builds State** from Session, Task, and service references
5. **Agent calls Planner.next(state)** to get Steps
6. **Agent executes Steps** (messages, tool calls, events)
7. **Agent updates State** based on Step results
8. **Agent continues loop** until FinishStep or budget constraints
9. **Agent records to ReplayService** and updates Session

## Key Benefits

- **Clear separation of concerns** - Planner reads, Agent writes
- **Comprehensive context** - State contains everything needed for planning
- **A2A compatibility** - Steps work with Agent-to-Agent protocol
- **Service injection** - Easy testing and different implementations  
- **Task lifecycle** - Full support for start/pause/resume/cancel
- **Reproducible execution** - ReplayService enables debugging and analysis 