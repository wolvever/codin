# Agent System Design

## Overview

The Agent System is the core of the CoDIN platform, implementing AI agents that follow the Agent-to-Agent (A2A) protocol for communication and coordination. It provides a framework-agnostic interface for building and orchestrating intelligent agents with pluggable planning strategies.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      Agent System                              │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Agent (ABC)   │    │  BaseAgent      │    │  CodeAgent      │ │
│  │                 │    │  (A2A Protocol) │    │ (Specialized)   │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│           │                       │                       │         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ Planner (ABC)   │    │ BasicPlanner    │    │ CodingPlanner   │ │
│  │                 │    │                 │    │                 │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│           │                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ AgentConfig     │    │ AgentFactory    │    │   AgentTypes    │ │
│  │                 │    │                 │    │  (Messages)     │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Interfaces

### Agent Protocol

```python
class Agent(ABC):
    """Abstract base class for all agents."""
    
    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for the agent."""
        pass
    
    @abstractmethod
    async def run(self, input_data: AgentRunInput) -> AsyncIterator[AgentRunOutput]:
        """Process input and yield output messages."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        pass
```

### Planner Protocol

```python
class Planner(ABC):
    """Strategy pattern for agent decision-making."""
    
    @abstractmethod
    async def next(self, state: AgentState) -> AsyncIterator[Step]:
        """Generate next steps based on current state."""
        pass
    
    @abstractmethod
    async def reset(self, state: AgentState) -> None:
        """Reset planner state."""
        pass
```

## Data Flow

### Agent Execution Loop

```mermaid
graph TD
    A[A2A Message] --> B[BaseAgent.run]
    B --> C[Build State]
    C --> D[Planner.next]
    D --> E{Step Type}
    E --> |Message| F[LLM Interaction]
    E --> |Tool| G[Tool Execution]
    E --> |Event| H[Event Emission]
    E --> |Think| I[Internal Processing]
    F --> J[Update State]
    G --> J
    H --> J
    I --> J
    J --> K{Continue?}
    K --> |Yes| D
    K --> |No| L[Generate Output]
    L --> M[AgentRunOutput]
```

### Message Processing

1. **Input Validation**: Validate A2A message format
2. **Context Building**: Retrieve conversation history from memory
3. **Planning Loop**: Iterative step generation and execution
4. **Tool Integration**: Execute tools and process results
5. **Output Generation**: Format response as A2A messages
6. **State Management**: Update conversation state and memory

## Implementation Details

### BaseAgent

The `BaseAgent` class provides the core A2A protocol implementation:

```python
class BaseAgent(Agent):
    def __init__(
        self,
        agent_id: str,
        planner: Planner,
        memory: MemoryService = None,
        mailbox: Mailbox = None
    ):
        self.agent_id = agent_id
        self.planner = planner
        self.memory = memory or MemMemoryService()
        self.mailbox = mailbox or LocalMailbox()
    
    async def run(self, input_data: AgentRunInput) -> AsyncIterator[AgentRunOutput]:
        # Build agent state from input and memory
        state = await self._build_state(input_data)
        
        # Execute planning loop
        async for step in self.planner.next(state):
            result = await self._execute_step(step, state)
            if result:
                yield result
            
            # Check termination conditions
            if self._should_terminate(state):
                break
```

### CodeAgent

Specialized agent for coding tasks with enhanced tool integration:

```python
class CodeAgent(Agent):
    def __init__(
        self,
        llm: LLM,
        memory: MemoryService,
        tool_registry: ToolRegistry,
        sandbox: Sandbox,
        agent_id: str = "code_agent"
    ):
        self.llm = llm
        self.memory = memory
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.agent_id = agent_id
    
    async def run(self, input_data: AgentRunInput) -> AsyncIterator[AgentRunOutput]:
        # Enhanced loop with tool calling and code execution
        async for message in self._conversation_loop(input_data):
            yield AgentRunOutput(
                id=generate_id(),
                messages=[message],
                runner_id=input_data.runner_id,
                request_id=input_data.request_id
            )
```

## Planning Strategies

### BasicPlanner

Simple sequential planning with predefined step types:

```python
class BasicPlanner(Planner):
    async def next(self, state: AgentState) -> AsyncIterator[Step]:
        # Analyze current state
        if not state.messages:
            yield ThinkStep("No previous messages, starting fresh")
            
        # Generate appropriate response step
        last_message = state.messages[-1]
        if last_message.role == Role.USER:
            yield MessageStep(
                recipient="user",
                content="Processing your request..."
            )
```

### CodingPlanner

Specialized planner for coding tasks with tool integration:

```python
class CodingPlanner(Planner):
    async def next(self, state: AgentState) -> AsyncIterator[Step]:
        # Analyze code-related requests
        if self._needs_file_analysis(state):
            yield ToolStep(
                tool_name="read_file",
                parameters={"path": self._extract_file_path(state)}
            )
        
        elif self._needs_code_execution(state):
            yield ToolStep(
                tool_name="run_command",
                parameters={"command": self._extract_command(state)}
            )
        
        else:
            # Standard message response
            yield MessageStep(
                recipient="user",
                content=await self._generate_response(state)
            )
```

## State Management

### AgentState

Encapsulates current agent execution context:

```python
@dataclass
class AgentState:
    agent_id: str
    runner_id: str
    request_id: str
    messages: List[Message]
    tools_available: List[str]
    budget: Budget
    metadata: Dict[str, Any]
    step_count: int = 0
    max_steps: int = 100
    
    def is_budget_exceeded(self) -> bool:
        return (
            self.step_count >= self.max_steps or
            self.budget.is_exceeded()
        )
```

### Memory Integration

Agents integrate with the memory system for context persistence:

```python
class AgentMemoryManager:
    def __init__(self, memory: MemoryService, agent_id: str):
        self.memory = memory
        self.agent_id = agent_id
    
    async def load_conversation_history(self, runner_id: str) -> List[Message]:
        """Load conversation history for the current session."""
        return await self.memory.get_history(
            session_id=f"{self.agent_id}:{runner_id}"
        )
    
    async def save_message(self, message: Message, runner_id: str) -> None:
        """Save message to conversation history."""
        await self.memory.add_message(
            message,
            session_id=f"{self.agent_id}:{runner_id}"
        )
```

## Event System

### Event Types

Agents emit events for monitoring and coordination:

```python
class EventType(Enum):
    AGENT_START = "agent_start"
    AGENT_STOP = "agent_stop"
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    TOOL_CALL = "tool_call"
    ERROR = "error"
```

### Event Emission

```python
class EventEmitter:
    def __init__(self):
        self.listeners: Dict[EventType, List[Callable]] = defaultdict(list)
    
    async def emit(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Emit event to all registered listeners."""
        for listener in self.listeners[event_type]:
            try:
                await listener(event_type, data)
            except Exception as e:
                logger.error(f"Error in event listener: {e}")
    
    def add_listener(self, event_type: EventType, listener: Callable) -> None:
        """Register event listener."""
        self.listeners[event_type].append(listener)
```

## A2A Protocol Compliance

### Message Format

All agent communications follow A2A protocol standards:

```python
@dataclass
class AgentRunInput:
    messages: List[Message]
    runner_id: str
    request_id: str
    tools: Optional[List[str]] = None
    budget: Optional[Budget] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AgentRunOutput:
    id: str
    messages: List[Message]
    runner_id: str
    request_id: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

### Error Handling

Standardized error propagation and recovery:

```python
class AgentError(Exception):
    def __init__(self, message: str, error_code: str, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

class AgentErrorHandler:
    async def handle_error(self, error: Exception, state: AgentState) -> AgentRunOutput:
        """Convert errors to standardized A2A error responses."""
        if isinstance(error, AgentError):
            return AgentRunOutput(
                id=generate_id(),
                messages=[],
                runner_id=state.runner_id,
                request_id=state.request_id,
                error=f"{error.error_code}: {error.message}"
            )
        else:
            return AgentRunOutput(
                id=generate_id(),
                messages=[],
                runner_id=state.runner_id,
                request_id=state.request_id,
                error=f"INTERNAL_ERROR: {str(error)}"
            )
```

## Configuration

### Agent Configuration

```python
@dataclass
class AgentConfig:
    agent_id: str
    planner_type: str = "basic"
    max_steps: int = 100
    memory_enabled: bool = True
    tool_approval_required: bool = False
    budget_limits: Dict[str, int] = field(default_factory=dict)
    custom_prompts: Dict[str, str] = field(default_factory=dict)
```

### Factory Pattern

```python
class AgentFactory:
    @staticmethod
    async def create_agent(config: AgentConfig, **dependencies) -> Agent:
        """Create agent instance based on configuration."""
        if config.agent_id == "code_agent":
            return CodeAgent(
                llm=dependencies["llm"],
                memory=dependencies["memory"],
                tool_registry=dependencies["tool_registry"],
                sandbox=dependencies["sandbox"]
            )
        else:
            planner = PlannerFactory.create_planner(config.planner_type)
            return BaseAgent(
                agent_id=config.agent_id,
                planner=planner,
                memory=dependencies.get("memory"),
                mailbox=dependencies.get("mailbox")
            )
```

## Integration Points

### Tool System Integration

Agents interact with tools through the registry:

```python
class AgentToolIntegration:
    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        context: ToolContext
    ) -> ToolCallResult:
        """Execute tool with agent context."""
        tool = await self.tool_registry.get_tool(tool_name)
        if not tool:
            raise AgentError(f"Tool not found: {tool_name}", "TOOL_NOT_FOUND")
        
        return await tool.execute(context, **parameters)
```

### Sandbox Integration

Code execution through sandbox environments:

```python
class AgentSandboxIntegration:
    async def execute_code(
        self, 
        code: str, 
        language: str = "python"
    ) -> SandboxResult:
        """Execute code in secure sandbox."""
        return await self.sandbox.run(
            SandboxRequest(
                code=code,
                language=language,
                timeout=30.0
            )
        )
```

## Performance Considerations

### Concurrency

- Agents are designed for concurrent execution
- State isolation prevents race conditions
- Async/await throughout for non-blocking operations

### Memory Efficiency

- Lazy loading of conversation history
- Automatic memory chunking for long conversations
- Cleanup of temporary state after execution

### Scalability

- Stateless agent design enables horizontal scaling
- Actor model isolation for fault tolerance
- Pluggable backends for different deployment scenarios

## Testing Strategy

### Unit Tests

- Mock dependencies for isolated testing
- Test each planner strategy independently
- Validate A2A protocol compliance

### Integration Tests

- End-to-end message processing
- Tool integration scenarios
- Error handling and recovery

### Performance Tests

- Concurrent agent execution
- Memory usage profiling
- Response time benchmarking

This design provides a flexible, scalable foundation for building AI agents that can work together in the CoDIN platform while maintaining protocol compliance and operational safety.