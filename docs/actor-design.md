# Actor System Design

## Overview

The Actor System implements the foundational concurrency model for the CoDIN platform, providing fault-tolerant, scalable execution of agents through an actor-based architecture. It manages the lifecycle, scheduling, and communication of actors that host and execute AI agents.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Actor System                               │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Dispatcher    │    │ ActorSupervisor │    │  TaskRegistry   │ │
│  │   (Routing)     │◄───┤  (Lifecycle)    │◄───┤   (Tracking)    │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│           │                       │                       │         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Envelope      │    │   ActorInfo     │    │   TaskInfo      │ │
│  │   (Messages)    │    │   (Metadata)    │    │   (State)       │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│           │                       │                       │         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Mailbox       │    │ CallableActor   │    │   WorkStealing  │ │
│  │   (Queue)       │    │   (Protocol)    │    │   (LoadBalance) │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Interfaces

### CallableActor Protocol

```python
class CallableActor(Protocol):
    """Protocol defining the interface for actors."""
    
    async def __call__(self, input_data: ActorRunInput) -> AsyncIterator[ActorRunOutput]:
        """Execute actor with input data and yield outputs."""
        ...
    
    @property
    def capabilities(self) -> Set[str]:
        """Return set of capabilities this actor provides."""
        ...
    
    async def cleanup(self) -> None:
        """Clean up actor resources."""
        ...
```

### Dispatcher Interface

```python
class Dispatcher(ABC):
    """Abstract dispatcher for routing requests to actors."""
    
    @abstractmethod
    async def submit(self, envelope_dict: Dict[Any, Any]) -> str:
        """Submit request envelope for processing."""
        pass
    
    @abstractmethod
    async def signal(self, runner_id: str, signal: str) -> bool:
        """Send control signal to running task."""
        pass
    
    @abstractmethod
    async def get_status(self, runner_id: str) -> Optional[DispatchResult]:
        """Get status of submitted task."""
        pass
    
    @abstractmethod
    async def get_stream_queue(self, runner_id: str) -> asyncio.Queue:
        """Get streaming output queue for task."""
        pass
```

## Message Flow

### Request Processing

```mermaid
graph TD
    A[Client Request] --> B[Envelope Creation]
    B --> C[Dispatcher.submit]
    C --> D[Task Registration]
    D --> E[Actor Acquisition]
    E --> F[Actor Execution]
    F --> G[Result Streaming]
    G --> H[Task Completion]
    H --> I[Resource Cleanup]
```

### Actor Lifecycle

```mermaid
graph LR
    A[Idle] --> B[Acquired]
    B --> C[Active]
    C --> D[Processing]
    D --> E[Completed]
    E --> F[Released]
    F --> A
    C --> G[Failed]
    G --> H[Error Handling]
    H --> F
```

## Implementation Details

### LocalDispatcher

The primary dispatcher implementation for single-node deployment:

```python
class LocalDispatcher(Dispatcher):
    def __init__(
        self,
        supervisor: ActorSupervisor,
        task_registry: TaskRegistry = None
    ):
        self.supervisor = supervisor
        self.task_registry = task_registry or TaskRegistry()
        self._stream_queues: Dict[str, asyncio.Queue] = {}
        self._stop_signals: Set[str] = set()
        self._pause_signals: Set[str] = set()
    
    async def submit(self, envelope_dict: Dict[Any, Any]) -> str:
        """Submit work envelope for processing."""
        envelope = Envelope.from_dict(envelope_dict)
        
        if envelope.kind == EnvelopeKind.WORK:
            return await self._process_work_envelope(envelope)
        elif envelope.kind == EnvelopeKind.CONTROL:
            return await self._process_control_envelope(envelope)
        else:
            raise ValueError(f"Invalid envelope kind: {envelope.kind}")
    
    async def _process_work_envelope(self, envelope: Envelope) -> str:
        """Process work envelope by acquiring actor and executing."""
        payload = envelope.payload
        runner_id = payload.runner_id
        
        # Register task
        task_id = await self.task_registry.add_task(
            runner_id=runner_id,
            request_id=payload.request_id,
            metadata={"agent_id": payload.agent_id}
        )
        
        # Acquire actor
        actor_info = await self.supervisor.acquire(payload.agent_id)
        
        # Execute in background
        asyncio.create_task(self._execute_actor(actor_info, payload, task_id))
        
        return runner_id
```

### ActorSupervisor

Manages actor lifecycle and resource allocation:

```python
class LocalActorManager(ActorSupervisor):
    def __init__(self, max_actors: int = 10):
        self._actors: Dict[str, ActorInfo] = {}
        self._max_actors = max_actors
        self._actor_lock = asyncio.Lock()
    
    async def acquire(self, agent_id: str) -> ActorInfo:
        """Acquire an actor for the specified agent."""
        async with self._actor_lock:
            # Check for idle actor
            if agent_id in self._actors:
                info = self._actors[agent_id]
                if info.status == "idle":
                    info.status = "active"
                    return info
            
            # Create new actor if under limit
            if len(self._actors) < self._max_actors:
                actor_instance = await self._create_actor_instance(agent_id)
                info = ActorInfo(
                    actor_id=agent_id,
                    actor_instance=actor_instance,
                    capabilities=actor_instance.capabilities,
                    status="active"
                )
                self._actors[agent_id] = info
                return info
            
            # Wait for available actor or create new instance
            actor_instance = await self._create_actor_instance(agent_id)
            return ActorInfo(
                actor_id=agent_id,
                actor_instance=actor_instance,
                capabilities=actor_instance.capabilities,
                status="active"
            )
    
    async def release(self, agent_id: str) -> None:
        """Release actor back to idle state."""
        if agent_id in self._actors:
            self._actors[agent_id].status = "idle"
    
    async def cleanup_idle_actors(self, max_idle_time: timedelta = None) -> None:
        """Clean up actors that have been idle too long."""
        max_idle_time = max_idle_time or timedelta(minutes=10)
        current_time = datetime.now()
        
        to_remove = []
        for agent_id, info in self._actors.items():
            if (info.status == "idle" and 
                current_time - info.last_activity > max_idle_time):
                to_remove.append(agent_id)
        
        for agent_id in to_remove:
            await self._actors[agent_id].actor_instance.cleanup()
            del self._actors[agent_id]
```

### TaskRegistry

Tracks task state and execution progress:

```python
class TaskRegistry:
    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._task_lock = asyncio.Lock()
    
    async def add_task(
        self, 
        runner_id: str, 
        request_id: str, 
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add new task to registry."""
        task_id = str(uuid.uuid4())
        
        async with self._task_lock:
            self._tasks[runner_id] = TaskInfo(
                task_id=task_id,
                runner_id=runner_id,
                request_id=request_id,
                state=TaskState.PENDING,
                metadata=metadata or {},
                created_at=datetime.now()
            )
        
        return task_id
    
    async def update_task_state(
        self, 
        runner_id: str, 
        state: TaskState,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Update task state and metadata."""
        async with self._task_lock:
            if runner_id in self._tasks:
                task_info = self._tasks[runner_id]
                task_info.state = state
                task_info.updated_at = datetime.now()
                
                if metadata:
                    task_info.metadata.update(metadata)
    
    async def get_task(self, runner_id: str) -> Optional[TaskInfo]:
        """Get task information."""
        return self._tasks.get(runner_id)
    
    async def list_all_tasks(self) -> List[TaskInfo]:
        """List all tasks in registry."""
        return list(self._tasks.values())
```

## Envelope System

### Envelope Types

```python
class EnvelopeKind(Enum):
    WORK = "work"
    CONTROL = "control"

@dataclass
class Envelope:
    kind: EnvelopeKind
    payload: Union[WorkPayload, ControlPayload]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_dict(cls, data: Dict[Any, Any]) -> "Envelope":
        """Create envelope from dictionary."""
        kind = EnvelopeKind(data["kind"])
        
        if kind == EnvelopeKind.WORK:
            payload = WorkPayload(**data["payload"])
        elif kind == EnvelopeKind.CONTROL:
            payload = ControlPayload(**data["payload"])
        else:
            raise ValueError(f"Unknown envelope kind: {kind}")
        
        return cls(
            kind=kind,
            payload=payload,
            metadata=data.get("metadata", {})
        )
```

### Work Payload

```python
@dataclass
class WorkPayload:
    runner_id: str
    request_id: str
    agent_id: str
    input: Dict[str, Any]
    tools: Optional[List[str]] = None
    budget: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
```

### Control Payload

```python
@dataclass
class ControlPayload:
    action: ControlAction
    runner_id: str
    signal: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ControlAction(Enum):
    SIGNAL = "signal"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
```

## Mailbox System

### Mailbox Interface

```python
class Mailbox(ABC):
    """Abstract mailbox for actor message passing."""
    
    @abstractmethod
    async def send(self, message: Any, recipient: str) -> None:
        """Send message to recipient."""
        pass
    
    @abstractmethod
    async def receive(self, timeout: float = None) -> Optional[Any]:
        """Receive message with optional timeout."""
        pass
    
    @abstractmethod
    async def peek(self) -> Optional[Any]:
        """Peek at next message without removing."""
        pass
```

### LocalMailbox

```python
class LocalMailbox(Mailbox):
    def __init__(self, maxsize: int = 1000):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._actor_id: Optional[str] = None
    
    async def send(self, message: Any, recipient: str) -> None:
        """Send message to local queue."""
        await self._queue.put(message)
    
    async def receive(self, timeout: float = None) -> Optional[Any]:
        """Receive message with timeout."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    async def peek(self) -> Optional[Any]:
        """Peek at next message."""
        if self._queue.empty():
            return None
        
        # Get message and put it back
        message = await self._queue.get()
        # Put back at front (not directly supported, so we use a temporary queue)
        temp_queue = asyncio.Queue()
        await temp_queue.put(message)
        
        # Swap queues
        old_queue = self._queue
        self._queue = temp_queue
        
        # Put remaining messages back
        while not old_queue.empty():
            await self._queue.put(await old_queue.get())
        
        return message
```

## Work Stealing

### Load Balancing Strategy

```python
class WorkStealingSystem:
    def __init__(self, supervisors: List[ActorSupervisor]):
        self.supervisors = supervisors
        self.load_metrics: Dict[str, LoadMetric] = {}
    
    async def select_supervisor(self, agent_id: str) -> ActorSupervisor:
        """Select supervisor based on load balancing."""
        # Update load metrics
        await self._update_load_metrics()
        
        # Find supervisor with lowest load
        min_load = float('inf')
        selected_supervisor = self.supervisors[0]
        
        for supervisor in self.supervisors:
            supervisor_id = id(supervisor)
            if supervisor_id in self.load_metrics:
                load = self.load_metrics[supervisor_id].current_load
                if load < min_load:
                    min_load = load
                    selected_supervisor = supervisor
        
        return selected_supervisor
    
    async def _update_load_metrics(self) -> None:
        """Update load metrics for all supervisors."""
        for supervisor in self.supervisors:
            supervisor_id = id(supervisor)
            active_actors = await supervisor.list()
            active_count = sum(1 for actor in active_actors if actor.status == "active")
            
            self.load_metrics[supervisor_id] = LoadMetric(
                supervisor_id=supervisor_id,
                active_actors=active_count,
                current_load=active_count / supervisor.max_actors
            )
```

## Error Handling

### Fault Tolerance

```python
class SupervisionStrategy:
    @staticmethod
    async def handle_actor_failure(
        actor_info: ActorInfo, 
        error: Exception,
        supervisor: ActorSupervisor
    ) -> None:
        """Handle actor failure with restart strategy."""
        logger.error(f"Actor {actor_info.actor_id} failed: {error}")
        
        # Clean up failed actor
        await actor_info.actor_instance.cleanup()
        
        # Update status
        actor_info.status = "failed"
        actor_info.last_error = str(error)
        
        # Restart actor based on error type
        if isinstance(error, RecoverableError):
            try:
                # Attempt restart
                new_instance = await supervisor._create_actor_instance(
                    actor_info.actor_id
                )
                actor_info.actor_instance = new_instance
                actor_info.status = "idle"
                actor_info.restart_count += 1
                
            except Exception as restart_error:
                logger.error(f"Failed to restart actor: {restart_error}")
                actor_info.status = "dead"
        else:
            # Non-recoverable error
            actor_info.status = "dead"
```

### Circuit Breaker

```python
class ActorCircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        recovery_timeout: float = 300.0
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout
```

## Monitoring & Metrics

### Actor Metrics

```python
@dataclass
class ActorMetrics:
    actor_id: str
    requests_processed: int = 0
    average_processing_time: float = 0.0
    error_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    memory_usage: float = 0.0
    cpu_usage: float = 0.0

class MetricsCollector:
    def __init__(self):
        self.metrics: Dict[str, ActorMetrics] = {}
    
    async def record_request(self, actor_id: str, processing_time: float) -> None:
        """Record successful request processing."""
        if actor_id not in self.metrics:
            self.metrics[actor_id] = ActorMetrics(actor_id=actor_id)
        
        metrics = self.metrics[actor_id]
        metrics.requests_processed += 1
        metrics.average_processing_time = (
            (metrics.average_processing_time * (metrics.requests_processed - 1) + processing_time) /
            metrics.requests_processed
        )
        metrics.last_activity = datetime.now()
    
    async def record_error(self, actor_id: str) -> None:
        """Record error in actor processing."""
        if actor_id not in self.metrics:
            self.metrics[actor_id] = ActorMetrics(actor_id=actor_id)
        
        self.metrics[actor_id].error_count += 1
```

## Configuration

### Actor System Configuration

```python
@dataclass
class ActorSystemConfig:
    max_actors: int = 10
    actor_timeout: float = 300.0
    cleanup_interval: float = 600.0
    enable_work_stealing: bool = False
    enable_circuit_breaker: bool = True
    mailbox_size: int = 1000
    supervision_strategy: str = "one_for_one"
```

## Performance Optimizations

### Connection Pooling

- Reuse actor instances when possible
- Pool management for expensive resources
- Lazy initialization of actors

### Batching

- Batch similar requests to same actor
- Reduce context switching overhead
- Optimize resource utilization

### Caching

- Cache actor metadata and capabilities
- Cache routing decisions
- Cache compiled configurations

## Usage Examples

### Example 1: Agent Task Dispatcher

```python
from codin.actor.dispatcher import LocalDispatcher
from codin.actor.supervisor import LocalActorManager
from codin.actor.envelope_types import Envelope, EnvelopeKind, WorkPayload
from codin.agent.base_agent import BaseAgent

async def setup_agent_dispatcher():
    """Set up actor system for dispatching agent tasks."""
    
    # Create actor supervisor to manage agent instances
    supervisor = LocalActorManager(max_actors=5)
    
    # Create dispatcher for routing tasks to agents
    dispatcher = LocalDispatcher(
        supervisor=supervisor,
        task_registry=TaskRegistry()
    )
    
    return dispatcher, supervisor

async def run_concurrent_tasks():
    """Execute multiple agent tasks concurrently using the actor system."""
    
    dispatcher, supervisor = await setup_agent_dispatcher()
    
    # Task 1: Code review
    code_review_envelope = {
        "kind": "work",
        "payload": {
            "runner_id": "code_review_001",
            "request_id": "req_001", 
            "agent_id": "code_agent",
            "input": {
                "session_id": "review_session",
                "message": {
                    "messageId": "review_msg",
                    "role": "user",
                    "parts": [{
                        "text": "Review the authentication module for security issues"
                    }],
                    "contextId": "review_session",
                    "kind": "message"
                },
                "options": {"enable_tools": True}
            }
        }
    }
    
    # Task 2: Test generation
    test_gen_envelope = {
        "kind": "work",
        "payload": {
            "runner_id": "test_gen_002", 
            "request_id": "req_002",
            "agent_id": "code_agent",
            "input": {
                "session_id": "test_session",
                "message": {
                    "messageId": "test_msg",
                    "role": "user", 
                    "parts": [{
                        "text": "Generate unit tests for the UserService class"
                    }],
                    "contextId": "test_session",
                    "kind": "message"
                },
                "options": {"enable_tools": True}
            }
        }
    }
    
    # Submit tasks to dispatcher
    print("🚀 Submitting tasks to actor system...")
    review_runner = await dispatcher.submit(code_review_envelope)
    test_runner = await dispatcher.submit(test_gen_envelope)
    
    print(f"✅ Code review task: {review_runner}")
    print(f"✅ Test generation task: {test_runner}")
    
    # Monitor task progress
    while True:
        review_status = await dispatcher.get_status(review_runner)
        test_status = await dispatcher.get_status(test_runner)
        
        print(f"📊 Review Status: {review_status.status if review_status else 'unknown'}")
        print(f"📊 Test Status: {test_status.status if test_status else 'unknown'}")
        
        if (review_status and review_status.status in ["completed", "failed"] and
            test_status and test_status.status in ["completed", "failed"]):
            break
            
        await asyncio.sleep(2)
    
    print("🎉 All tasks completed!")
```

### Example 2: Auto-Scaling Agent Workforce

```python
from codin.actor.work_stealing import WorkStealingSystem
from codin.actor.supervisor import ActorSupervisor

class ScalableAgentSystem:
    """Auto-scaling system for agent workloads."""
    
    def __init__(self, min_agents=2, max_agents=10):
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.supervisors = []
        self.dispatcher = None
        self.work_stealer = None
        
    async def initialize(self):
        """Initialize the scalable agent system."""
        
        # Create multiple supervisors for load distribution
        for i in range(3):  # Start with 3 supervisor nodes
            supervisor = LocalActorManager(max_actors=self.max_agents // 3)
            self.supervisors.append(supervisor)
        
        # Set up work stealing for load balancing
        self.work_stealer = WorkStealingSystem(self.supervisors)
        
        # Create dispatcher with work stealing
        self.dispatcher = LocalDispatcher(
            supervisor=self.supervisors[0],  # Primary supervisor
            work_stealer=self.work_stealer
        )
        
        print(f"🏗️ Initialized scalable system with {len(self.supervisors)} supervisors")
    
    async def submit_batch_tasks(self, task_batch):
        """Submit a batch of tasks and handle auto-scaling."""
        
        submitted_tasks = []
        
        for i, task in enumerate(task_batch):
            # Select optimal supervisor based on load
            supervisor = await self.work_stealer.select_supervisor("code_agent")
            
            envelope = {
                "kind": "work",
                "payload": {
                    "runner_id": f"batch_task_{i}",
                    "request_id": f"req_batch_{i}",
                    "agent_id": "code_agent",
                    "input": task
                }
            }
            
            runner_id = await self.dispatcher.submit(envelope)
            submitted_tasks.append(runner_id)
            
            print(f"📤 Submitted task {i+1}/{len(task_batch)}: {runner_id}")
        
        return submitted_tasks
    
    async def monitor_system_health(self):
        """Monitor system health and performance metrics."""
        
        while True:
            total_active = 0
            total_capacity = 0
            
            for supervisor in self.supervisors:
                actors = await supervisor.list()
                active_count = sum(1 for actor in actors if actor.status == "active")
                total_active += active_count
                total_capacity += supervisor.max_actors
            
            utilization = (total_active / total_capacity) * 100 if total_capacity > 0 else 0
            
            print(f"📈 System Utilization: {utilization:.1f}% ({total_active}/{total_capacity})")
            
            # Auto-scaling logic
            if utilization > 80:
                print("⚡ High load detected - scaling up...")
                await self._scale_up()
            elif utilization < 30:
                print("📉 Low load detected - scaling down...")
                await self._scale_down()
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def _scale_up(self):
        """Scale up the system by adding more actors."""
        for supervisor in self.supervisors:
            if supervisor.max_actors < self.max_agents:
                await supervisor.scale_up(2)  # Add 2 more actors
                print(f"🔼 Scaled up supervisor to {supervisor.max_actors} actors")
                break
    
    async def _scale_down(self):
        """Scale down the system by removing idle actors.""" 
        for supervisor in self.supervisors:
            if supervisor.max_actors > self.min_agents:
                await supervisor.scale_down(1)  # Remove 1 actor
                print(f"🔽 Scaled down supervisor to {supervisor.max_actors} actors")
                break

# Usage example
async def run_scalable_development_tasks():
    system = ScalableAgentSystem(min_agents=3, max_agents=15)
    await system.initialize()
    
    # Simulate varying workload
    task_batches = [
        # Light workload
        [{"task": f"Review file_{i}.py"} for i in range(5)],
        # Heavy workload  
        [{"task": f"Refactor module_{i}"} for i in range(20)],
        # Medium workload
        [{"task": f"Generate tests for class_{i}"} for i in range(10)]
    ]
    
    # Start system monitoring
    monitor_task = asyncio.create_task(system.monitor_system_health())
    
    try:
        for batch_num, batch in enumerate(task_batches):
            print(f"\n🔄 Processing batch {batch_num + 1} with {len(batch)} tasks")
            
            # Submit batch
            task_runners = await system.submit_batch_tasks(batch)
            
            # Wait for batch completion
            await system._wait_for_batch_completion(task_runners)
            
            print(f"✅ Batch {batch_num + 1} completed")
            
            # Pause between batches
            await asyncio.sleep(5)
    
    finally:
        monitor_task.cancel()
```

### Example 3: Fault-Tolerant Agent Pipeline

```python
from codin.actor.supervisor import SupervisionStrategy
from codin.actor.types import ActorInfo, TaskState

class DevelopmentPipelineProcessor:
    """Fault-tolerant pipeline for software development tasks."""
    
    def __init__(self):
        self.dispatcher = LocalDispatcher(
            supervisor=LocalActorManager(),
            supervision_strategy=DevelopmentSupervisionStrategy()
        )
        
        # Define pipeline stages in order
        self.pipeline_stages = [
            "requirement_analyzer",
            "architect", 
            "developer",
            "tester",
            "reviewer"
        ]
    
    async def process_development_request(self, project_spec):
        """Process a software development request through the pipeline."""
        
        pipeline_context = {
            "project_id": project_spec["project_id"],
            "requirements": project_spec["requirements"],
            "stages_completed": [],
            "artifacts": {}
        }
        
        for stage_num, stage in enumerate(self.pipeline_stages):
            try:
                print(f"🔄 Stage {stage_num + 1}: {stage}")
                
                # Prepare stage input with context from previous stages
                stage_input = {
                    "session_id": f"{pipeline_context['project_id']}_{stage}",
                    "message": {
                        "messageId": f"stage_{stage}",
                        "role": "user",
                        "parts": [{
                            "text": self._generate_stage_prompt(stage, pipeline_context)
                        }],
                        "contextId": pipeline_context["project_id"],
                        "kind": "message"
                    },
                    "options": {"enable_tools": True}
                }
                
                # Submit to appropriate agent
                envelope = {
                    "kind": "work",
                    "payload": {
                        "runner_id": f"{pipeline_context['project_id']}_{stage}",
                        "request_id": f"req_{stage}",
                        "agent_id": "code_agent",  # Using single agent type for simplicity
                        "input": stage_input
                    }
                }
                
                runner_id = await self.dispatcher.submit(envelope)
                
                # Wait for stage completion with fault tolerance
                stage_result = await self._wait_for_stage_with_retries(
                    runner_id, stage, max_retries=3
                )
                
                # Update pipeline context
                pipeline_context["stages_completed"].append(stage)
                pipeline_context["artifacts"][stage] = stage_result
                
                print(f"✅ Stage {stage} completed successfully")
                
            except Exception as e:
                print(f"❌ Stage {stage} failed: {e}")
                
                # Attempt recovery
                if await self._can_recover_from_failure(stage, e):
                    print(f"🔄 Attempting recovery for stage {stage}")
                    await self._recover_stage(stage, pipeline_context, e)
                else:
                    print(f"💥 Unrecoverable failure at stage {stage}")
                    break
        
        return pipeline_context
    
    def _generate_stage_prompt(self, stage, context):
        """Generate appropriate prompt for each pipeline stage."""
        
        prompts = {
            "requirement_analyzer": f"""
Analyze these software requirements and create a detailed specification:
{context['requirements']}

Provide:
1. Functional requirements breakdown
2. Non-functional requirements
3. Technical constraints
4. Success criteria
""",
            "architect": f"""
Based on the analyzed requirements, design the system architecture:
Previous analysis: {context['artifacts'].get('requirement_analyzer', 'None')}

Provide:
1. High-level system design
2. Component architecture
3. Technology stack recommendations
4. Database schema design
""",
            "developer": f"""
Implement the system based on this architecture:
Architecture: {context['artifacts'].get('architect', 'None')}

Provide:
1. Core application code
2. Database setup scripts
3. Configuration files
4. Documentation
""",
            "tester": f"""
Create comprehensive tests for the implemented system:
Implementation: {context['artifacts'].get('developer', 'None')}

Provide:
1. Unit test suite
2. Integration tests  
3. End-to-end tests
4. Test automation setup
""",
            "reviewer": f"""
Review the complete implementation and provide feedback:
Implementation: {context['artifacts'].get('developer', 'None')}
Tests: {context['artifacts'].get('tester', 'None')}

Provide:
1. Code review feedback
2. Architecture assessment
3. Quality recommendations
4. Deployment readiness checklist
"""
        }
        
        return prompts.get(stage, f"Process stage: {stage}")
    
    async def _wait_for_stage_with_retries(self, runner_id, stage, max_retries=3):
        """Wait for stage completion with retry logic."""
        
        for attempt in range(max_retries):
            try:
                # Wait for completion with timeout
                timeout = 300  # 5 minutes per stage
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    status = await self.dispatcher.get_status(runner_id)
                    
                    if status and status.status == "completed":
                        # Get results from stream
                        stream_queue = await self.dispatcher.get_stream_queue(runner_id)
                        if not stream_queue.empty():
                            result = await stream_queue.get()
                            return result
                        return {"status": "completed", "stage": stage}
                    
                    elif status and status.status == "failed":
                        error_msg = status.metadata.get("error", "Unknown error")
                        raise Exception(f"Stage failed: {error_msg}")
                    
                    await asyncio.sleep(5)
                
                raise TimeoutError(f"Stage {stage} timed out after {timeout} seconds")
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Stage {stage} attempt {attempt + 1} failed: {e}")
                    print(f"🔄 Retrying in {2 ** attempt} seconds...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    async def _can_recover_from_failure(self, stage, error):
        """Determine if a stage failure can be recovered."""
        recoverable_errors = ["timeout", "memory", "temporary"]
        error_str = str(error).lower()
        return any(err in error_str for err in recoverable_errors)
    
    async def _recover_stage(self, stage, context, error):
        """Implement stage-specific recovery logic."""
        print(f"🛠️ Implementing recovery for {stage}: {error}")
        # Recovery implementation would go here

class DevelopmentSupervisionStrategy(SupervisionStrategy):
    """Custom supervision strategy for development pipeline."""
    
    async def handle_actor_failure(self, actor_info, error, supervisor):
        """Handle actor failures in development pipeline."""
        
        print(f"🚨 Actor failure: {actor_info.actor_id} - {error}")
        
        # Restart actor with exponential backoff
        if actor_info.restart_count < 3:
            backoff_time = 2 ** actor_info.restart_count
            print(f"⏳ Restarting {actor_info.actor_id} in {backoff_time} seconds...")
            await asyncio.sleep(backoff_time)
            await supervisor.restart_actor(actor_info.actor_id)
        else:
            print(f"💀 Actor {actor_info.actor_id} exceeded restart limit")
            actor_info.status = "permanently_failed"

# Usage example
async def run_development_pipeline():
    processor = DevelopmentPipelineProcessor()
    
    project_spec = {
        "project_id": "ecommerce_api",
        "requirements": """
Build a REST API for an e-commerce platform with:
- User authentication and authorization
- Product catalog management  
- Shopping cart functionality
- Order processing system
- Payment integration
- Inventory tracking
- Admin dashboard

Technology requirements:
- Python/FastAPI backend
- PostgreSQL database
- Redis for caching
- Docker containerization
- Comprehensive testing
"""
    }
    
    print("🚀 Starting development pipeline...")
    result = await processor.process_development_request(project_spec)
    
    print("\n📋 Pipeline Results:")
    print(f"Project: {result['project_id']}")
    print(f"Completed stages: {', '.join(result['stages_completed'])}")
    print(f"Artifacts generated: {len(result['artifacts'])}")
    
    for stage, artifact in result['artifacts'].items():
        print(f"  {stage}: {type(artifact)}")
```

This actor system design provides a robust foundation for concurrent agent execution with fault tolerance, load balancing, and comprehensive monitoring capabilities.