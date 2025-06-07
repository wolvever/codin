"""Task management and event processing system.

This module provides a task manager that handles task submission, scheduling,
and execution with observability features including OpenTelemetry and Prometheus
metrics. It supports FIFO task processing and agent coordination.
"""

import asyncio
import logging
import typing as _t
import uuid

from collections import deque
from datetime import datetime

# Prometheus imports
import prometheus_client as prom

from a2a.types import Task, TaskState, TaskStatus

# OpenTelemetry imports
from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from ..agent.base import Agent


__all__ = [
    'TaskManager',
]

# Setup logging
logger = logging.getLogger(__name__)

# Create tracer and metrics
tracer = trace.get_tracer('codin.task.manager')
meter = metrics.get_meter('codin.task.manager')

# Define OpenTelemetry metrics
task_submit_counter = meter.create_counter(
    name='task_submits',
    description='Number of task submissions',
    unit='1',
)

task_complete_counter = meter.create_counter(
    name='task_completes',
    description='Number of task completions',
    unit='1',
)

task_duration = meter.create_histogram(
    name='task_duration',
    description='Duration of task executions',
    unit='s',
)

task_queue_length = meter.create_up_down_counter(
    name='task_queue_length',
    description='Current length of task queue',
    unit='1',
)

# Define Prometheus metrics - use try/except to avoid duplicate registration
try:
    prom_tasks_submitted = prom.Counter('codin_tasks_submitted_total', 'Number of tasks submitted', ['agent_id'])
except ValueError:
    prom_tasks_submitted = prom.REGISTRY._names_to_collectors['codin_tasks_submitted_total']

try:
    prom_tasks_completed = prom.Counter(
        'codin_tasks_completed_total', 'Number of tasks completed', ['agent_id', 'status']
    )
except ValueError:
    prom_tasks_completed = prom.REGISTRY._names_to_collectors['codin_tasks_completed_total']

try:
    prom_task_duration = prom.Histogram(
        'codin_task_duration_seconds',
        'Duration of task executions',
        ['agent_id'],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
    )
except ValueError:
    prom_task_duration = prom.REGISTRY._names_to_collectors['codin_task_duration_seconds']

try:
    prom_task_queue_length = prom.Gauge('codin_task_queue_length', 'Current length of task queue')
except ValueError:
    prom_task_queue_length = prom.REGISTRY._names_to_collectors['codin_task_queue_length']


class TaskManager:
    """Simple FIFO task scheduler & event bus for agents (MVP)."""

    def __init__(self):
        """Initialize a task manager with empty queues and metrics."""
        self._queue: deque[tuple[Task, Agent]] = deque()
        self._lock = asyncio.Lock()
        self._running_tasks: dict[str, tuple[Task, Agent, datetime]] = {}
        self._completed_tasks: dict[str, Task] = {}
        logger.info('TaskManager initialized')

    async def submit(self, goal: str, agent: Agent, *, context: dict[str, _t.Any] | None = None) -> str:
        """Submit a task to be executed by the specified agent.

        Args:
            goal: The goal of the task
            agent: The agent to execute the task
            context: Optional context data for the task

        Returns:
            The task ID
        """
        with tracer.start_as_current_span('submit_task') as span:
            task_id = str(uuid.uuid4())
            span.set_attribute('task.id', task_id)
            span.set_attribute('task.goal', goal)
            span.set_attribute('agent.id', agent.id or 'unknown')

            status = TaskStatus(state=TaskState.SUBMITTED, timestamp=datetime.utcnow())
            task = Task(id=task_id, status=status, metadata={'goal': goal, 'context': context})

            async with self._lock:
                self._queue.append((task, agent))
                queue_length = len(self._queue)
                task_queue_length.add(1)
                prom_task_queue_length.set(queue_length)

            logger.info(f'Task {task_id} submitted for agent {agent.id or "unknown"}: {goal}')
            task_submit_counter.add(1, {'agent_id': agent.id or 'unknown'})
            prom_tasks_submitted.labels(agent_id=agent.id or 'unknown').inc()

            return task_id

    async def run_forever(self):
        """Continuously process tasks sequentially (for MVP)."""
        logger.info('TaskManager starting task processing loop')

        while True:
            if not self._queue:
                await asyncio.sleep(0.1)
                continue

            async with self._lock:
                if not self._queue:
                    continue
                task, agent = self._queue.popleft()
                queue_length = len(self._queue)
                task_queue_length.add(-1)
                prom_task_queue_length.set(queue_length)
                self._running_tasks[task.id] = (task, agent, datetime.utcnow())

            logger.info(f'Processing task {task.id} with agent {agent.id or "unknown"}')

            with tracer.start_as_current_span(f'process_task_{task.id}') as span:
                span.set_attribute('task.id', task.id)
                span.set_attribute('agent.id', agent.id or 'unknown')
                span.set_attribute('task.initial_state', task.status.state)

                start_time = datetime.utcnow()

                try:
                    updated_task = await agent.run_task(task)

                    # Calculate duration and record metrics
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    task_duration.record(
                        duration, {'agent_id': agent.id or 'unknown', 'status': updated_task.status.state}
                    )
                    prom_task_duration.labels(agent_id=agent.id or 'unknown').observe(duration)

                    # Log completion
                    logger.info(f'Task {task.id} completed with state={updated_task.status.state} in {duration:.2f}s')
                    task_complete_counter.add(
                        1, {'agent_id': agent.id or 'unknown', 'status': updated_task.status.state}
                    )
                    prom_tasks_completed.labels(agent_id=agent.id or 'unknown', status=updated_task.status.state).inc()

                    # Store completed task
                    self._completed_tasks[task.id] = updated_task

                    # Update span
                    span.set_attribute('task.final_state', updated_task.status.state)
                    span.set_attribute('task.duration', duration)

                except Exception as e:
                    # Handle exceptions
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    logger.exception(f'Error processing task {task.id}: {e!s}')

                    # Create failure status
                    failure_status = TaskStatus(
                        state=TaskState.FAILED,
                        timestamp=datetime.utcnow(),
                    )
                    updated_task = Task(
                        id=task.id,
                        status=failure_status,
                        metadata={**(task.metadata or {}), 'error': str(e)},
                    )

                    # Record metrics
                    task_duration.record(duration, {'agent_id': agent.id or 'unknown', 'status': 'failed'})
                    task_complete_counter.add(1, {'agent_id': agent.id or 'unknown', 'status': 'failed'})
                    prom_task_duration.labels(agent_id=agent.id or 'unknown').observe(duration)
                    prom_tasks_completed.labels(agent_id=agent.id or 'unknown', status='failed').inc()

                    # Store failed task
                    self._completed_tasks[task.id] = updated_task

                    # Update span
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute('task.final_state', 'failed')
                    span.set_attribute('task.duration', duration)

                finally:
                    # Remove from running tasks
                    if task.id in self._running_tasks:
                        del self._running_tasks[task.id]

                # Handle task publishing (in real impl via A2A push)
                await self._handle_task(updated_task)

    async def _handle_task(self, task: Task):
        """Handle a completed task.

        In a real implementation, this would publish events via A2A push, store task results, etc.

        Args:
            task: The completed task
        """
        logger.debug(f'Task {task.id} processed with state={task.status.state}')
        # Here we would add code for event publishing, persistence, etc.

    async def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: The ID of the task to get

        Returns:
            The task if found, otherwise None
        """
        # Check completed tasks
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id]

        # Check running tasks
        if task_id in self._running_tasks:
            return self._running_tasks[task_id][0]

        # Check queue
        async with self._lock:
            for queued_task, _ in self._queue:
                if queued_task.id == task_id:
                    return queued_task

        return None

    async def get_queue_length(self) -> int:
        """Get the current queue length.

        Returns:
            The number of tasks in the queue
        """
        async with self._lock:
            return len(self._queue)

    async def get_running_tasks_count(self) -> int:
        """Get the number of currently running tasks.

        Returns:
            The number of running tasks
        """
        return len(self._running_tasks)
