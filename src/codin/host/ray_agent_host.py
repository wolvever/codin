"""Ray-based agent host implementation for codin agents.

This module provides a distributed agent hosting infrastructure using
Ray for scalable multi-agent deployments and coordination.
"""

from __future__ import annotations

import asyncio
import json
import logging
import typing as _t
import uuid

from datetime import datetime
from pathlib import Path

# Prometheus imports
import prometheus_client as prom

# Ray imports
import ray


# OpenTelemetry imports
from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from ..agent.base import Agent
from ..model.base import BaseLLM
from ..protocol.types import AgentRunInput, AgentRunOutput, Message, TaskState, TextPart
from ..protocol.types import TaskStatus as ProtocolTaskStatus
from ..tool.base import Tool

# Import concrete agent implementations
from ..agent.base_agent import BaseAgent


# Setup logging
logger = logging.getLogger(__name__)

# Create tracer and metrics
tracer = trace.get_tracer('codin.host.ray_agent_host')
meter = metrics.get_meter('codin.host.ray_agent_host')

# Define OpenTelemetry metrics
ray_agent_run_counter = meter.create_counter(
    name='ray_agent_runs',
    description='Number of Ray agent runs',
    unit='1',
)

ray_agent_run_duration = meter.create_histogram(
    name='ray_agent_run_duration',
    description='Duration of Ray agent runs',
    unit='s',
)

# Define Prometheus metrics
prom_ray_agent_runs = prom.Counter('codin_ray_agent_runs_total', 'Number of Ray agent runs', ['agent_id', 'status'])

prom_ray_agent_run_duration = prom.Histogram(
    'codin_ray_agent_run_duration_seconds',
    'Duration of Ray agent runs',
    ['agent_id'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

prom_ray_tasks_queued = prom.Gauge(
    'codin_ray_tasks_queued',
    'Number of tasks currently queued in Ray',
)

prom_ray_cluster_resources = prom.Gauge(
    'codin_ray_cluster_resources', 'Available resources in the Ray cluster', ['resource_type']
)


# Ray remote function to run an agent
@ray.remote
def run_agent(agent_state_dict: dict, agent_input_dict: dict) -> dict:
    """Ray remote function to run an agent.

    Args:
        agent_state_dict: Dictionary representation of agent state
        agent_input_dict: Dictionary representation of agent input

    Returns:
        Dictionary representation of agent output
    """
    import asyncio
    import time

    from ..agent.base import Agent
    from ..protocol.types import AgentRunInput, AgentRunOutput

    # Reconstruct agent from state dict
    agent = Agent._from_dict(agent_state_dict)

    # Reconstruct input from dict
    agent_input = AgentRunInput(**agent_input_dict)

    # Track execution time
    start_time = time.time()

    # Run the agent
    try:
        # Since Ray doesn't support asyncio directly, we need to run using asyncio.run
        output = asyncio.run(agent.run(agent_input))
        status = 'success'
    except Exception as e:
        # Create error output
        from ..protocol.types import Message, TaskState, TaskStatus, TextPart

        error_message = Message(role='assistant', parts=[TextPart(text=f'Error: {e!s}')])
        status = TaskStatus(state=TaskState.FAILED, message=error_message, timestamp=time.time())
        output = AgentRunOutput(status=status, artifacts=None, metadata={'error': str(e)})
        status = 'error'

    # Calculate duration
    duration = time.time() - start_time

    # Return output as dict with metadata
    return {
        'output': output.dict() if hasattr(output, 'dict') else vars(output),
        'execution_time': duration,
        'status': status,
        'agent_id': agent.id,
    }


class RayAgentHost:
    """Host for running agents using Ray for distributed computation.

    This class provides functionality for running agents in a distributed manner
    using Ray as the compute engine.
    """

    def __init__(
        self,
        session_id: str | None = None,
        workspace_dir: str | Path | None = None,
        ray_address: str = 'auto',
        init_ray: bool = True,
    ):
        """Initialize the Ray agent host.

        Args:
            session_id: Optional session ID (generated if not provided)
            workspace_dir: Directory for agent workspace
            ray_address: Address of the Ray cluster (default: "auto")
            init_ray: Whether to initialize Ray if not already initialized
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.agents: dict[str, Agent] = {}
        self.history: list[dict[str, _t.Any]] = []

        # Ensure workspace directory exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Ray if needed
        if init_ray and not ray.is_initialized():
            logger.info(f'Initializing Ray with address: {ray_address}')
            ray.init(address=ray_address)

        # Log Ray cluster resources
        self._update_ray_metrics()
        logger.info(f'RayAgentHost initialized with session_id={self.session_id}, connected to Ray cluster')

    def _update_ray_metrics(self):
        """Update Ray cluster metrics."""
        try:
            resources = ray.cluster_resources()
            for resource, value in resources.items():
                if isinstance(value, (int, float)):
                    prom_ray_cluster_resources.labels(resource_type=resource).set(value)

            # Log available resources
            logger.info(f'Ray cluster resources: {resources}')
        except Exception as e:
            logger.warning(f'Failed to update Ray metrics: {e}')

    def add_agent(self, agent: Agent, agent_id: str | None = None) -> str:
        """Add an agent to the host.

        Args:
            agent: The agent to add
            agent_id: Optional agent ID (generated if not provided)

        Returns:
            The agent ID
        """
        agent_id = agent_id or agent.id or str(uuid.uuid4())

        if agent_id in self.agents:
            logger.warning(f'Agent with ID {agent_id} already exists, will be overwritten')

        self.agents[agent_id] = agent
        logger.info(f'Added agent {agent.name} with ID {agent_id}')

        return agent_id

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID.

        Args:
            agent_id: ID of the agent to get

        Returns:
            The agent, or None if not found
        """
        return self.agents.get(agent_id)

    async def send_message_to_agent(
        self,
        agent_id: str,
        message: str | Message,
        metadata: dict[str, _t.Any] | None = None,
    ) -> AgentRunOutput:
        """Send a message to a specific agent using Ray.

        Args:
            agent_id: ID of the agent to send the message to
            message: Message to send
            metadata: Optional metadata to include with the message

        Returns:
            The agent's response

        Raises:
            ValueError: If the agent is not found
        """
        with tracer.start_as_current_span(f'ray_send_message_to_agent_{agent_id}') as span:
            span.set_attribute('agent.id', agent_id)

            agent = self.get_agent(agent_id)
            if not agent:
                error_msg = f'Agent with ID {agent_id} not found'
                logger.error(error_msg)
                span.set_status(Status(StatusCode.ERROR, error_msg))
                raise ValueError(error_msg)

            # Convert string message to Message object if needed
            if isinstance(message, str):
                message = Message(role='user', parts=[TextPart(text=message)])

            # Add standard metadata
            full_metadata = {
                'session_id': self.session_id,
                'task_id': str(uuid.uuid4()),
                'workspace_dir': str(self.workspace_dir),
                'agent_id': agent_id,
            }

            # Add custom metadata if provided
            if metadata:
                full_metadata.update(metadata)

            # Create the input
            agent_input = AgentRunInput(
                message=message,
                metadata=full_metadata,
            )

            # Record in history
            self.history.append(
                {
                    'timestamp': datetime.utcnow().isoformat(),
                    'direction': 'to_agent',
                    'agent_id': agent_id,
                    'message': message,
                }
            )

            # Prepare agent and input for Ray
            agent_state_dict = agent._to_dict() if hasattr(agent, '_to_dict') else vars(agent)
            agent_input_dict = agent_input.dict() if hasattr(agent_input, 'dict') else vars(agent_input)

            # Increment tasks queued
            prom_ray_tasks_queued.inc()

            # Execute through Ray
            try:
                logger.info(f'Submitting task to Ray for agent {agent_id}')
                start_time = datetime.utcnow()

                # Submit to Ray and get result
                ray_result = await asyncio.to_thread(
                    lambda: ray.get(run_agent.remote(agent_state_dict, agent_input_dict))
                )

                # Record metrics
                duration = (datetime.utcnow() - start_time).total_seconds()
                ray_agent_run_duration.record(duration, {'agent_id': agent_id})
                ray_agent_run_counter.add(1, {'agent_id': agent_id, 'status': ray_result['status']})
                prom_ray_agent_run_duration.labels(agent_id=agent_id).observe(duration)
                prom_ray_agent_runs.labels(agent_id=agent_id, status=ray_result['status']).inc()

                logger.info(
                    f'Ray task completed for agent {agent_id} in {duration:.2f}s with status {ray_result["status"]}'
                )

                # Reconstruct output from result
                from ..protocol.types import AgentRunOutput

                output_dict = ray_result['output']
                output = AgentRunOutput(**output_dict)

                # Record in history
                self.history.append(
                    {
                        'timestamp': datetime.utcnow().isoformat(),
                        'direction': 'from_agent',
                        'agent_id': agent_id,
                        'output': output,
                    }
                )

                return output

            except Exception as e:
                # Log the error
                logger.exception(f'Error running Ray task for agent {agent_id}')

                # Create error response
                error_message = Message(role='assistant', parts=[TextPart(text=f'Ray execution error: {e!s}')])

                # Create task status with error
                status = ProtocolTaskStatus(state=TaskState.FAILED, message=error_message, timestamp=datetime.utcnow())

                error_output = AgentRunOutput(status=status, artifacts=None, metadata={'error': str(e)})

                # Record in history
                self.history.append(
                    {
                        'timestamp': datetime.utcnow().isoformat(),
                        'direction': 'from_agent',
                        'agent_id': agent_id,
                        'output': error_output,
                        'error': str(e),
                    }
                )

                # Add to span
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                return error_output
            finally:
                # Decrement tasks queued
                prom_ray_tasks_queued.dec()

                # Update Ray metrics
                self._update_ray_metrics()

    async def broadcast_message(
        self,
        message: str | Message,
        exclude_agents: list[str] | None = None,
        metadata: dict[str, _t.Any] | None = None,
    ) -> dict[str, AgentRunOutput]:
        """Broadcast a message to all agents using Ray.

        Args:
            message: Message to broadcast
            exclude_agents: Optional list of agent IDs to exclude
            metadata: Optional metadata to include with the message

        Returns:
            Dictionary mapping agent IDs to their responses
        """
        with tracer.start_as_current_span('ray_broadcast_message') as span:
            exclude_agents = exclude_agents or []

            # Convert string message to Message object if needed
            if isinstance(message, str):
                message = Message(role='user', parts=[TextPart(text=message)])

            span.set_attribute('excluded_agents', str(exclude_agents))
            span.set_attribute('agent_count', len(self.agents) - len(exclude_agents))

            # Create tasks for all agents - we'll run these in parallel through Ray
            agent_tasks = []
            for agent_id, agent in self.agents.items():
                if agent_id not in exclude_agents:
                    agent_tasks.append((agent_id, self.send_message_to_agent(agent_id, message, metadata)))

            # Wait for all tasks to complete
            results = {}
            for agent_id, task in agent_tasks:
                try:
                    results[agent_id] = await task
                except Exception as e:
                    logger.exception(f'Error broadcasting message to agent {agent_id}')
                    span.record_exception(e)
                    results[agent_id] = None

            return results

    def save_history(self, output_file: str | Path) -> None:
        """Save the interaction history to a file.

        Args:
            output_file: Path to the output file
        """
        output_path = Path(output_file)

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save history
        with open(output_path, 'w') as f:
            json.dump(
                {
                    'session_id': self.session_id,
                    'agents': {
                        agent_id: {
                            'name': agent.name,
                            'description': agent.description,
                            'id': agent.id,
                        }
                        for agent_id, agent in self.agents.items()
                    },
                    'history': self.history,
                },
                f,
                indent=2,
                default=str,  # Handle non-serializable objects
            )

        logger.info(f'Saved interaction history to {output_path}')

    async def shutdown(self):
        """Shutdown Ray and release resources."""
        if ray.is_initialized():
            logger.info('Shutting down Ray')
            ray.shutdown()


async def create_ray_agent_host(
    llm: BaseLLM,
    agent_specs: list[dict[str, _t.Any]],
    session_id: str | None = None,
    workspace_dir: str | Path | None = None,
    tools: list[Tool] | None = None,
    ray_address: str = 'auto',
    init_ray: bool = True,
) -> RayAgentHost:
    """Create a Ray-based agent host with the specified agents.

    Args:
        llm: Language model to use for all agents
        agent_specs: List of agent specifications
        session_id: Optional session ID
        workspace_dir: Directory for agent workspace
        tools: List of tools to provide to the agents
        ray_address: Address of the Ray cluster
        init_ray: Whether to initialize Ray

    Returns:
        A RayAgentHost instance with the specified agents
    """
    with tracer.start_as_current_span('create_ray_agent_host') as span:
        # Prepare the LLM
        await llm.prepare()

        # Create the host
        host = RayAgentHost(
            session_id=session_id,
            workspace_dir=workspace_dir,
            ray_address=ray_address,
            init_ray=init_ray,
        )

        span.set_attribute('agent_count', len(agent_specs))

        # Add agents
        for spec in agent_specs:
            agent_id = spec.get('id') or str(uuid.uuid4())
            agent_name = spec.get('name', 'Agent')
            agent_desc = spec.get('description', '')

            # Create a basic agent - use concrete BaseAgent implementation
            from ..memory.base import MemMemoryService
            from ..agent.base_planner import BasePlanner
            
            # Create a basic planner for the BaseAgent
            planner = BasePlanner(llm=llm)
            
            agent = BaseAgent(
                agent_id=agent_id,
                name=agent_name,
                description=agent_desc,
                planner=planner,
                memory=MemMemoryService(),
                tools=tools or [],
                llm=llm,
            )

            # Add the agent to the host
            host.add_agent(
                agent=agent,
                agent_id=agent_id,
            )

        logger.info(f'Created RayAgentHost with {len(agent_specs)} agents')
        return host
