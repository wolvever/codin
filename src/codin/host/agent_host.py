"""Agent host implementation for codin agents.

This module provides the core agent hosting infrastructure for running
and managing individual agents with lifecycle management and communication.
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


# OpenTelemetry imports
from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from ..agent.base import Agent
from ..agent.dag_planner import DAGExecutor, DAGPlanner
from ..model.base import BaseLLM
from ..protocol.types import AgentRunInput, AgentRunOutput, Message, TaskState, TextPart
from ..protocol.types import TaskStatus as ProtocolTaskStatus
from ..tool.base import Tool

# Import concrete agent implementations
from ..agent.base_agent import BaseAgent


# Setup logging
logger = logging.getLogger(__name__)

# Create tracer and metrics
tracer = trace.get_tracer('codin.host.agent_host')
meter = metrics.get_meter('codin.host.agent_host')

# Define OpenTelemetry metrics
agent_run_counter = meter.create_counter(
    name='agent_runs',
    description='Number of agent runs',
    unit='1',
)

agent_run_duration = meter.create_histogram(
    name='agent_run_duration',
    description='Duration of agent runs',
    unit='s',
)

agent_errors = meter.create_counter(
    name='agent_errors',
    description='Number of agent errors',
    unit='1',
)

# Define Prometheus metrics
prom_agent_runs = prom.Counter('codin_agent_runs_total', 'Number of agent runs', ['agent_id', 'status'])

prom_agent_run_duration = prom.Histogram(
    'codin_agent_run_duration_seconds',
    'Duration of agent runs',
    ['agent_id'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

prom_agent_errors = prom.Counter('codin_agent_errors_total', 'Number of agent errors', ['agent_id', 'error_type'])

prom_messages_sent = prom.Counter('codin_messages_sent_total', 'Number of messages sent', ['agent_id'])

prom_messages_received = prom.Counter('codin_messages_received_total', 'Number of messages received', ['agent_id'])


class AgentHost:
    """Host for running one or more codin agents.

    This class provides functionality for running one or multiple agents that can collaborate.
    It handles the lifecycle of the agents, including initialization, execution, and cleanup.
    """

    def __init__(
        self,
        session_id: str | None = None,
        workspace_dir: str | Path | None = None,
        interactive: bool = True,
    ) -> None:
        """Initialize the agent host.

        Args:
            session_id: Optional session ID (generated if not provided)
            workspace_dir: Directory for agent workspace
            interactive: Whether to run in interactive mode
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.interactive = interactive
        self.agents: dict[str, Agent] = {}
        self.messages: list[dict[str, _t.Any]] = []
        self.history: list[dict[str, _t.Any]] = []

        # Ensure workspace directory exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f'Initialized AgentHost with session_id={self.session_id}, workspace_dir={self.workspace_dir}')

    def add_agent(
        self,
        agent: Agent,
        agent_id: str | None = None,
    ) -> str:
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
        """Send a message to a specific agent.

        Args:
            agent_id: ID of the agent to send the message to
            message: Message to send
            metadata: Optional metadata to include with the message

        Returns:
            The agent's response

        Raises:
            ValueError: If the agent is not found
        """
        with tracer.start_as_current_span(f'send_message_to_agent_{agent_id}') as span:
            span.set_attribute('agent.id', agent_id)

            agent = self.get_agent(agent_id)
            if not agent:
                error_msg = f'Agent with ID {agent_id} not found'
                logger.error(error_msg)
                span.set_status(Status(StatusCode.ERROR, error_msg))
                prom_agent_errors.labels(agent_id=agent_id, error_type='agent_not_found').inc()
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

            # Record the message
            self.messages.append(
                {
                    'timestamp': datetime.utcnow().isoformat(),
                    'to_agent_id': agent_id,
                    'message': message,
                    'metadata': full_metadata,
                }
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

            prom_messages_sent.labels(agent_id=agent_id).inc()

            # Create the input
            agent_input = AgentRunInput(
                message=message,
                metadata=full_metadata,
            )

            # Send to agent and measure time
            span.set_attribute('message.role', message.role)
            start_time = datetime.utcnow()

            try:
                output = await agent.run(agent_input)

                # Calculate duration
                duration = (datetime.utcnow() - start_time).total_seconds()
                agent_run_duration.record(duration, {'agent_id': agent_id})
                prom_agent_run_duration.labels(agent_id=agent_id).observe(duration)
                agent_run_counter.add(1, {'agent_id': agent_id, 'status': 'success'})
                prom_agent_runs.labels(agent_id=agent_id, status='success').inc()

                # Record the response
                self.messages.append(
                    {
                        'timestamp': datetime.utcnow().isoformat(),
                        'from_agent_id': agent_id,
                        'output': output,
                    }
                )

                # Record in history
                self.history.append(
                    {
                        'timestamp': datetime.utcnow().isoformat(),
                        'direction': 'from_agent',
                        'agent_id': agent_id,
                        'output': output,
                    }
                )

                prom_messages_received.labels(agent_id=agent_id).inc()

                return output

            except Exception as e:
                # Log the error
                logger.exception(f'Error running agent {agent_id}')

                # Record metrics
                agent_run_counter.add(1, {'agent_id': agent_id, 'status': 'error'})
                agent_errors.add(1, {'agent_id': agent_id, 'error': str(e)[:100]})
                prom_agent_runs.labels(agent_id=agent_id, status='error').inc()
                prom_agent_errors.labels(agent_id=agent_id, error_type='exception').inc()

                # Create error response
                error_message = Message(role='assistant', parts=[TextPart(text=f'Error: {e!s}')])

                # Create task status with error
                status = ProtocolTaskStatus(state=TaskState.FAILED, message=error_message, timestamp=datetime.utcnow())

                error_output = AgentRunOutput(status=status, artifacts=None, metadata={'error': str(e)})

                # Record in messages
                self.messages.append(
                    {
                        'timestamp': datetime.utcnow().isoformat(),
                        'from_agent_id': agent_id,
                        'output': error_output,
                        'error': str(e),
                    }
                )

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

                prom_messages_received.labels(agent_id=agent_id).inc()

                # Add to span
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                return error_output

    async def broadcast_message(
        self,
        message: str | Message,
        exclude_agents: list[str] | None = None,
        metadata: dict[str, _t.Any] | None = None,
    ) -> dict[str, AgentRunOutput]:
        """Broadcast a message to all agents.

        Args:
            message: Message to broadcast
            exclude_agents: Optional list of agent IDs to exclude
            metadata: Optional metadata to include with the message

        Returns:
            Dictionary mapping agent IDs to their responses
        """
        with tracer.start_as_current_span('broadcast_message') as span:
            exclude_agents = exclude_agents or []

            # Convert string message to Message object if needed
            if isinstance(message, str):
                message = Message(role='user', parts=[TextPart(text=message)])

            span.set_attribute('excluded_agents', str(exclude_agents))
            span.set_attribute('agent_count', len(self.agents) - len(exclude_agents))

            # Create tasks for all agents
            tasks = []
            for agent_id, agent in self.agents.items():
                if agent_id not in exclude_agents:
                    task = self.send_message_to_agent(agent_id, message, metadata)
                    tasks.append((agent_id, asyncio.create_task(task)))

            # Wait for all tasks to complete
            results = {}
            for agent_id, task in tasks:
                try:
                    results[agent_id] = await task
                except Exception as e:
                    logger.exception(f'Error broadcasting message to agent {agent_id}')
                    span.record_exception(e)
                    results[agent_id] = None

            return results

    async def run_conversation(
        self,
        initial_prompt: str,
        max_turns: int = 10,
        coordinator_agent_id: str | None = None,
    ) -> list[dict[str, _t.Any]]:
        """Run a multi-agent conversation.

        Args:
            initial_prompt: Initial prompt to start the conversation
            max_turns: Maximum number of conversation turns
            coordinator_agent_id: Optional ID of an agent to use as coordinator

        Returns:
            List of conversation messages
        """
        with tracer.start_as_current_span('run_conversation') as span:
            span.set_attribute('max_turns', max_turns)
            span.set_attribute('has_coordinator', coordinator_agent_id is not None)

            if not self.agents:
                error_msg = 'No agents added to the host'
                logger.error(error_msg)
                span.set_status(Status(StatusCode.ERROR, error_msg))
                raise ValueError(error_msg)

            conversation: list[dict[str, _t.Any]] = []

            # Start with the initial prompt
            if coordinator_agent_id:
                # Send to coordinator
                coordinator = self.get_agent(coordinator_agent_id)
                if not coordinator:
                    error_msg = f'Coordinator agent with ID {coordinator_agent_id} not found'
                    logger.error(error_msg)
                    span.set_status(Status(StatusCode.ERROR, error_msg))
                    raise ValueError(error_msg)

                response = await self.send_message_to_agent(
                    coordinator_agent_id,
                    initial_prompt,
                    {'role': 'coordinator'},
                )

                conversation.append(
                    {
                        'turn': 0,
                        'agent_id': coordinator_agent_id,
                        'prompt': initial_prompt,
                        'response': response,
                    }
                )
            else:
                # Broadcast to all agents
                responses = await self.broadcast_message(initial_prompt)

                for agent_id, response in responses.items():
                    conversation.append(
                        {
                            'turn': 0,
                            'agent_id': agent_id,
                            'prompt': initial_prompt,
                            'response': response,
                        }
                    )

            # Continue the conversation for max_turns
            for turn in range(1, max_turns):
                with tracer.start_as_current_span(f'conversation_turn_{turn}') as turn_span:
                    turn_span.set_attribute('turn', turn)

                    # In a real implementation, you would have logic here to determine
                    # which agent should speak next and what they should say based on
                    # the conversation so far

                    # For now, we'll just have a simple back-and-forth if there's a coordinator
                    if coordinator_agent_id:
                        # Get the last response from the coordinator
                        last_response = conversation[-1]['response']

                        # Extract text from the response
                        text = ''
                        if last_response and last_response.status.message:
                            for part in last_response.status.message.parts:
                                if hasattr(part, 'text'):
                                    text += part.text

                        # Determine the next agent to speak (simple round-robin)
                        other_agents = [a_id for a_id in self.agents.keys() if a_id != coordinator_agent_id]
                        if not other_agents:
                            logger.info('No other agents to continue conversation')
                            break

                        next_agent_id = other_agents[turn % len(other_agents)]
                        turn_span.set_attribute('next_agent', next_agent_id)

                        # Send the coordinator's message to the next agent
                        response = await self.send_message_to_agent(
                            next_agent_id,
                            text,
                            {'turn': turn, 'from_agent_id': coordinator_agent_id},
                        )

                        conversation.append(
                            {
                                'turn': turn,
                                'agent_id': next_agent_id,
                                'prompt': text,
                                'response': response,
                            }
                        )

                        # Send the response back to the coordinator
                        if response and response.status.message:
                            response_text = ''
                            for part in response.status.message.parts:
                                if hasattr(part, 'text'):
                                    response_text += part.text

                            coordinator_response = await self.send_message_to_agent(
                                coordinator_agent_id,
                                response_text,
                                {'turn': turn, 'from_agent_id': next_agent_id},
                            )

                            conversation.append(
                                {
                                    'turn': turn,
                                    'agent_id': coordinator_agent_id,
                                    'prompt': response_text,
                                    'response': coordinator_response,
                                }
                            )
                    else:
                        # Without a coordinator, we'll just have each agent respond to the initial prompt
                        # This is a simplified implementation - in a real system, you would have more
                        # sophisticated conversation management
                        logger.info(f'No coordinator specified, ending conversation after turn {turn}')
                        break

            logger.info(f'Conversation completed with {len(conversation)} exchanges')
            return conversation

    async def run_once(self, agent_id: str, message: str | Message) -> AgentRunOutput:
        """Run a single agent once with the given message.

        Args:
            agent_id: ID of the agent to run
            message: The message to send to the agent

        Returns:
            The agent's response
        """
        return await self.send_message_to_agent(agent_id, message)

    async def run_interactive(self, agent_id: str) -> None:
        """Run the agent in interactive mode, accepting user input.

        Args:
            agent_id: ID of the agent to run interactively
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f'Agent with ID {agent_id} not found')

        print(f'Starting interactive session with {agent.name}')
        print(f'Session ID: {self.session_id}')
        print(f'Workspace: {self.workspace_dir}')
        print("Type 'exit' or 'quit' to end the session")
        print('-' * 50)

        while True:
            try:
                # Get user input
                user_input = input('\nYou: ')

                # Check for exit command
                if user_input.lower() in ['exit', 'quit']:
                    print('Ending session.')
                    break

                # Run the agent
                output = await self.run_once(agent_id, user_input)

                # Display the response
                print('\nAgent:', end=' ')

                if output.status.message:
                    for part in output.status.message.parts:
                        if hasattr(part, 'text'):
                            print(part.text)

                # Display artifacts if any
                if output.artifacts:
                    print('\nArtifacts:')
                    for artifact in output.artifacts:
                        print(f'- {artifact.name}: {artifact.description}')

            except KeyboardInterrupt:
                print('\nSession interrupted. Ending session.')
                break
            except Exception as e:
                print(f'\nError: {e!s}')

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
                    'messages': self.messages,
                    'history': self.history,
                },
                f,
                indent=2,
                default=str,  # Handle non-serializable objects
            )

        logger.info(f'Saved interaction history to {output_path}')


async def create_agent_host(
    llm: BaseLLM,
    agent_specs: list[dict[str, _t.Any]],
    session_id: str | None = None,
    workspace_dir: str | Path | None = None,
    tools: list[Tool] | None = None,
) -> AgentHost:
    """Create an agent host with the specified agents.

    Args:
        llm: Language model to use for all agents
        agent_specs: List of agent specifications
        session_id: Optional session ID
        workspace_dir: Directory for agent workspace
        tools: List of tools to provide to the agents

    Returns:
        An AgentHost instance with the specified agents
    """
    with tracer.start_as_current_span('create_agent_host') as span:
        # Prepare the LLM
        await llm.prepare()

        # Create the host
        host = AgentHost(
            session_id=session_id,
            workspace_dir=workspace_dir,
        )

        span.set_attribute('agent_count', len(agent_specs))

        # Add agents
        for spec in agent_specs:
            # Create the agent based on the spec
            # This is a placeholder - in a real implementation, you would
            # create different types of agents based on the spec

            agent_id = spec.get('id') or str(uuid.uuid4())
            agent_name = spec.get('name', 'Agent')
            agent_desc = spec.get('description', '')
            agent_type = spec.get('type', 'basic')

            span.set_attribute(f'agent.{agent_id}.type', agent_type)

            # Create different types of agents
            if agent_type == 'planner':
                agent = DAGPlanner(
                    llm=llm,
                    name=agent_name,
                    description=agent_desc,
                    tools=tools or [],
                )
            elif agent_type == 'executor':
                agent = DAGExecutor(
                    llm=llm,
                    name=agent_name,
                    description=agent_desc,
                    tools=tools or [],
                )
            else:
                # Basic agent - use concrete BaseAgent implementation
                from ..memory.base import MemMemoryService
                from ..agent.code_planner import CodePlanner
                
                # Create a basic planner for the BaseAgent
                planner = CodePlanner(llm=llm)
                
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

        logger.info(f'Created AgentHost with {len(agent_specs)} agents')
        return host


async def create_single_agent_host(
    llm: BaseLLM,
    agent_name: str,
    agent_description: str = '',
    tools: list[Tool] | None = None,
    session_id: str | None = None,
    workspace_dir: str | Path | None = None,
    interactive: bool = True,
) -> AgentHost:
    """Create a host with a single agent.

    Args:
        llm: Language model to use
        agent_name: Name of the agent
        agent_description: Description of the agent
        tools: List of tools to provide to the agent
        session_id: Optional session ID
        workspace_dir: Directory for agent workspace
        interactive: Whether to run in interactive mode

    Returns:
        An AgentHost instance with a single agent
    """
    with tracer.start_as_current_span('create_single_agent_host') as span:
        span.set_attribute('agent_name', agent_name)

        # Create a single agent spec
        agent_specs = [{'id': str(uuid.uuid4()), 'name': agent_name, 'description': agent_description, 'type': 'basic'}]

        # Create the host
        host = await create_agent_host(
            llm=llm,
            agent_specs=agent_specs,
            session_id=session_id,
            workspace_dir=workspace_dir,
            tools=tools,
        )

        host.interactive = interactive

        return host


async def create_dag_agent_host(
    llm: BaseLLM,
    tools: list[Tool] | None = None,
    session_id: str | None = None,
    workspace_dir: str | Path | None = None,
    interactive: bool = True,
) -> AgentHost:
    """Create a host with a DAG-based agent (planner + executor).

    Args:
        llm: Language model to use
        tools: List of tools to provide to the agent
        session_id: Optional session ID
        workspace_dir: Directory for agent workspace
        interactive: Whether to run in interactive mode

    Returns:
        An AgentHost instance with a DAG-based agent
    """
    with tracer.start_as_current_span('create_dag_agent_host') as span:
        # Create DAG agent specs
        agent_specs = [
            {
                'id': 'planner',
                'name': 'Codin Planner',
                'description': 'Creates plans for coding tasks',
                'type': 'planner',
            },
            {
                'id': 'executor',
                'name': 'Codin Executor',
                'description': 'Executes coding tasks according to a plan',
                'type': 'executor',
            },
        ]

        # Create the host
        host = await create_agent_host(
            llm=llm,
            agent_specs=agent_specs,
            session_id=session_id,
            workspace_dir=workspace_dir,
            tools=tools,
        )

        host.interactive = interactive

        return host
