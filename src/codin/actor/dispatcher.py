"""Dispatcher for handling HTTP requests and agent orchestration.

This module provides request dispatching services for routing A2A requests
to appropriate agents, managing agent lifecycles, and coordinating
multi-agent workflows in the codin framework.
"""

from __future__ import annotations

import asyncio
import typing as _t
import uuid
from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field

from ..agent.types import Message, Role, TextPart
from .mailbox import Mailbox
from .supervisor import ActorSupervisor
from .utils import make_message
from ..replay import BaseReplay

if _t.TYPE_CHECKING:
    from ..agent.base import Agent
    from ..agent.types import AgentRunInput
    # _t.Callable and _t.Optional are available

__all__ = [
    'DispatchRequest',
    'DispatchResult',
    'Dispatcher',
    'LocalDispatcher',
]


class DispatchRequest(BaseModel):
    """Request for dispatching work to agents."""

    request_id: str
    a2a_request: dict[str, _t.Any]
    metadata: dict[str, _t.Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class DispatchResult(BaseModel):
    """Result from dispatching work to agents."""

    runner_id: str
    request_id: str
    status: str  # "started", "completed", "failed"
    agents: list[str] = Field(default_factory=list)  # Agent IDs involved
    metadata: dict[str, _t.Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class Dispatcher(ABC):
    """Abstract dispatcher protocol from design document."""

    @abstractmethod
    async def submit(self, a2a_request: dict) -> str:
        """Submit an A2A request and return runner_id."""

    @abstractmethod
    async def signal(self, agent_id: str, ctrl: str) -> None:
        """Send a control signal to an agent."""

    @abstractmethod
    async def get_status(self, runner_id: str) -> DispatchResult | None:
        """Get the status of a running request."""

    @abstractmethod
    async def list_active_runs(self) -> list[DispatchResult]:
        """List all active runs."""


class LocalDispatcher(Dispatcher):
    """Local implementation of dispatcher using asyncio."""

    def __init__(self, actor_manager: ActorSupervisor, max_concurrency: int | None = None):
        self.actor_manager = actor_manager
        self._active_runs: dict[str, DispatchResult] = {}
        self._run_tasks: dict[str, asyncio.Task] = {}
        # per-run stream queues for realtime outputs
        self._run_streams: dict[str, asyncio.Queue] = {}
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrency) if max_concurrency else None
        )

    async def submit(self, a2a_request: dict) -> str:
        """Submit an A2A request and return runner_id.

        Args:
            a2a_request: The A2A (Agent-to-Agent or Application-to-Agent) request dictionary.
                         This dictionary can optionally contain a 'replay_factory' key
                         (e.g., `a2a_request['replay_factory'] = lambda rid: FileReplay(session_id=rid)`).
                         If provided, its value should be a callable that accepts a `runner_id` (str)
                         and returns an object conforming to the `BaseReplay` interface.
                         This factory will be called with the generated `runner_id` for this request,
                         and the returned replay instance will be used for recording message exchanges.
                         The `LocalDispatcher` will call `cleanup()` on this instance
                         upon completion or failure of the request handling within `_handle_request`.
        """
        request_id = str(uuid.uuid4())
        runner_id = f'run_{request_id[:8]}'

        # Create dispatch request
        dispatch_request = DispatchRequest(request_id=request_id, a2a_request=a2a_request)

        # Initialize result tracking
        result = DispatchResult(runner_id=runner_id, request_id=request_id, status='started')
        self._active_runs[runner_id] = result

        # Stream queue for this run
        stream_queue: asyncio.Queue = asyncio.Queue()
        self._run_streams[runner_id] = stream_queue

        # Get replay_factory and create replay_instance if factory is provided
        replay_factory: _t.Optional[_t.Callable[[str], BaseReplay]] = a2a_request.get('replay_factory')
        replay_instance: _t.Optional[BaseReplay] = None
        if replay_factory and callable(replay_factory):
            replay_instance = replay_factory(runner_id)

        # Start async task to handle the request
        task = asyncio.create_task(
            self._handle_request(dispatch_request, result, stream_queue, replay_instance)
        )
        self._run_tasks[runner_id] = task

        return runner_id

    async def signal(self, agent_id: str, ctrl: str) -> None:
        """Send a control signal to an agent."""
        # Find the agent through actor manager
        actor_info = await self.actor_manager.info(agent_id)
        if not actor_info:
            raise ValueError(f'Agent {agent_id} not found')

        agent = actor_info.agent

        # Send control signal through agent's mailbox if it has one
        if hasattr(agent, 'mailbox') and isinstance(agent.mailbox, Mailbox):
            control_message = Message(
                messageId=str(uuid.uuid4()),
                role=Role.system,
                parts=[TextPart(text=f'Control signal: {ctrl}')],
                contextId=agent_id,
                kind='message',
                metadata={'control': ctrl, 'signal_type': 'dispatcher'},
            )
            await agent.mailbox.put_inbox(control_message)

    async def get_status(self, runner_id: str) -> DispatchResult | None:
        """Get the status of a running request."""
        return self._active_runs.get(runner_id)

    async def list_active_runs(self) -> list[DispatchResult]:
        """List all active runs."""
        return list(self._active_runs.values())

    async def _handle_request(
        self,
        request: DispatchRequest,
        result: DispatchResult,
        stream_queue: asyncio.Queue,
        replay_instance: _t.Optional[BaseReplay] = None,
    ) -> None:
        """Handle an A2A request by creating and orchestrating agents."""
        created_agents: list[str] = []
        try:
            a2a_data = request.a2a_request
            context_id = a2a_data.get('contextId', str(uuid.uuid4()))
            message_data = a2a_data.get('message', {})
            message = make_message(message_data, context_id)

            agents_to_create = getattr(
                self,
                'agents_to_create',
                [('main_agent', context_id)],
            )

            agents: list[Agent] = []
            for agent_type, key in agents_to_create:
                agent_obj = await self.actor_manager.acquire(agent_type, key)
                agents.append(agent_obj)
                agent_id = (
                    agent_obj.agent_id if hasattr(agent_obj, 'agent_id') else f'{agent_type}:{key}'
                )
                created_agents.append(agent_id)
                result.agents.append(agent_id)

            from ..agent.types import AgentRunInput
            run_input = AgentRunInput(
                id=request.request_id, message=message, session_id=context_id, metadata=request.metadata
            )

            from ..agent.concurrent_runner import ConcurrentRunner
            from ..agent.runner import AgentRunner

            runner_group = ConcurrentRunner()
            for agent_item in agents:
                runner_group.add_runner(AgentRunner(agent_item, replay_backend=replay_instance))

            await runner_group.start_all()

            tasks = [
                asyncio.create_task(self._run_agent_with_semaphore(a, run_input, stream_queue, result))
                for a in agents
            ]
            errors = await asyncio.gather(*tasks, return_exceptions=True)

            await runner_group.stop_all()

            exceptions = [e for e in errors if isinstance(e, Exception)]
            if exceptions:
                result.status = 'failed'
                result.metadata['errors'] = [str(e) for e in exceptions]
            else:
                result.status = 'completed'

        except Exception as e:
            result.status = 'failed'
            result.metadata['error'] = str(e)
            result.metadata['error_timestamp'] = datetime.now().isoformat()

        finally:
            for agent_id in created_agents:
                await self.actor_manager.release(agent_id)

            if replay_instance:
                await replay_instance.cleanup()

            await stream_queue.put(None)

            if result.runner_id in self._run_tasks:
                del self._run_tasks[result.runner_id]
            if result.runner_id in self._run_streams:
                del self._run_streams[result.runner_id]

    async def _run_agent(
        self,
        agent: Agent,
        run_input: AgentRunInput,
        stream_queue: asyncio.Queue,
        result: DispatchResult,
    ) -> None:
        async for output in agent.run(run_input):
            await stream_queue.put(
                output.dict() if hasattr(output, 'dict') else str(output)
            )

            if 'outputs' not in result.metadata:
                result.metadata['outputs'] = []
            result.metadata['outputs'].append(
                {
                    'timestamp': datetime.now().isoformat(),
                    'output': output.dict() if hasattr(output, 'dict') else str(output),
                }
            )

    async def _run_agent_with_semaphore(
        self,
        agent: Agent,
        run_input: AgentRunInput,
        stream_queue: asyncio.Queue,
        result: DispatchResult,
    ) -> None:
        if self._semaphore is None:
            await self._run_agent(agent, run_input, stream_queue, result)
            return

        async with self._semaphore:
            await self._run_agent(agent, run_input, stream_queue, result)


    def get_stream_queue(self, runner_id: str) -> asyncio.Queue | None:
        return self._run_streams.get(runner_id)

    async def cleanup(self) -> None:
        for task in self._run_tasks.values():
            if not task.done():
                task.cancel()

        if self._run_tasks:
            await asyncio.gather(*self._run_tasks.values(), return_exceptions=True)

        self._run_tasks.clear()
        self._active_runs.clear()
        self._run_streams.clear()
