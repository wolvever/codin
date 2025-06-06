"""Dispatcher for handling HTTP requests and agent orchestration."""

import typing as _t
import uuid
import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field

from a2a.types import Message, Role, TextPart

from .scheduler import ActorScheduler
from .mailbox import Mailbox

if _t.TYPE_CHECKING:
    from ..agent.base import Agent
    from ..agent.types import AgentRunInput

__all__ = ["Dispatcher", "LocalDispatcher", "DispatchRequest", "DispatchResult"]


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
        pass
    
    @abstractmethod
    async def signal(self, agent_id: str, ctrl: str) -> None:
        """Send a control signal to an agent."""
        pass
    
    @abstractmethod
    async def get_status(self, runner_id: str) -> DispatchResult | None:
        """Get the status of a running request."""
        pass
    
    @abstractmethod
    async def list_active_runs(self) -> list[DispatchResult]:
        """List all active runs."""
        pass


class LocalDispatcher(Dispatcher):
    """Local implementation of dispatcher using asyncio."""
    
    def __init__(self, actor_manager: ActorScheduler):
        self.actor_manager = actor_manager
        self._active_runs: dict[str, DispatchResult] = {}
        self._run_tasks: dict[str, asyncio.Task] = {}
        
    async def submit(self, a2a_request: dict) -> str:
        """Submit an A2A request and return runner_id."""
        request_id = str(uuid.uuid4())
        runner_id = f"run_{request_id[:8]}"
        
        # Create dispatch request
        dispatch_request = DispatchRequest(
            request_id=request_id,
            a2a_request=a2a_request
        )
        
        # Initialize result tracking
        result = DispatchResult(
            runner_id=runner_id,
            request_id=request_id,
            status="started"
        )
        self._active_runs[runner_id] = result
        
        # Start async task to handle the request
        task = asyncio.create_task(self._handle_request(dispatch_request, result))
        self._run_tasks[runner_id] = task
        
        return runner_id
    
    async def signal(self, agent_id: str, ctrl: str) -> None:
        """Send a control signal to an agent."""
        # Find the agent through actor manager
        actor_info = await self.actor_manager.get_actor_info(agent_id)
        if not actor_info:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = actor_info.agent
        
        # Send control signal through agent's mailbox if it has one
        if hasattr(agent, 'mailbox') and isinstance(agent.mailbox, Mailbox):
            control_message = Message(
                messageId=str(uuid.uuid4()),
                role=Role.system,
                parts=[TextPart(text=f"Control signal: {ctrl}")],
                contextId=agent_id,
                kind="message",
                metadata={"control": ctrl, "signal_type": "dispatcher"}
            )
            await agent.mailbox.put_inbox(control_message)
    
    async def get_status(self, runner_id: str) -> DispatchResult | None:
        """Get the status of a running request."""
        return self._active_runs.get(runner_id)
    
    async def list_active_runs(self) -> list[DispatchResult]:
        """List all active runs."""
        return list(self._active_runs.values())
    
    async def _handle_request(self, request: DispatchRequest, result: DispatchResult) -> None:
        """Handle an A2A request by creating and orchestrating agents."""
        try:
            # Parse A2A request
            a2a_data = request.a2a_request
            
            # Extract key information from A2A request
            context_id = a2a_data.get("contextId", str(uuid.uuid4()))
            message_data = a2a_data.get("message", {})
            
            # Create message from A2A data
            message = self._create_message_from_a2a(message_data, context_id)
            
            # Determine which agents to create (for now, create a single main agent)
            # In the future, this could parse the request to determine multiple agents
            agents_to_create = [
                ("main_agent", context_id),
                # Could add: ("planner_agent", context_id), ("executor_agent", context_id)
            ]
            
            # Create agents through actor manager
            agents: list["Agent"] = []
            for agent_type, key in agents_to_create:
                agent = await self.actor_manager.get_or_create(agent_type, key)
                agents.append(agent)
                result.agents.append(agent.agent_id if hasattr(agent, 'agent_id') else f"{agent_type}:{key}")
            
            # Create agent run input
            from ..agent.types import AgentRunInput
            run_input = AgentRunInput(
                id=request.request_id,
                message=message,
                session_id=context_id,
                metadata=request.metadata
            )
            
            # Run the main agent (could be extended to orchestrate multiple agents)
            main_agent = agents[0]
            async for output in main_agent.run(run_input):
                # Stream outputs - in a real implementation, this would be sent to client
                # For now, just update metadata
                if "outputs" not in result.metadata:
                    result.metadata["outputs"] = []
                result.metadata["outputs"].append({
                    "timestamp": datetime.now().isoformat(),
                    "output": output.dict() if hasattr(output, 'dict') else str(output)
                })
            
            # Mark as completed
            result.status = "completed"
            
        except Exception as e:
            # Mark as failed
            result.status = "failed"
            result.metadata["error"] = str(e)
            result.metadata["error_timestamp"] = datetime.now().isoformat()
        
        finally:
            # Clean up task reference
            if result.runner_id in self._run_tasks:
                del self._run_tasks[result.runner_id]
    
    def _create_message_from_a2a(self, message_data: dict, context_id: str) -> Message:
        """Create a Message object from A2A message data."""
        # Extract text content from parts or fallback to direct text
        parts = []
        if "parts" in message_data:
            for part_data in message_data["parts"]:
                if part_data.get("kind") == "text":
                    parts.append(TextPart(text=part_data.get("text", "")))
        elif "text" in message_data:
            parts.append(TextPart(text=message_data["text"]))
        else:
            parts.append(TextPart(text=""))
        
        return Message(
            messageId=message_data.get("messageId", str(uuid.uuid4())),
            role=Role(message_data.get("role", "user")),
            parts=parts,
            contextId=context_id,
            kind="message",
            metadata=message_data.get("metadata", {})
        )
    
    async def cleanup(self) -> None:
        """Clean up all active runs and tasks."""
        # Cancel all running tasks
        for task in self._run_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation
        if self._run_tasks:
            await asyncio.gather(*self._run_tasks.values(), return_exceptions=True)
        
        self._run_tasks.clear()
        self._active_runs.clear()
