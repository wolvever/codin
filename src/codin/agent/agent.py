import time
import uuid
from typing import AsyncIterator, List

from .base import Agent, Planner
from .types import (
    AgentRunInput,
    AgentRunOutput,
    State,
    MessageStep,
    FinishStep,
    Message,
    Role,
    TextPart,
    RunConfig,
)
from .session import Session


class BasicAgent(Agent):
    """Stateful agent execution loop controller."""

    def __init__(self, planner: Planner, *, tools: List = None, memory=None) -> None:
        super().__init__(name="BasicAgent", description="Simple agent", tools=tools)
        self.planner = planner
        self.memory = memory

    async def run(self, input: AgentRunInput) -> AsyncIterator[AgentRunOutput]:
        session = Session(memory=self.memory, tools=self.tools, config=input.metadata.get("config", {}) if input.metadata else RunConfig())
        state = State(session_id=input.session_id or uuid.uuid4().hex, agent_id=self.id, history=await session.get_history(), pending=[input.message], tools=self.tools)
        steps = await self.planner.plan(state)
        for step in steps:
            if isinstance(step, MessageStep) and step.message:
                await session.add_message(step.message)
                yield step.message
            if isinstance(step, FinishStep):
                break
        return

    async def cleanup(self) -> None:
        pass
