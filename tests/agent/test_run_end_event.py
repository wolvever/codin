import asyncio
import pytest
from codin.agent.base_agent import BaseAgent
from codin.agent.base import Planner
from codin.agent.types import AgentRunInput, Message, TextPart, Role, FinishStep
from codin.actor.mailbox import LocalMailbox

class FinishPlanner(Planner):
    async def next(self, state):
        yield FinishStep(step_id="finish")
    async def reset(self, state):
        pass

@pytest.mark.asyncio
async def test_run_end_event_emitted():
    mailbox = LocalMailbox()
    agent = BaseAgent(
        name="TestAgent",
        description="test",
        planner=FinishPlanner(),
        mailbox=mailbox,
    )

    user_msg = Message(
        messageId="u1",
        role=Role.user,
        parts=[TextPart(text="hi")],
        contextId="sess",
        kind="message",
    )
    input_data = AgentRunInput(message=user_msg, session_id="sess")

    async for _ in agent.run(input_data):
        pass

    events = []
    try:
        while True:
            msgs = await mailbox.get_outbox(timeout=0.1)
            for m in msgs:
                if m.metadata and m.metadata.get("event_type"):
                    events.append(m.metadata["event_type"])
    except asyncio.TimeoutError:
        pass

    assert "run_end" in events
