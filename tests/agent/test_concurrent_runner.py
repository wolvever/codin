import asyncio
import time
import pytest

from codin.agent.base import Planner
from codin.agent.base_agent import BaseAgent
from codin.agent.runner import AgentRunner
from codin.agent.concurrent_runner import ConcurrentRunner
from codin.agent.types import AgentRunInput, AgentRunOutput, Message, TextPart, Role


class DummyPlanner(Planner):
    async def next(self, state):
        if False:
            yield  # pragma: no cover

    async def reset(self, state):
        pass


class SleepAgent(BaseAgent):
    async def run(self, input_data: AgentRunInput):
        await asyncio.sleep(0.5)
        msg = Message(
            messageId="m",
            role=Role.agent,
            parts=[TextPart(text=self.agent_id)],
            contextId=input_data.session_id or "ctx",
            kind="message",
        )
        await self.mailbox.put_outbox(msg)
        yield AgentRunOutput(id="1", result=msg)


@pytest.mark.asyncio
async def test_concurrent_runner_start_stop():
    agent1 = SleepAgent(agent_id="a1", name="a1", description="d", planner=DummyPlanner())
    agent2 = SleepAgent(agent_id="a2", name="a2", description="d", planner=DummyPlanner())
    runner1 = AgentRunner(agent1)
    runner2 = AgentRunner(agent2)

    group = ConcurrentRunner()
    group.add_runner(runner1)
    group.add_runner(runner2)

    await group.start_all()

    msg = Message(messageId="u", role=Role.user, parts=[TextPart(text="hi")], contextId="ctx", kind="message")
    await asyncio.gather(runner1.mailbox.put_inbox(msg), runner2.mailbox.put_inbox(msg))

    async def drain(runner):
        out = await runner.mailbox.get_outbox(timeout=2.0)
        return out[0].parts[0].text

    start = time.monotonic()
    results = await asyncio.gather(drain(runner1), drain(runner2))
    elapsed = time.monotonic() - start

    assert set(results) == {"a1", "a2"}
    assert elapsed < 1.0

    await group.stop_all()
