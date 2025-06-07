import asyncio
import time
import pytest
import codin.actor.supervisor as scheduler

from codin.actor.dispatcher import LocalDispatcher
from codin.actor.supervisor import LocalActorManager, ActorInfo
from codin.agent.base_agent import BaseAgent
from codin.agent.base import Planner
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
        yield AgentRunOutput(
            id="1",
            result=Message(
                messageId="m",
                role=Role.agent,
                parts=[TextPart(text=self.agent_id)],
                contextId=input_data.session_id or "ctx",
                kind="message",
            ),
        )


async def factory(agent_type: str, key: str) -> BaseAgent:
    return SleepAgent(agent_id=f"{agent_type}:{key}", name=agent_type, description="d", planner=DummyPlanner())


def message_converter(self, data: dict, ctx: str) -> Message:
    return Message(
        messageId=data.get("messageId"),
        role=Role(data.get("role", "user")),
        parts=[TextPart(text=data.get("parts", [{"text": ""}])[0]["text"])],
        contextId=ctx,
        kind="message",
    )


@pytest.mark.asyncio
async def test_multiple_agents_concurrent():
    scheduler.Agent = BaseAgent
    ActorInfo.model_rebuild()
    manager = LocalActorManager(agent_factory=factory)
    dispatcher = LocalDispatcher(manager)
    dispatcher._create_message_from_a2a = message_converter.__get__(dispatcher, LocalDispatcher)
    dispatcher.agents_to_create = [("a1", "ctx"), ("a2", "ctx")]

    a2a_request = {
        "contextId": "ctx",
        "message": {
            "messageId": "u1",
            "role": "user",
            "parts": [{"kind": "text", "text": "hi"}],
            "kind": "message",
        },
    }

    start = time.monotonic()
    runner_id = await dispatcher.submit(a2a_request)
    while True:
        status = await dispatcher.get_status(runner_id)
        if status.status != "started":
            break
        await asyncio.sleep(0.1)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    outputs = status.metadata["outputs"]
    assert len(outputs) == 2
    texts = {o["output"]["result"]["parts"][0]["text"] for o in outputs}
    assert texts == {"a1:ctx", "a2:ctx"}

