import asyncio
import time

import pytest

import codin.actor.supervisor as scheduler
from codin.actor.dispatcher import LocalDispatcher
from codin.actor.supervisor import ActorInfo, LocalActorManager
from codin.actor.work_stealing import WorkStealingActorSystem
from codin.agent.base import Planner
from codin.agent.base_agent import BaseAgent
from codin.agent.types import AgentRunInput, AgentRunOutput, Message, Role, TextPart


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
async def test_work_stealing_actor_system():
    scheduler.Agent = BaseAgent
    ActorInfo.model_rebuild()
    manager = LocalActorManager(agent_factory=factory)
    dispatcher = LocalDispatcher(manager)
    dispatcher._create_message_from_a2a = message_converter.__get__(dispatcher, LocalDispatcher)
    dispatcher.agents_to_create = [("a", "ctx")]

    system = WorkStealingActorSystem(
        dispatcher,
        manager,
        workers=2,
        choose_worker=lambda req: 0,
    )
    await system.up()

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
    fut1 = await system.submit(a2a_request)
    fut2 = await system.submit(a2a_request)
    rid1 = await fut1
    rid2 = await fut2

    async def wait_done(rid):
        while True:
            status = await dispatcher.get_status(rid)
            if status.status != "started":
                return status
            await asyncio.sleep(0.05)

    s1, s2 = await asyncio.gather(wait_done(rid1), wait_done(rid2))
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert s1.status == "completed"
    assert s2.status == "completed"

    await system.down()
