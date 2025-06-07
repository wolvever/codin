import asyncio
import pytest

from codin.actor.dispatcher import LocalDispatcher
from codin.actor.scheduler import LocalActorManager, ActorInfo
import codin.actor.scheduler as scheduler
from codin.agent.base_agent import BaseAgent
from codin.agent.base import Planner
from codin.agent.types import (
    Message,
    TextPart,
    Role,
    AgentRunInput,
    AgentRunOutput,
    Task,
)
from a2a.types import TaskStatusUpdateEvent, TaskStatus, TaskState
from codin.model.base import BaseLLM


class DummyPlanner(Planner):
    async def next(self, state):
        if False:
            yield  # pragma: no cover

    async def reset(self, state):
        pass


class DummyLLM(BaseLLM):
    @classmethod
    def supported_models(cls):
        return ["mock-llm"]

    async def prepare(self):
        pass

    async def generate(self, *args, **kwargs):
        return "ok"

    async def generate_with_tools(self, *args, **kwargs):
        return {"content": "ok", "tool_calls": []}


class ExampleAgent(BaseAgent):
    async def run(self, input_data: AgentRunInput):
        ctx = input_data.session_id or "ctx"
        await asyncio.sleep(1)
        yield AgentRunOutput(
            id="1",
            result=Message(
                messageId="m1",
                role=Role.agent,
                parts=[TextPart(text="hi")],
                contextId=ctx,
                kind="message",
            ),
        )
        await asyncio.sleep(1)
        yield AgentRunOutput(
            id="2",
            result=TaskStatusUpdateEvent(
                contextId=ctx,
                taskId="t1",
                status=TaskStatus(state=TaskState.working),
                final=False,
            ),
        )
        await asyncio.sleep(1)
        yield AgentRunOutput(
            id="3",
            result=TaskStatusUpdateEvent(
                contextId=ctx,
                taskId="t1",
                status=TaskStatus(state=TaskState.working),
                final=False,
            ),
        )
        await asyncio.sleep(1)
        yield AgentRunOutput(
            id="4",
            result=Task(
                id="t1",
                contextId=ctx,
                status=TaskStatus(state=TaskState.completed),
                message=Message(
                    messageId="m2",
                    role=Role.agent,
                    parts=[TextPart(text="done")],
                    contextId=ctx,
                    kind="message",
                ),
            ),
        )


async def agent_factory(agent_type: str, key: str) -> BaseAgent:
    return ExampleAgent(
        agent_id=f"{agent_type}:{key}",
        name="test",
        description="d",
        planner=DummyPlanner(),
        llm=DummyLLM("mock-llm"),
    )


def message_converter(self, data: dict, ctx: str) -> Message:
    return Message(
        messageId=data.get("messageId"),
        role=Role(data.get("role", "user")),
        parts=[TextPart(text=data.get("parts", [{"text": ""}])[0]["text"])],
        contextId=ctx,
        kind="message",
    )


@pytest.mark.asyncio
async def test_dispatcher_reconnect():
    scheduler.Agent = BaseAgent
    ActorInfo.model_rebuild()
    manager = LocalActorManager(agent_factory=agent_factory)
    dispatcher = LocalDispatcher(manager)
    dispatcher._create_message_from_a2a = message_converter.__get__(dispatcher, LocalDispatcher)

    a2a_request = {
        "contextId": "c1",
        "message": {
            "messageId": "u1",
            "role": "user",
            "parts": [{"kind": "text", "text": "hi"}],
            "kind": "message",
        },
    }

    runner_id = await dispatcher.submit(a2a_request)

    # Simulate client disconnect while agent runs
    await asyncio.sleep(5.5)

    status = await dispatcher.get_status(runner_id)
    assert status.status == "completed"
    outputs = status.metadata["outputs"]
    assert len(outputs) == 4
    assert outputs[-1]["output"]["result"]["status"]["state"] == TaskState.completed
