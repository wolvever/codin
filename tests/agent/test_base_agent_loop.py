import pytest
from codin.agent.base_agent import BaseAgent
from codin.agent.base import Planner
from codin.agent.types import RunConfig, AgentRunInput, Message, TextPart, Role, State
from codin.model.base import BaseLLM
from codin.tool.base import Tool
from codin.artifact.base import ArtifactService

# Ensure Pydantic models referencing Tool are fully initialized
State.model_rebuild(force=True, _types_namespace={
    'Tool': Tool,
    'ArtifactService': ArtifactService,
    'Message': Message,
})

class BasePlanner(Planner):
    def __init__(self):
        self.calls = 0

    async def next(self, state):
        self.calls += 1
        if False:
            yield  # pragma: no cover

    async def reset(self, state):
        self.calls = 0

class MockLLM(BaseLLM):
    @classmethod
    def supported_models(cls):
        return ["mock-llm"]

    async def prepare(self):
        pass

    async def generate(self, *args, **kwargs):
        return "ok"

    async def generate_with_tools(self, *args, **kwargs):
        return {"content": "ok", "tool_calls": []}


class DummyMemory:
    async def add_message(self, message: Message) -> None:
        pass

    async def get_history(self, limit: int = 50, query: str | None = None) -> list[Message]:
        return []

    async def set_chunk_builder(self, chunk_builder):
        pass

    async def build_chunk(self, start_index: int | None = None, end_index: int | None = None) -> int:
        return 0

    async def search_chunk(self, session_id: str, query: str, limit: int = 5) -> list:
        return []

@pytest.mark.asyncio
async def test_base_agent_loops_five_iterations():
    planner = BasePlanner()
    llm = MockLLM("mock-llm")
    agent = BaseAgent(
        name="test",
        description="d",
        planner=planner,
        llm=llm,
        memory=DummyMemory(),
        default_config=RunConfig(turn_budget=10),
    )

    msg = Message(messageId="u1", role=Role.user, parts=[TextPart(text="hi")], contextId="ctx", kind="message")
    input_data = AgentRunInput(message=msg, options={"config": RunConfig(turn_budget=5)})

    outputs = [o async for o in agent.run(input_data)]

    assert outputs == []
    assert planner.calls == 5

    await agent.cleanup()
