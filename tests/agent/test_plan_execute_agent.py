import pytest

from codin.agent.plan_execute_agent import PlanExecuteAgent
from codin.agent.plan_execute_planner import PlanExecutePlanner
from codin.agent.types import AgentRunInput, Message, Role, RunConfig, State, TextPart
from codin.artifact.base import ArtifactService
from codin.tool.base import Tool

# ensure State is fully built for tests
State.model_rebuild(force=True, _types_namespace={'Tool': Tool, 'ArtifactService': ArtifactService, 'Message': Message})


class DummyPlanner(PlanExecutePlanner):
    async def _create_plan(self, state):
        return ["a", "b"]


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
async def test_plan_execute_agent_runs(monkeypatch):
    planner = DummyPlanner()
    agent = PlanExecuteAgent(
        name="p",
        description="d",
        planner=planner,
        memory=DummyMemory(),
        default_config=RunConfig(turn_budget=5),
    )

    msg = Message(messageId="u1", role=Role.user, parts=[TextPart(text="go")], contextId="ctx", kind="message")
    inp = AgentRunInput(message=msg)

    outputs = [o async for o in agent.run(inp)]
    assert outputs
    assert any("plan complete" in o.result.parts[0].text if hasattr(o.result, "parts") else False for o in outputs)
