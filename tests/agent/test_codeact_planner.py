import pytest

from codin.agent.base_agent import BaseAgent
from codin.agent.codeact_planner import CodeActPlanner
from codin.memory.local import MemMemoryService
from codin.sandbox.local import LocalSandbox
from codin.agent.types import AgentRunInput, Message, TextPart, Role, State, RunConfig
from codin.tool.base import Tool
from codin.artifact.base import ArtifactService

# rebuild pydantic models for tests
State.model_rebuild(force=True, _types_namespace={
    'Tool': Tool,
    'ArtifactService': ArtifactService,
    'Message': Message,
})
class DummyModel:
    def __init__(self):
        self.calls = 0
    def invoke(self, messages):
        if self.calls == 0:
            self.calls += 1
            return {"role": "assistant", "content": "```python\nprint('Hello world')\n```"}
        return {"role": "assistant", "content": "done"}

@pytest.mark.asyncio
async def test_codeact_planner_hello_world(tmp_path):
    sandbox = LocalSandbox(workdir=str(tmp_path))
    model = DummyModel()
    planner = CodeActPlanner(model, sandbox=sandbox)
    agent = BaseAgent(name="ca", description="d", planner=planner, memory=MemMemoryService(), default_config=RunConfig())

    user_msg = Message(messageId="u1", role=Role.user, parts=[TextPart(text="do it")], contextId="s1", kind="message")
    inputs = AgentRunInput(session_id="s1", message=user_msg)

    outputs = [o async for o in agent.run(inputs)]
    assert outputs
    final = outputs[-1].result
    assert any("done" in p.text for p in final.parts if hasattr(p, "text"))
    history = await agent.memory.get_history()
    assert any("Hello world" in p.text for m in history for p in m.parts if hasattr(p, "text"))
