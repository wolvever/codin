import asyncio
import ray

from a2a.types import Role, TextPart

from codin.actor.ray_scheduler import RayAgentActor
from codin.agent.base_agent import BaseAgent
from codin.agent.base_planner import BasePlanner
from codin.agent.types import AgentRunInput, Message, RunConfig
from codin.memory.base import MemMemoryService
from codin.prompt.engine import PromptEngine
import codin.prompt.run as prompt_run_module
from codin.model.base import BaseLLM
from codin.actor.supervisor import ActorInfo
from codin.agent.types import State, Message
from codin.tool.base import Tool
from codin.artifact.base import ArtifactService


class MockLLM(BaseLLM):
    """Simple mock LLM returning a preset response."""

    def __init__(self, model: str = "mock-llm"):
        super().__init__(model)
        self.response = ""
        self._prepared = False

    @classmethod
    def supported_models(cls) -> list[str]:
        return ["mock-.*"]

    async def prepare(self) -> None:
        self._prepared = True

    async def generate(self, prompt, *, stream: bool = False, **kwargs):
        if stream:
            async def _s():
                yield self.response
            return _s()
        return self.response

    async def generate_with_tools(self, prompt, tools, *, stream: bool = False, **kwargs):
        if stream:
            async def _s():
                yield {"content": self.response, "tool_calls": []}
            return _s()
        return {"content": self.response, "tool_calls": []}


async def create_agent(agent_type: str, key: str) -> BaseAgent:
    planner = BasePlanner()
    memory = MemMemoryService()
    return BaseAgent(agent_id=f"{agent_type}:{key}", name=f"{agent_type}-{key}", planner=planner, memory=memory)


async def main() -> None:
    mock_llm = MockLLM()
    mock_llm.response = '{"message": "done", "tool_calls": [], "should_continue": false}'
    prompt_run_module._engine = PromptEngine(mock_llm, endpoint="fs://./prompt_templates")

    import typing as _t
    ActorInfo.__annotations__['agent'] = _t.Any
    ActorInfo.model_rebuild(force=True)
    State.model_rebuild(force=True, _types_namespace={'Tool': Tool, 'ArtifactService': ArtifactService, 'Message': Message})

    ray.init(ignore_reinit_error=True)

    tasks = []
    for i in range(3):
        agent = await create_agent("agent", str(i))
        handle = RayAgentActor.remote(agent)
        msg = Message(
            messageId=f"m{i}",
            role=Role.user,
            parts=[TextPart(text=f"task {i}")],
            contextId=f"session-{i}",
            kind="message",
        )
        agent_input = AgentRunInput(session_id=f"session-{i}", message=msg, options={"config": RunConfig(turn_budget=1)})
        tasks.append(handle.run.remote(agent_input.dict()))

    results = await asyncio.gather(*(asyncio.to_thread(ray.get, t) for t in tasks))
    for idx, outputs in enumerate(results):
        print(f"Agent {idx} outputs:")
        for out in outputs:
            print(out)

    ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
