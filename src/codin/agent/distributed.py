import ray

from .agent import BasicAgent
from .base import Agent, Planner
from .types import AgentRunInput


@ray.remote
class RayAgent:
    """Agent running as Ray actor on remote compute node."""

    def __init__(self, planner: Planner):
        self.planner = planner

    async def run(self, input_dict: dict) -> list[dict]:
        input_data = AgentRunInput.model_validate(input_dict)
        agent = BasicAgent(self.planner)
        results = []
        async for output in agent.run(input_data):
            if hasattr(output, "model_dump"):
                results.append(output.model_dump())
            else:
                results.append(output)
        return results


class AgentFactory:
    """Factory for creating different agent implementations."""

    @staticmethod
    def create_agent(deployment_type: str, planner: Planner) -> Agent:
        if deployment_type == "local":
            return BasicAgent(planner)
        if deployment_type == "ray":
            return RayAgent.remote(planner)
        raise ValueError(f"Unknown deployment: {deployment_type}")
