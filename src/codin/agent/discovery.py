from typing import Dict, List

from .base import Agent
from .types import Message


class AgentRegistry:
    """Service for agent discovery and routing."""

    def __init__(self) -> None:
        self._agents: Dict[str, Agent] = {}
        self._capabilities: Dict[str, List[str]] = {}

    def register_agent(self, agent_id: str, agent: Agent, capabilities: List[str]) -> None:
        self._agents[agent_id] = agent
        self._capabilities[agent_id] = capabilities

    def _extract_content(self, message: Message) -> str:
        return " ".join(getattr(p, "text", "") for p in message.parts)

    def find_agent(self, message: Message, context: dict) -> Agent | None:
        for agent_id, caps in self._capabilities.items():
            if any(keyword in self._extract_content(message).lower() for keyword in caps):
                return self._agents[agent_id]
        return next(iter(self._agents.values()), None)
