"""Agent that uses :class:`PlanExecutePlanner` for plan-and-execute control."""

from __future__ import annotations

from .base import Planner
from .base_agent import BaseAgent
from .plan_execute_planner import PlanExecutePlanner


class PlanExecuteAgent(BaseAgent):
    """BaseAgent configured with :class:`PlanExecutePlanner`."""

    def __init__(self, *, planner: Planner | None = None, **kwargs) -> None:
        planner = planner or PlanExecutePlanner()
        super().__init__(planner=planner, **kwargs)
