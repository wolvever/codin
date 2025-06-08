import pytest

from codin.agent.plan_execute_planner import PlanExecutePlanner
from codin.agent.types import FinishStep, MessageStep, State, ThinkStep


@pytest.mark.asyncio
async def test_plan_execute_planner_sequence(monkeypatch):
    planner = PlanExecutePlanner()

    async def fake_create_plan(state):
        return ["step1", "step2"]

    monkeypatch.setattr(planner, "_create_plan", fake_create_plan)

    state = State.model_construct(session_id="s1")

    steps = [step async for step in planner.next(state)]
    assert isinstance(steps[0], ThinkStep)
    assert isinstance(steps[1], MessageStep)

    steps = [step async for step in planner.next(state)]
    assert isinstance(steps[0], MessageStep)

    steps = [step async for step in planner.next(state)]
    assert isinstance(steps[0], FinishStep)
