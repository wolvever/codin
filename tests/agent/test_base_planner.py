import pytest

import codin.agent.base_planner as base_planner
from codin.agent.base_planner import BasePlanner
from codin.agent.types import ErrorStep, FinishStep, State

ErrorStep.model_rebuild()


@pytest.mark.asyncio
async def test_error_step_emitted(monkeypatch):
    async def fail_prompt_run(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(base_planner, "prompt_run", fail_prompt_run)

    planner = BasePlanner()
    state = State.model_construct(session_id="s2")

    steps = [step async for step in planner.next(state)]
    assert isinstance(steps[0], ErrorStep)
    assert isinstance(steps[1], FinishStep)
