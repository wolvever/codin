"""Simple planner implementing a plan-and-execute approach.

This planner follows the two phase strategy popularized by open source
projects such as Teknium's `plan-and-execute` (2023) where an LLM first
creates a high level plan and subsequent calls execute each step.

The planner keeps the generated plan in memory and yields ``MessageStep``
for each step in sequence until completion. A ``FinishStep`` is emitted
when all steps have been executed.
"""
from __future__ import annotations

import logging
import typing as _t
import uuid
from datetime import datetime

from ..prompt.run import prompt_run
from .types import (
    FinishStep,
    Message,
    MessageStep,
    Planner,
    Role,
    State,
    Step,
    TextPart,
    ThinkStep,
)

logger = logging.getLogger(__name__)


class PlanExecutePlanner(Planner):
    """Planner that generates a plan once then yields steps sequentially."""

    def __init__(self, *, plan_prompt: str = "plan_execute_plan") -> None:
        self.plan_prompt = plan_prompt
        self._plan: list[str] | None = None
        self._index = 0

    async def _create_plan(self, state: State) -> list[str]:
        """Create a plan using the configured prompt template."""
        user_input = ""
        if state.history:
            last = state.history[-1]
            if last.role == Role.user:
                user_input = last.parts[0].text if last.parts else ""
        resp = await prompt_run(self.plan_prompt, variables={"user_input": user_input})
        text = ""
        if hasattr(resp, "message") and resp.message:
            for p in resp.message.parts:
                if hasattr(p, "text"):
                    text += p.text
        elif hasattr(resp, "content"):
            text = str(resp.content)
        # Split lines as plan steps
        plan = [line.strip() for line in text.splitlines() if line.strip()]
        return plan

    async def next(self, state: State) -> _t.AsyncGenerator[Step]:
        if self._plan is None:
            try:
                self._plan = await self._create_plan(state)
                self._index = 0
                plan_overview = "; ".join(self._plan)
                yield ThinkStep(
                    step_id=str(uuid.uuid4()),
                    thinking=f"Plan created: {plan_overview}",
                    created_at=datetime.now(),
                )
            except Exception as e:  # pragma: no cover - network or parsing error
                yield FinishStep(
                    step_id=str(uuid.uuid4()),
                    reason=f"planning failed: {e!s}",
                    created_at=datetime.now(),
                )
                return

        if self._index < len(self._plan):
            step_text = self._plan[self._index]
            self._index += 1
            msg = Message(
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                parts=[TextPart(text=step_text)],
                contextId=state.session_id,
                kind="message",
            )
            yield MessageStep(
                step_id=str(uuid.uuid4()),
                message=msg,
                created_at=datetime.now(),
            )
            return

        final_msg = Message(
            messageId=str(uuid.uuid4()),
            role=Role.agent,
            parts=[TextPart(text="plan complete")],
            contextId=state.session_id,
            kind="message",
        )

        yield FinishStep(
            step_id=str(uuid.uuid4()),
            reason="plan complete",
            final_message=final_msg,
            created_at=datetime.now(),
        )

    async def reset(self, state: State) -> None:
        self._plan = None
        self._index = 0
