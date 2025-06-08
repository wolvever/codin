"""Agent implementation that performs simple search over planner suggestions.

This agent runs the planner multiple times per turn, evaluates the candidate
step sequences and executes the one with the highest score.  It inherits most
functionality from :class:`BaseAgent` and only overrides the planning loop.
"""

from __future__ import annotations

import time
import uuid
import typing as _t


from .base_agent import BaseAgent
from .types import (
    AgentRunOutput,
    FinishStep,
    Message,
    MessageStep,
    Role,
    State,
    Step,
    StepType,
    TextPart,
)

__all__ = ["SearchAgent"]


class SearchAgent(BaseAgent):
    """Agent that evaluates multiple planner outputs each turn."""

    def __init__(self, *args, search_width: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.search_width = max(1, search_width)

    async def _execute_planning_loop(
        self, state: State, session_id: str, start_time: float
    ) -> _t.AsyncGenerator[AgentRunOutput]:
        """Execute planning loop with naive search strategy."""

        await self._emit_event(
            "task_start",
            {"session_id": session_id, "iteration": state.iteration, "elapsed_time": 0.0},
        )

        while state.iteration < (state.config.turn_budget or 100):
            search_width = int(state.metadata.get("badgers", self.search_width))
            should_continue = await self.check_inbox_for_control()
            if not should_continue:
                break
            if self._paused:
                await self.wait_while_paused()
                if self._cancelled:
                    break

            elapsed_time = time.time() - start_time
            state.metrics.time_used = elapsed_time
            exceeded, reason = state.config.is_budget_exceeded(state.metrics, elapsed_time)
            if exceeded:
                finish_reason = f"Budget exceeded: {reason}"
                finish_msg = Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[TextPart(text=finish_reason)],
                    contextId=session_id,
                    kind="message",
                )
                finish_step = FinishStep(step_id=str(uuid.uuid4()), reason=finish_reason, final_message=finish_msg)
                await self.mailbox.put_outbox(finish_msg)
                async for output in self._execute_step(finish_step, state, session_id):
                    yield output
                break

            await self._emit_event(
                "turn_start",
                {"session_id": session_id, "iteration": state.iteration, "metrics": state.metrics.__dict__},
            )

            candidate_runs: list[list[Step]] = []
            for _ in range(search_width):
                steps: list[Step] = []
                async for s in self.planner.next(state):
                    steps.append(s)
                    if len(steps) >= 10 or s.step_type == StepType.FINISH:
                        break
                candidate_runs.append(steps)
                if hasattr(self.planner, "reset"):
                    await self.planner.reset(state)

            def score(steps: list[Step]) -> int:
                val = 0
                for st in steps:
                    if st.step_type == StepType.FINISH:
                        val += 100
                    elif st.step_type == StepType.MESSAGE:
                        val += 10
                    elif st.step_type == StepType.TOOL_CALL:
                        val += 5
                    elif st.step_type == StepType.THINK:
                        val += 1
                return val

            best_steps = max(candidate_runs, key=score) if candidate_runs else []

            steps_executed = 0
            task_finished = False
            for step in best_steps:
                should_continue = await self.check_inbox_for_control()
                if not should_continue:
                    break
                if self._paused:
                    await self.wait_while_paused()
                    if self._cancelled:
                        break

                steps_executed += 1
                if step.step_type == StepType.MESSAGE and isinstance(step, MessageStep) and step.message:
                    if not any(h.messageId == step.message.messageId for h in state.history if h.messageId and step.message.messageId):
                        await self.memory.add_message(step.message)
                        state.history.append(step.message)

                async for output in self._execute_step(step, state, session_id):
                    yield output

                if step.step_type == StepType.FINISH:
                    task_finished = True
                    await self._emit_event(
                        "task_complete",
                        {
                            "session_id": session_id,
                            "iteration": state.iteration,
                            "reason": step.reason if isinstance(step, FinishStep) and step.reason else "Finished",
                        },
                    )
                    break

                if steps_executed >= 10:
                    break

            if task_finished or self._cancelled:
                break

            state.iteration += 1
            await self._emit_event(
                "turn_end",
                {"session_id": session_id, "iteration": state.iteration, "steps_executed": steps_executed},
            )

        await self._emit_event(
            "task_end",
            {
                "session_id": session_id,
                "iteration": state.iteration,
                "elapsed_time": time.time() - start_time,
            },
        )

