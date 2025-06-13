from ..base import Planner
from ..types import State, Step, MessageStep, FinishStep, Message, Role, TextPart


class ReactivePlanner(Planner):
    """Simple reactive planner for immediate responses."""

    async def plan(self, state: State) -> list[Step]:
        if not state.pending:
            return [FinishStep(step_id="finish", reason="No pending messages")]

        latest = state.pending[-1]
        response_text = f"I received: {self._extract_content(latest)}"
        msg = Message(role=Role.agent, parts=[TextPart(text=response_text)])
        return [MessageStep(step_id="reply", message=msg, is_streaming=True)]

    def _extract_content(self, message: Message) -> str:
        return " ".join(getattr(p, "text", "") for p in message.parts)
