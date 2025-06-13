import json
import uuid
from typing import Any, Dict, List

from .base import Planner
from .types import (
    State,
    Step,
    StepType,
    MessageStep,
    ToolCallStep,
    FinishStep,
    Message,
    Role,
    TextPart,
    ToolUsePart,
)
from ..model.base import BaseLLM


class BasicPlanner(Planner):
    """JSON action-based LLM planner generating structured responses."""

    def __init__(self, llm: BaseLLM, system_prompt: str | None = None):
        self.llm = llm
        self.system_prompt = system_prompt or self._default_system_prompt()

    async def plan(self, state: State) -> List[Step]:
        """Generate next steps using LLM with JSON action output."""
        context = await self._build_context(state)
        messages = await self._build_llm_messages(context, state)
        response = await self.llm.generate(messages=messages, temperature=0.3)
        return await self._parse_json_response_to_steps(response, state)

    async def _build_context(self, state: State) -> Dict[str, Any]:
        return {
            "available_tools": [t.name for t in state.tools],
            "recent_history": [self._extract_content(m) for m in state.history[-5:]],
            "constraints": "",
        }

    async def _build_llm_messages(self, context: Dict[str, Any], state: State) -> List[Dict[str, str]]:
        system = self.system_prompt.format(**context)
        messages = [{"role": "system", "content": system}]
        if state.pending:
            user_msg = self._extract_content(state.pending[-1])
            messages.append({"role": "user", "content": user_msg})
        return messages

    async def _parse_json_response_to_steps(self, response: str, state: State) -> List[Step]:
        try:
            data = json.loads(response)
        except Exception:
            msg = Message(role=Role.agent, parts=[TextPart(text=response)])
            return [MessageStep(step_id=str(uuid.uuid4()), message=msg)]

        if isinstance(data, str) and data.strip().lower() == "done":
            return [FinishStep(step_id=str(uuid.uuid4()), reason="LLM signaled done")]

        steps: List[Step] = []
        if isinstance(data, list):
            for item in data:
                if item.get("type") == "message":
                    msg = Message(role=Role.agent, parts=[TextPart(text=item.get("content", ""))])
                    steps.append(MessageStep(step_id=str(uuid.uuid4()), message=msg))
                elif item.get("type") == "tool_call":
                    call = ToolUsePart(
                        kind="tool-use",
                        type="call",
                        id=str(uuid.uuid4()),
                        name=item.get("tool", ""),
                        input=item.get("parameters", {}),
                    )
                    steps.append(ToolCallStep(step_id=str(uuid.uuid4()), tool_call=call))
        return steps

    def _extract_content(self, message: Message) -> str:
        texts = [getattr(p, "text", "") for p in message.parts]
        return "\n".join(texts)

    def _default_system_prompt(self) -> str:
        return (
            "You are a helpful AI assistant. Respond with either \"done\" or JSON actions.\n\n"
            "CONTEXT:\n"
            "Available Tools: {available_tools}\n"
            "Recent History: {recent_history}\n"
            "Constraints: {constraints}\n\n"
            "RESPONSE FORMAT:\n"
            "1. \"done\" - if task is complete\n"
            "2. JSON list: [{\"type\": \"message\", \"content\": \"text\"}, {\"type\": \"tool_call\", \"tool\": \"name\", \"parameters\": {...}}]"
        )
