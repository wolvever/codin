"""Simple CodeAct-style planner without external dependencies."""

from __future__ import annotations

import asyncio
import re
import typing as _t
import uuid

from ..id import new_id
from ..sandbox.local import LocalSandbox
# from ..utils.message import extract_text_from_message # Removed
from .base import Planner

# Uses Role enum from .types
from .types import FinishStep, Message, MessageStep, Role, State, Step, TextPart


class CodeActPlanner(Planner):
    """Simplistic planner that generates code and executes it."""

    _CODE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

    def __init__(self, model: _t.Any, sandbox: LocalSandbox | None = None, prompt: str | None = None) -> None:
        self.model = model
        self.sandbox = sandbox or LocalSandbox()
        self.prompt = prompt

    async def _invoke_model(self, messages: list[_t.Any]) -> _t.Any:
        """Invoke the underlying model handling sync/async methods."""
        if hasattr(self.model, "ainvoke"):
            return await self.model.ainvoke(messages)
        result = self.model.invoke(messages)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _eval_code(self, code: str) -> str:
        await self.sandbox.up()
        try:
            result = await self.sandbox.run_code(code, language="python")
        finally:
            await self.sandbox.down()
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        return output.strip()

    async def next(self, state: State) -> _t.AsyncGenerator[Step]:
        lc_messages: list[dict[str, str]] = []
        for hist_msg in state.history: # Renamed msg to hist_msg to avoid conflict
            text = hist_msg.get_text_content() # Updated
            if hist_msg.role == Role.user:
                lc_messages.append({"role": "user", "content": text})
            else:
                lc_messages.append({"role": "assistant", "content": text})

        while True:
            result = await self._invoke_model(lc_messages)
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
            else:
                content = str(result)

            msg = Message.from_text( # Updated
                text=str(content),
                role=Role.agent,
                contextId=state.session_id,
                messageId=new_id("msg")
            )

            code_blocks = self._CODE_RE.findall(str(content))
            if code_blocks:
                # Yield the code block as a message
                yield MessageStep(step_id=str(uuid.uuid4()), message=msg)
                # Use the already created msg's text content for lc_messages
                lc_messages.append({"role": "assistant", "content": msg.get_text_content()})


                for block in code_blocks:
                    output = await self._eval_code(block)
                    out_msg = Message.from_text( # Updated
                        text=output,
                        role=Role.agent,
                        contextId=state.session_id,
                        messageId=new_id("msg")
                    )
                    yield MessageStep(step_id=str(uuid.uuid4()), message=out_msg)
                    lc_messages.append({"role": "assistant", "content": output})
                # Ask the model again with the new messages
                continue

            # No code produced - consider this the final message
            yield FinishStep(step_id=str(uuid.uuid4()), final_message=msg, reason="CodeAct finished")
            break

    async def reset(self, state: State) -> None:
        """Reset the planner state."""
        # Stateless implementation, nothing to reset
        return
