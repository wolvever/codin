from __future__ import annotations

import asyncio
import uuid

from ..actor.mailbox import LocalMailbox, Mailbox
from ..session.base import SessionManager
from .base_agent import BaseAgent
from .types import AgentRunInput, ControlSignal, RunnerControl

__all__ = ["AgentRunner"]


class AgentRunner:
    """Simple runner that bridges a mailbox with an agent.

    It listens for incoming messages on the mailbox inbox, creates or
    retrieves a session using :class:`SessionManager`, and dispatches the
    message to the wrapped agent. Agent outputs are already placed on the
    mailbox outbox by :class:`BaseAgent` so the runner simply drains the
    agent's output generator.
    """

    def __init__(
        self,
        agent: BaseAgent,
        /,
        *,
        mailbox: Mailbox | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.agent = agent
        self.mailbox = mailbox or agent.mailbox or LocalMailbox()
        self.session_manager = session_manager or SessionManager()
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start processing mailbox messages."""
        if self._task is None:
            self._running = True
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the runner."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
            self._task = None

    async def _run_loop(self) -> None:
        async for msg in self.mailbox.subscribe_inbox():
            if not self._running:
                break

            if msg.metadata and msg.metadata.get("control"):
                signal = ControlSignal(msg.metadata["control"])
                control = RunnerControl(signal=signal, metadata=msg.metadata)
                await self.agent.handle_control(control)
                continue

            session_id = msg.contextId or f"session_{uuid.uuid4().hex[:8]}"
            await self.session_manager.get_or_create_session(session_id)

            agent_input = AgentRunInput(message=msg, session_id=session_id)
            async for _ in self.agent.run(agent_input):
                pass
            await asyncio.sleep(0)
