"""Provides concrete implementations of the CallableActor protocol.

This module includes:
- `AgentActor`: An actor designed to handle A2A (Agent-to-Agent) messages and tasks.
- `PlainActor`: An actor that processes generic JSON-like tasks, validating
  their payloads against a provided Pydantic schema.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional, Type
from pydantic import BaseModel, ValidationError

from .types import CallableActor, ActorRunInput, ActorRunOutput
from .envelope_types import Capability, EnvelopeKind, TaskState
from ..a2a.types import A2ATaskPayload # Import for A2A_TASK payload deserialization
from ..agent.types import Message as A2AMessage

logger = logging.getLogger(__name__)


AGENT_CAP = Capability(
    accepts={EnvelopeKind.A2A_MESSAGE, EnvelopeKind.A2A_TASK, EnvelopeKind.CONTROL},
    # For A2A_MESSAGE, schema is A2AMessage. For A2A_TASK, it's A2ATaskPayload.
    # A Union or a more sophisticated schema registration could be used here.
    # For now, individual actors handle specific payload types for their accepted kinds.
    payload_schema=None
)


class AgentActor(CallableActor):
    """Processes A2A messages/tasks, with support for cancel, pause, resume."""
    capability: Capability = AGENT_CAP

    def __init__(self, actor_id: str, **kwargs: Any):
        self.actor_id = actor_id
        self._cancel_requested: bool = False
        self._paused: bool = False
        if kwargs:
            logger.warning(f"AgentActor {self.actor_id} received unexpected kwargs: {kwargs}")

    async def request_cancel(self) -> None:
        logger.info(f"AgentActor {self.actor_id} received cancel request.")
        self._cancel_requested = True

    async def request_pause(self) -> None:
        logger.info(f"AgentActor {self.actor_id} received pause request.")
        self._paused = True

    async def request_resume(self, followup_data: Optional[dict]) -> None:
        logger.info(f"AgentActor {self.actor_id} received resume request. Followup: {bool(followup_data)}. Resuming...")
        self._paused = False
        if followup_data:
            logger.info(f"AgentActor {self.actor_id} received followup_data on resume: {str(followup_data)[:200]}")


    async def run(self, input: ActorRunInput) -> AsyncIterator[ActorRunOutput]:
        logger.info(f"AgentActor {self.actor_id} run started for request {input.request_id}, kind: {input.envelope_kind}")

        if self._cancel_requested:
            logger.info(f"AgentActor {self.actor_id} run for request {input.request_id} cancelled before start.")
            yield {"status": TaskState.CANCELED.value, "detail": "Processing cancelled by request before start."}
            return

        while self._paused and not self._cancel_requested:
            logger.debug(f"AgentActor {self.actor_id} is paused (request {input.request_id}). Checking again in 1s.")
            await asyncio.sleep(1)

        if self._cancel_requested:
            logger.info(f"AgentActor {self.actor_id} run for request {input.request_id} cancelled after pause.")
            yield {"status": TaskState.CANCELED.value, "detail": "Processing cancelled by request after pause."}
            return

        if input.envelope_kind == EnvelopeKind.A2A_MESSAGE:
            if not isinstance(input.payload, dict):
                logger.error(f"AgentActor {self.actor_id}: A2A_MESSAGE payload not dict for request {input.request_id}.")
                yield {"error": "A2A_MESSAGE payload must be a dictionary", "request_id": input.request_id, "status": TaskState.FAILED.value}
                return
            try:
                if self._cancel_requested: # Checkpoint
                    logger.info(f"AgentActor {self.actor_id} cancelled before A2A_MESSAGE processing for {input.request_id}.")
                    yield {"status": TaskState.CANCELED.value, "detail": "Cancelled before A2A_MESSAGE processing."}; return

                a2a_message = A2AMessage.model_validate(input.payload)
                logger.info(f"AgentActor {self.actor_id}: Parsed A2A_MESSAGE for request {input.request_id}, role: {a2a_message.role.value}")
                # TODO: Implement actual A2A message processing logic (e.g., invoke BaseAgent's loop).
                # This logic should also periodically check self._cancel_requested and self._paused.
                yield {
                    "status": TaskState.RUNNING.value, # Indicate start of processing this message
                    "detail": f"AgentActor {self.actor_id} started processing a2a.message",
                    "request_id": input.request_id,
                }
                # Simulate work for A2A message
                await asyncio.sleep(0.5)
                if self._cancel_requested:
                    logger.info(f"AgentActor {self.actor_id} cancelled during A2A_MESSAGE processing for {input.request_id}.")
                    yield {"status": TaskState.CANCELED.value, "detail": "Cancelled during A2A_MESSAGE processing."}; return

                yield {
                    "status": TaskState.COMPLETED.value, # Indicate completion of this message
                    "detail": f"AgentActor {self.actor_id} successfully processed a2a.message",
                    "request_id": input.request_id,
                    "parsed_message_role": a2a_message.role.value,
                    "num_parts": len(a2a_message.parts),
                }
            except ValidationError as e:
                logger.error(f"AgentActor {self.actor_id}: A2A_MESSAGE validation failed for request {input.request_id}: {e}", exc_info=True)
                yield {"error": "A2A_MESSAGE payload validation failed", "details": e.errors(), "request_id": input.request_id, "status": TaskState.FAILED.value}

        elif input.envelope_kind == EnvelopeKind.A2A_TASK:
            logger.info(f"AgentActor {self.actor_id} received A2A_TASK. Raw payload: {str(input.payload)[:200]} for request {input.request_id}")
            if self._cancel_requested:
                logger.info(f"AgentActor {self.actor_id} cancelled before A2A_TASK processing for {input.request_id}.")
                yield {"status": TaskState.CANCELED.value, "detail": "Processing cancelled by request."}; return

            if not isinstance(input.payload, dict):
                logger.error(f"AgentActor {self.actor_id}: A2A_TASK payload is not a dict for request {input.request_id}.")
                yield {"error": "A2A_TASK payload must be a dictionary", "request_id": input.request_id, "status": TaskState.FAILED.value}
                return
            try:
                a2a_task = A2ATaskPayload.model_validate(input.payload)
                logger.info(f"AgentActor {self.actor_id}: Successfully parsed A2A_TASK '{a2a_task.task_name or 'unnamed'}' for request {input.request_id}")

                yield {
                    "status": TaskState.RUNNING.value,
                    "detail": f"Started processing A2A_TASK: {a2a_task.task_name or 'unnamed'}",
                    "request_id": input.request_id,
                    "task_params_preview": str(a2a_task.params)[:100] if a2a_task.params else None
                }

                # TODO: Implement actual A2A task execution logic based on a2a_task.task_name or params.
                # This logic should also periodically check self._cancel_requested and self._paused.
                await asyncio.sleep(1) # Simulate work

                if self._cancel_requested:
                    logger.info(f"AgentActor {self.actor_id} cancelled during A2A_TASK processing for {input.request_id}.")
                    yield {"status": TaskState.CANCELED.value, "detail": f"A2A_TASK '{a2a_task.task_name or 'unnamed'}' cancelled."}; return

                # Simulate pause check within longer task
                while self._paused and not self._cancel_requested:
                    logger.debug(f"AgentActor {self.actor_id} A2A_TASK processing paused for request {input.request_id}. Checking in 1s.")
                    await asyncio.sleep(1)
                if self._cancel_requested:
                    logger.info(f"AgentActor {self.actor_id} A2A_TASK for request {input.request_id} cancelled after pause.");
                    yield {"status": TaskState.CANCELED.value, "detail": "Cancelled after pause during A2A_TASK."}; return

                processed_result = {"example_output": f"task {a2a_task.task_name or 'unnamed'} completed successfully"}
                yield {
                    "status": TaskState.COMPLETED.value,
                    "detail": f"Completed A2A_TASK: {a2a_task.task_name or 'unnamed'}",
                    "request_id": input.request_id,
                    "result": processed_result
                }
            except ValidationError as e:
                logger.error(f"AgentActor {self.actor_id}: A2A_TASK payload validation failed for request {input.request_id}: {e}", exc_info=True)
                yield {"error": "A2A_TASK payload validation failed.", "details": e.errors(), "request_id": input.request_id, "status": TaskState.FAILED.value}


        elif input.envelope_kind == EnvelopeKind.CONTROL:
            control_data = input.payload
            logger.info(f"AgentActor {self.actor_id}: Received CONTROL via run method for request {input.request_id}, payload: {control_data}")
            yield { "status": f"AgentActor {self.actor_id} acknowledged control signal in run (should be rare)", "request_id": input.request_id, "control_info": control_data }
        else:
            logger.warning(f"AgentActor {self.actor_id}: Unhandled kind '{input.envelope_kind}' for request {input.request_id}")
            yield { "error": f"AgentActor {self.actor_id} cannot handle kind '{input.envelope_kind}'", "request_id": input.request_id, "status": TaskState.FAILED.value }

        logger.info(f"AgentActor {self.actor_id} finished run method for request {input.request_id}")

    async def cleanup(self) -> None:
        logger.info(f"AgentActor {self.actor_id} cleaning up.")
        pass


class PlainActor(CallableActor):
    """Processes generic tasks, with support for cancel, pause, resume."""

    def __init__(self, actor_id: str, payload_schema: Type[BaseModel], **kwargs: Any):
        self.actor_id = actor_id
        self.payload_schema = payload_schema
        self.capability = Capability(
            accepts={EnvelopeKind.PLAIN_TASK, EnvelopeKind.CONTROL},
            payload_schema=payload_schema
        )
        self._cancel_requested: bool = False
        self._paused: bool = False
        if kwargs:
            logger.warning(f"PlainActor {self.actor_id} received unexpected kwargs: {kwargs}")

    async def request_cancel(self) -> None:
        logger.info(f"PlainActor {self.actor_id} received cancel request.")
        self._cancel_requested = True

    async def request_pause(self) -> None:
        logger.info(f"PlainActor {self.actor_id} received pause request.")
        self._paused = True

    async def request_resume(self, followup_data: Optional[dict]) -> None:
        logger.info(f"PlainActor {self.actor_id} received resume request. Followup: {bool(followup_data)}. Resuming...")
        self._paused = False
        if followup_data:
            logger.info(f"PlainActor {self.actor_id} received followup_data on resume: {str(followup_data)[:200]}")

    async def run(self, input: ActorRunInput) -> AsyncIterator[ActorRunOutput]:
        logger.info(f"PlainActor {self.actor_id} received input for request {input.request_id}, kind: {input.envelope_kind}")

        if self._cancel_requested:
            logger.info(f"PlainActor {self.actor_id} run for request {input.request_id} cancelled before start.")
            yield {"status": TaskState.CANCELED.value, "detail": "Processing cancelled by request before start."}
            return

        while self._paused and not self._cancel_requested:
            logger.debug(f"PlainActor {self.actor_id} is paused (request {input.request_id}). Checking again in 1s.")
            await asyncio.sleep(1)

        if self._cancel_requested:
            logger.info(f"PlainActor {self.actor_id} run for request {input.request_id} cancelled after pause.")
            yield {"status": TaskState.CANCELED.value, "detail": "Processing cancelled by request after pause."}
            return

        if input.envelope_kind == EnvelopeKind.PLAIN_TASK:
            if not isinstance(input.payload, dict) or "args" not in input.payload:
                logger.error(f"PlainActor {self.actor_id}: PLAIN_TASK payload for request {input.request_id} is not dict with 'args'. Payload: {str(input.payload)[:200]}")
                yield { "error": f"PlainActor {self.actor_id} expects PLAIN_TASK payload to be a dict with 'args' key.", "request_id": input.request_id, "status": TaskState.FAILED.value }
                return

            try:
                if self._cancel_requested:
                    logger.info(f"PlainActor {self.actor_id} run for request {input.request_id} cancelled before validation.")
                    yield {"status": TaskState.CANCELED.value, "detail": "Processing cancelled by request."}; return

                parsed_args = self.payload_schema.model_validate(input.payload["args"])
                logger.info(f"PlainActor {self.actor_id}: Successfully parsed PLAIN_TASK args for request {input.request_id}.")

                yield {
                    "status": TaskState.RUNNING.value,
                    "detail": f"PlainActor {self.actor_id} started processing plain.task",
                    "request_id": input.request_id,
                }

                for i in range(3):
                    if self._cancel_requested:
                        logger.info(f"PlainActor {self.actor_id} run for request {input.request_id} cancelled during processing step {i+1}.")
                        yield {"status": TaskState.CANCELED.value, "detail": f"Processing cancelled by request during step {i+1}."}; return

                    while self._paused and not self._cancel_requested: # Inner pause check for longer steps
                        logger.debug(f"PlainActor {self.actor_id} processing step {i+1} paused for request {input.request_id}. Checking in 1s.")
                        await asyncio.sleep(1)
                    if self._cancel_requested:
                        logger.info(f"PlainActor {self.actor_id} run for request {input.request_id} cancelled after pause in step {i+1}.");
                        yield {"status": TaskState.CANCELED.value, "detail": f"Cancelled after pause in step {i+1}."}; return

                    logger.debug(f"PlainActor {self.actor_id} processing step {i+1} for request {input.request_id}")
                    await asyncio.sleep(0.01)

                yield {
                    "status": TaskState.COMPLETED.value,
                    "detail": f"PlainActor {self.actor_id} processed plain.task successfully",
                    "request_id": input.request_id,
                    "parsed_args": parsed_args.model_dump()
                }
            except ValidationError as e:
                logger.error(f"PlainActor {self.actor_id}: PLAIN_TASK validation failed for request {input.request_id}: {e}", exc_info=True)
                yield { "error": f"PlainActor {self.actor_id} PLAIN_TASK payload validation failed", "details": e.errors(), "request_id": input.request_id, "status": TaskState.FAILED.value }
            except Exception as e:
                logger.error(f"PlainActor {self.actor_id}: Unexpected error processing PLAIN_TASK for request {input.request_id}: {e}", exc_info=True)
                yield { "error": f"PlainActor {self.actor_id} encountered an unexpected processing error", "details": str(e), "request_id": input.request_id, "status": TaskState.FAILED.value }

        elif input.envelope_kind == EnvelopeKind.CONTROL:
            control_data = input.payload
            logger.info(f"PlainActor {self.actor_id}: Received CONTROL via run method for request {input.request_id}, payload: {control_data}")
            yield { "status": f"PlainActor {self.actor_id} acknowledged control signal in run (should be rare)", "request_id": input.request_id, "control_info": control_data }

        else:
            logger.warning(f"PlainActor {self.actor_id}: Unhandled kind '{input.envelope_kind}' for request {input.request_id}")
            yield { "error": f"PlainActor {self.actor_id} cannot handle kind '{input.envelope_kind}'", "request_id": input.request_id, "status": TaskState.FAILED.value }

        logger.info(f"PlainActor {self.actor_id} finished run method for request {input.request_id}")

    async def cleanup(self) -> None:
        logger.info(f"PlainActor {self.actor_id} cleaning up.")
        pass


__all__ = [
    "AgentActor",
    "PlainActor",
    "AGENT_CAP",
]
