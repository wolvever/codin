"""Dispatcher for handling client requests and orchestrating actor execution.

This module provides services for receiving enveloped requests,
routing them to appropriate actors (instances conforming to `CallableActor`),
managing actor lifecycles via an `ActorSupervisor`, and coordinating
multi-actor workflows within the CoDIN framework. It also integrates with
a `TaskRegistry` to track the state of tasks.
"""

from __future__ import annotations

import asyncio
import logging
import typing as _t
import uuid
import warnings # Added for deprecation
from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

from .mailbox import Mailbox
from .supervisor import ActorSupervisor, ActorInfo
from .types import CallableActor, ActorRunInput, ActorRunOutput
from .envelope_types import Envelope, EnvelopeKind, ControlPayload, Capability, ControlAction, TaskState
from .task_manager import TaskRegistry, TaskInfo

logger = logging.getLogger(__name__)

__all__ = [
    'DispatchResult',
    'Dispatcher',
    'LocalDispatcher',
]


class DispatchResult(BaseModel):
    """Represents the outcome of a dispatched work request."""
    runner_id: str
    request_id: str
    task_id: str
    status: str
    agents: list[str] = Field(default_factory=list)
    metadata: dict[str, _t.Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class Dispatcher(ABC):
    """Abstract base class for a dispatcher."""
    @abstractmethod
    async def submit(self, envelope_dict: dict[_t.Any, _t.Any]) -> str:
        """Submits a request envelope for processing."""
        ...
    @abstractmethod
    async def signal(self, actor_id: str, ctrl: str) -> None:
        """Sends a control signal to a specific actor."""
        ...
    @abstractmethod
    async def get_status(self, runner_id: str) -> DispatchResult | None:
        """Retrieves the status of a specific execution run."""
        ...
    @abstractmethod
    async def list_active_runs(self) -> list[DispatchResult]:
        """Lists all currently active runs."""
        ...


class LocalDispatcher(Dispatcher):
    """Local, in-process Dispatcher that integrates with a TaskRegistry."""

    def __init__(self, actor_manager: ActorSupervisor, max_concurrency: int | None = None):
        self.actor_manager: ActorSupervisor = actor_manager
        self._task_registry: TaskRegistry = TaskRegistry()
        self._active_runs: dict[str, DispatchResult] = {}
        self._run_tasks: dict[str, asyncio.Task] = {}
        self._run_streams: dict[str, asyncio.Queue[ActorRunOutput | None]] = {}
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrency) if max_concurrency else None
        )

    async def submit(self, envelope_dict: dict[_t.Any, _t.Any]) -> str:
        logger.info(f"LocalDispatcher received submission: {str(envelope_dict)[:256]}...")
        try:
            envelope = Envelope.model_validate(envelope_dict)
        except ValidationError as e:
            logger.error(f"Invalid envelope submitted: {e}", exc_info=True)
            raise ValueError(f"Invalid envelope format: {e}") from e

        request_id = envelope.request_id
        current_task_id = envelope.task_id or f"task_{request_id}_{uuid.uuid4().hex[:8]}"
        runner_id_base = current_task_id
        runner_id = f'run_{runner_id_base}_{uuid.uuid4().hex[:4]}'

        logger.info(f"Processing Envelope request_id: {request_id}, task_id: {current_task_id}, kind: {envelope.kind}, runner_id: {runner_id}")

        task_info = TaskInfo(
            task_id=current_task_id, runner_id=runner_id, actor_id="",
            current_state=TaskState.SUBMITTED, envelope_kind=envelope.kind,
            reply_to=envelope.headers.reply_to
        )
        await self._task_registry.add_task(task_info)

        result = DispatchResult(
            runner_id=runner_id, request_id=request_id, task_id=current_task_id,
            status=TaskState.SUBMITTED.value
        )
        self._active_runs[runner_id] = result
        stream_queue: asyncio.Queue[ActorRunOutput | None] = asyncio.Queue()
        self._run_streams[runner_id] = stream_queue

        async_task = asyncio.create_task(self._handle_request(envelope, result, stream_queue))
        self._run_tasks[runner_id] = async_task
        return runner_id

    async def signal(self, actor_id: str, ctrl: str) -> None:
        """Sends a control signal to a specific actor via its mailbox.

        .. deprecated:: 0.1.0
           Use envelope-based control by submitting an `Envelope` with
           `kind=EnvelopeKind.CONTROL` and a `ControlPayload` instead.
           This method will be removed in a future version.

        Args:
            actor_id: The ID of the target actor.
            ctrl: The control string/message (specific to the old mechanism).

        Raises:
            ValueError: If the actor is not found.
        """
        warnings.warn(
            "LocalDispatcher.signal() is deprecated and will be removed. "
            "Use envelope-based control (EnvelopeKind.CONTROL) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.warning(
            f"Deprecated LocalDispatcher.signal() called for actor_id '{actor_id}' with ctrl '{ctrl}'. "
            f"Migrate to envelope-based control."
        )
        actor_info = await self.actor_manager.info(actor_id)
        if not actor_info:
            raise ValueError(f'Actor {actor_id} not found for signal.')
        actor_instance = actor_info.agent
        if hasattr(actor_instance, 'mailbox') and isinstance(actor_instance.mailbox, Mailbox):
            from ..agent.types import Message, Role, TextPart # Local import for specific, deprecated message type
            control_message = Message(
                messageId=str(uuid.uuid4()), role=Role.system,
                parts=[TextPart(text=f'Control signal: {ctrl}')], contextId=actor_id,
                kind='message', metadata={'control': ctrl, 'signal_type': 'dispatcher_signal_deprecated'},
            )
            await actor_instance.mailbox.put_inbox(control_message)
        else:
            logger.warning(f"Actor {actor_id} does not have a mailbox for deprecated signal '{ctrl}'.")

    async def get_status(self, runner_id: str) -> DispatchResult | None:
        dispatch_result = self._active_runs.get(runner_id)
        if dispatch_result:
            task_info = await self._task_registry.get_task(dispatch_result.task_id)
            if task_info:
                 dispatch_result.status = task_info.current_state.value
                 if task_info.error_info:
                     dispatch_result.metadata["task_error"] = task_info.error_info
        return dispatch_result

    async def list_active_runs(self) -> list[DispatchResult]:
        return list(self._active_runs.values())

    async def _handle_request(
        self, envelope: Envelope, result: DispatchResult, stream_queue: asyncio.Queue[ActorRunOutput | None],
    ) -> None:
        task_id_for_registry = result.task_id
        logger.info(f"Handling task: {task_id_for_registry}, envelope_id: {envelope.request_id}, kind: {envelope.kind}, actor_hint: {envelope.actor_hint}")
        created_actor_ids: list[str] = []
        actors: list[CallableActor] = []

        actor_type_hint = envelope.actor_hint or "default_actor"
        actor_key = envelope.session_id or task_id_for_registry
        actor_id_for_check = f"{actor_type_hint}:{actor_key}"

        try:
            await self._task_registry.update_task_state(task_id_for_registry, TaskState.PENDING)
            result.status = TaskState.PENDING.value

            existing_actor_info = await self.actor_manager.info(actor_id_for_check)
            if existing_actor_info and existing_actor_info.capability:
                if envelope.kind not in existing_actor_info.capability.accepts:
                    error_message = f"Existing actor {actor_id_for_check} does not accept kind '{envelope.kind}'."
                    logger.error(f"{error_message} Task: {task_id_for_registry}. Accepted: {existing_actor_info.capability.accepts}")
                    result.status = TaskState.FAILED.value; result.metadata['error'] = error_message
                    await self._task_registry.update_task_state(task_id_for_registry, TaskState.FAILED, error_info=error_message)
                    await stream_queue.put({"error": error_message, "status": result.status, "accepted_kinds": list(existing_actor_info.capability.accepts)});
                    return
                logger.debug(f"Capability check passed for existing actor {actor_id_for_check}, task: {task_id_for_registry}")

            if envelope.kind == EnvelopeKind.CONTROL:
                logger.info(f"Processing CONTROL envelope for task: {task_id_for_registry}")
                target_control_task_id = envelope.task_id
                if not target_control_task_id:
                    logger.error(f"CONTROL envelope {envelope.request_id} must specify a target 'task_id'. This control run ({task_id_for_registry}) will fail.")
                    result.status = TaskState.FAILED.value; result.metadata['error'] = "Control envelope lacks target task_id."
                    await self._task_registry.update_task_state(task_id_for_registry, TaskState.FAILED, error_info=result.metadata['error'])
                    await stream_queue.put({"error": result.metadata['error'], "status": result.status})
                    return
                try:
                    control_payload = ControlPayload.model_validate(envelope.payload)
                    action = control_payload.task_control
                    followup_data = control_payload.followup
                    logger.info(f"Control action '{action.value}' for target_task_id '{target_control_task_id}'. Followup: {bool(followup_data)}. Envelope: {envelope.request_id}")

                    new_task_state: Optional[TaskState] = None
                    actor_instance: Optional[CallableActor] = None
                    target_task_info = await self._task_registry.get_task(target_control_task_id)

                    if target_task_info and target_task_info.actor_id:
                        actor_instance = await self.actor_manager.get_actor_instance(target_task_info.actor_id)

                    if action == ControlAction.PAUSE:
                        logger.info(f"Processing PAUSE for task_id '{target_control_task_id}'")
                        if actor_instance:
                            try: await actor_instance.request_pause()
                            except Exception as e: logger.error(f"Error calling request_pause on actor {target_task_info.actor_id if target_task_info else 'unknown'}: {e}", exc_info=True)
                        else: logger.warning(f"No active actor instance found for {target_task_info.actor_id if target_task_info else 'unknown actor'} (task {target_control_task_id}) to send pause signal.")
                        new_task_state = TaskState.PAUSED
                    elif action == ControlAction.RESUME:
                        logger.info(f"Processing RESUME for task_id '{target_control_task_id}' with followup: {bool(followup_data)}")
                        if actor_instance:
                            try: await actor_instance.request_resume(followup_data)
                            except Exception as e: logger.error(f"Error calling request_resume on actor {target_task_info.actor_id if target_task_info else 'unknown'}: {e}", exc_info=True)
                        else: logger.warning(f"No active actor instance found for {target_task_info.actor_id if target_task_info else 'unknown actor'} (task {target_control_task_id}) to send resume signal.")
                        new_task_state = TaskState.PENDING
                    elif action == ControlAction.CANCEL:
                        logger.info(f"Processing CANCEL for task_id '{target_control_task_id}'")
                        if actor_instance:
                            try: await actor_instance.request_cancel()
                            except Exception as e: logger.error(f"Error calling request_cancel on actor {target_task_info.actor_id if target_task_info else 'unknown'}: {e}", exc_info=True)
                        else: logger.warning(f"No active actor instance for {target_task_info.actor_id if target_task_info else 'unknown actor'} (task {target_control_task_id}) to send cancel signal.")
                        new_task_state = TaskState.CANCELED

                    if new_task_state:
                        update_success = await self._task_registry.update_task_state(target_control_task_id, new_task_state)
                        if not update_success:
                            logger.warning(f"Target task_id '{target_control_task_id}' for control action '{action.value}' not found in registry for state update.")
                            result.status = f"CONTROL_TARGET_NOT_FOUND"; result.metadata["control_error"] = f"Target task {target_control_task_id} not found."
                        else:
                            result.status = f"CONTROL_{action.value.upper()}_PROCESSED"
                    else: result.status = "CONTROL_UNKNOWN_ACTION"

                    result.metadata["control_info"] = f"Processed control: {action.value} for task {target_control_task_id}"
                    await stream_queue.put({"status": result.status, "task_id": target_control_task_id, "action": action.value})
                except ValidationError as ve:
                    logger.error(f"Invalid ControlPayload for envelope {envelope.request_id}: {ve}", exc_info=True)
                    result.status = TaskState.FAILED.value; result.metadata["error"] = f"Invalid ControlPayload: {ve.errors()}"
                    await self._task_registry.update_task_state(task_id_for_registry, TaskState.FAILED, error_info=result.metadata['error'])
                    await stream_queue.put({"error": "Invalid ControlPayload", "details": ve.errors()})
                return

            actors_to_create_spec = [(actor_type_hint, actor_key)]
            actors, created_actor_ids = await self._create_actors(actors_to_create_spec, result)

            if not actors:
                logger.warning(f"Actor creation failed for spec {actors_to_create_spec}, task: {task_id_for_registry}")
                if not result.metadata.get('error'): result.status = TaskState.FAILED.value; result.metadata['error'] = "Actor creation phase failed."
                await self._task_registry.update_task_state(task_id_for_registry, TaskState.FAILED, error_info=result.metadata['error'])
                return

            primary_actor_id = created_actor_ids[0]
            task_info_to_update = await self._task_registry.get_task(task_id_for_registry)
            if task_info_to_update:
                task_info_to_update.actor_id = primary_actor_id
                await self._task_registry.add_task(task_info_to_update)
            else: logger.error(f"TaskInfo for {task_id_for_registry} not found after actor creation!")

            if not existing_actor_info:
                actor_info_after_creation = await self.actor_manager.info(primary_actor_id)
                if actor_info_after_creation and actor_info_after_creation.capability:
                    if envelope.kind not in actor_info_after_creation.capability.accepts:
                        error_message = f"Newly acquired actor {primary_actor_id} does not accept kind '{envelope.kind}'."
                        logger.error(f"{error_message} Task: {task_id_for_registry}. Accepted: {list(actor_info_after_creation.capability.accepts)}")
                        result.status = TaskState.FAILED.value; result.metadata['error'] = error_message
                        await self._task_registry.update_task_state(task_id_for_registry, TaskState.FAILED, error_info=error_message)
                        await stream_queue.put({"error": error_message, "status": result.status, "accepted_kinds": list(actor_info_after_creation.capability.accepts)});
                        return
                    logger.debug(f"Capability check passed for new actor {primary_actor_id}, task: {task_id_for_registry}")
                elif not actor_info_after_creation :
                     logger.error(f"Failed to retrieve info for actor {primary_actor_id}. Task: {task_id_for_registry}")
                     result.status = TaskState.FAILED.value; result.metadata['error'] = f"Info for {primary_actor_id} unavailable post-acquisition."
                     await self._task_registry.update_task_state(task_id_for_registry, TaskState.FAILED, error_info=result.metadata['error'])
                     await stream_queue.put({"error": result.metadata['error'], "status": result.status});
                     return

            await self._task_registry.update_task_state(task_id_for_registry, TaskState.RUNNING)
            result.status = TaskState.RUNNING.value
            run_input = ActorRunInput(
                request_id=envelope.request_id, context_id=envelope.session_id,
                envelope_kind=envelope.kind, headers=envelope.headers,
                payload=envelope.payload, metadata={}
            )
            logger.debug(f"Calling _run_and_collect_outputs for task: {task_id_for_registry}")
            await self._run_and_collect_outputs(actors, run_input, stream_queue, result)

            if result.status == TaskState.COMPLETED.value:
                await self._task_registry.update_task_state(task_id_for_registry, TaskState.COMPLETED)
            elif result.status == TaskState.FAILED.value:
                 await self._task_registry.update_task_state(task_id_for_registry, TaskState.FAILED, error_info=result.metadata.get('error', 'Unknown error during run'))

        except Exception as e:
            logger.error(f"Unhandled error in _handle_request for task {task_id_for_registry}: {e}", exc_info=True)
            result.status = TaskState.FAILED.value; result.metadata['error'] = str(e)
            result.metadata['error_timestamp'] = datetime.now().isoformat()
            await self._task_registry.update_task_state(task_id_for_registry, TaskState.FAILED, error_info=str(e))
            try:
                await stream_queue.put({"error": str(e), "status": result.status})
            except Exception as sqe:
                logger.error(f"Failed to put error on stream_queue for task {task_id_for_registry}: {sqe}")
        finally:
            if created_actor_ids:
                logger.debug(f"Releasing actors: {created_actor_ids} for task: {task_id_for_registry}")
                for actor_id in created_actor_ids:
                    await self.actor_manager.release(actor_id)
            await stream_queue.put(None)
            if result.runner_id in self._run_tasks: del self._run_tasks[result.runner_id]
            if result.runner_id in self._run_streams: del self._run_streams[result.runner_id]
            logger.info(f"Finished handling task {task_id_for_registry}, final DispatchResult status: {result.status}")

    async def _create_actors(self, actors_to_create_spec: list[tuple[str, str]], result: DispatchResult) -> tuple[list[CallableActor], list[str]]:
        actors: list[CallableActor] = []
        created_actor_ids: list[str] = []
        for actor_type, key in actors_to_create_spec:
            actor_instance = await self.actor_manager.acquire(actor_type, key)
            actors.append(actor_instance)
            actor_id_on_instance = getattr(actor_instance, 'actor_id', None)
            final_actor_id = str(actor_id_on_instance if actor_id_on_instance else f'{actor_type}:{key}')
            created_actor_ids.append(final_actor_id)
            result.agents.append(final_actor_id)
        return actors, created_actor_ids

    async def _run_and_collect_outputs(self, actors: list[CallableActor], run_input: ActorRunInput, stream_queue: asyncio.Queue[ActorRunOutput | None], result: DispatchResult) -> None:
        from ..agent.concurrent_runner import ConcurrentRunner
        from ..agent.runner import AgentRunner
        runner_group = ConcurrentRunner()
        for actor_instance in actors: runner_group.add_runner(AgentRunner(actor_instance)) # type: ignore
        await runner_group.start_all()
        tasks = [asyncio.create_task(self._run_actor_with_semaphore(actor, run_input, stream_queue, result)) for actor in actors]
        errors = await asyncio.gather(*tasks, return_exceptions=True)
        await runner_group.stop_all()
        exceptions = [e for e in errors if isinstance(e, Exception)]
        if exceptions:
            result.status = TaskState.FAILED.value
            if 'errors' not in result.metadata: result.metadata['errors'] = []
            for e_idx, e_item in enumerate(exceptions): result.metadata['errors'].append(f"ActorError{e_idx}: {str(e_item)}")
        else:
            if result.status not in [TaskState.FAILED.value] and not result.status.startswith("CONTROL_"):
                 result.status = TaskState.COMPLETED.value


    async def _run_actor(self, actor: CallableActor, run_input: ActorRunInput, stream_queue: asyncio.Queue[ActorRunOutput | None], result: DispatchResult,) -> None:
        async for output_item in actor.run(run_input):
            processed_output = output_item.model_dump() if hasattr(output_item, 'model_dump') else \
                               (output_item.dict() if hasattr(output_item, 'dict') else output_item)
            await stream_queue.put(processed_output)
            if 'outputs' not in result.metadata: result.metadata['outputs'] = []
            actor_id_str = getattr(actor, 'actor_id', 'unknown_actor')
            result.metadata['outputs'].append(
                {'timestamp': datetime.now().isoformat(), 'output': processed_output, 'actor_id': actor_id_str}
            )

    async def _run_actor_with_semaphore(self, actor: CallableActor, run_input: ActorRunInput, stream_queue: asyncio.Queue[ActorRunOutput | None], result: DispatchResult,) -> None:
        if self._semaphore is None: await self._run_actor(actor, run_input, stream_queue, result)
        else:
            async with self._semaphore: await self._run_actor(actor, run_input, stream_queue, result)

    def get_stream_queue(self, runner_id: str) -> asyncio.Queue[ActorRunOutput | None] | None:
        return self._run_streams.get(runner_id)

    async def cleanup(self) -> None:
        logger.info(f"LocalDispatcher cleaning up {len(self._run_tasks)} active run tasks and {len(self._active_runs)} active runs.")
        active_run_keys = list(self._active_runs.keys())
        for task_obj in self._run_tasks.values():
            if not task_obj.done(): task_obj.cancel()
        if self._run_tasks:
            await asyncio.gather(*self._run_tasks.values(), return_exceptions=True)
        for runner_id in active_run_keys:
            dispatch_result = self._active_runs.get(runner_id)
            if dispatch_result and dispatch_result.task_id:
                task_info = await self._task_registry.get_task(dispatch_result.task_id)
                if task_info and task_info.current_state not in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
                    logger.warning(f"Task {dispatch_result.task_id} (runner {runner_id}) was {task_info.current_state.value} during dispatcher cleanup. Marking as FAILED.")
                    await self._task_registry.update_task_state(dispatch_result.task_id, TaskState.FAILED, error_info="Dispatcher cleanup initiated while task was active.")
        self._run_tasks.clear(); self._active_runs.clear(); self._run_streams.clear()
        logger.info("LocalDispatcher cleanup complete.")

[end of src/codin/actor/dispatcher.py]
