"""Defines core data structures and protocols for actor interactions in CoDIN.

This module includes:
- `ActorRunInput`: Standardized input structure for actor execution, derived
  from unpacked `Envelope` data.
- `ActorRunOutput`: A flexible type alias for actor execution results.
- `CallableActor`: A protocol defining the essential interface for runnable actors,
  including task execution, cleanup, and control (cancel, pause, resume).
"""

from __future__ import annotations

from typing import Any, Protocol, AsyncIterator, Optional # Added Optional
from pydantic import BaseModel, Field

# Import types from the envelope definitions
from .envelope_types import EnvelopeKind, EnvelopeHeaders


class ActorRunInput(BaseModel):
    """Represents the unpacked input for an actor's execution run, derived from an Envelope.

    This structure provides an actor with the necessary information from an
    incoming `Envelope`, including its kind, headers, and the raw payload.
    The actor is responsible for interpreting and deserializing the `payload`
    based on the `envelope_kind` and potentially information in `headers`.

    Attributes:
        request_id: A unique identifier for the specific request. This typically
            originates from the `Envelope.request_id` and is crucial for
            tracking, logging, and correlation.
        context_id: An optional identifier for a broader context or session,
            often mapped from `Envelope.session_id`. Used to correlate
            multiple actor runs or associate them with a user session.
        envelope_kind: The `EnvelopeKind` of the original envelope, indicating the
            nature of the payload (e.g., A2A message, plain task, control signal).
        headers: The `EnvelopeHeaders` from the original envelope, providing
            additional metadata like tracing information, reply addresses, etc.
        payload: The raw, unparsed payload from the `Envelope.payload`. The actor
            must deserialize or interpret this field based on `envelope_kind`
            and/or `headers` (e.g., a content-type header).
        metadata: A dictionary for any additional internal metadata that might be
            relevant for the actor's execution, not directly part of the
            envelope structure but potentially added during dispatching or
            pre-processing.
    """
    request_id: str
    context_id: str | None = None
    envelope_kind: EnvelopeKind
    headers: EnvelopeHeaders
    payload: Any # Raw payload, actor needs to deserialize based on kind/headers
    metadata: dict[str, Any] = Field(default_factory=dict)


ActorRunOutput = Any


class CallableActor(Protocol):
    """Defines the essential interface for an actor managed by the CoDIN system.

    Actors implementing this protocol can be invoked to process tasks via `run`,
    can be requested to cancel, pause, or resume their current work, and
    provide a mechanism for resource cleanup.
    """

    async def run(self, input: ActorRunInput) -> AsyncIterator[ActorRunOutput]:
        """Processes the given input and asynchronously yields outputs."""
        ...

    async def request_cancel(self) -> None:
        """Requests the actor to gracefully cancel its current work.

        Implementations should set an internal flag that `run` can check.
        The `run` method should then stop processing, release transient resources,
        and potentially yield a final status indicating cancellation.
        This method should be quick and non-blocking.
        """
        ...

    async def request_pause(self) -> None:
        """Requests the actor to pause its current work, if pausable.

        Implementations should set an internal flag. The `run` method should
        check this flag and, if set, suspend its operation (e.g., by awaiting
        a resume signal or periodically sleeping). The actor should ideally still
        be responsive to other control signals like `request_cancel`.
        This method should be quick and non-blocking.
        """
        ...

    async def request_resume(self, followup_data: Optional[dict]) -> None:
        """Requests the actor to resume its paused work.

        Implementations should clear the internal pause flag. The `run` method,
        if paused, should detect this change and continue its operation.
        The `followup_data` can be used to provide additional context or
        instructions needed for resumption.

        Args:
            followup_data: Optional dictionary containing data that might be
                           needed for the actor to resume its work correctly.
        """
        ...

    async def cleanup(self) -> None:
        """Performs any necessary cleanup before the actor is discarded."""
        ...


__all__ = [
    "ActorRunInput",
    "ActorRunOutput",
    "CallableActor",
]
