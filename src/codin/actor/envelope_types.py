"""Defines Pydantic models for message enveloping, task states, and control actions.

This module includes types for Envelope, EnvelopeHeaders, EnvelopeKind,
AuthDetails, ControlPayload, Capability, TaskState, and ControlAction,
which are used to wrap, describe, and manage messages or tasks sent to actors.
"""

from __future__ import annotations

import uuid # Moved import to the top
from typing import Any, Optional
from typing import Set as PySet
from enum import Enum
from pydantic import BaseModel, Field


class TaskState(str, Enum):
    """Enumerates the possible states of a task within the actor system."""
    SUBMITTED = "SUBMITTED"  # Task has been submitted to the dispatcher
    PENDING = "PENDING"    # Task is accepted, awaiting resource or scheduling
    RUNNING = "RUNNING"    # Task is actively being processed by an actor
    PAUSED = "PAUSED"      # Task execution is paused (e.g., awaiting input, explicit pause)
    COMPLETED = "COMPLETED"  # Task finished successfully
    FAILED = "FAILED"      # Task terminated due to an error
    CANCELED = "CANCELED"    # Task was canceled by a user or system request


class EnvelopeKind(str, Enum):
    """Enumerates the kinds of payloads an Envelope can carry."""
    A2A_MESSAGE = "a2a.message"
    A2A_TASK = "a2a.task"
    PLAIN_TASK = "plain.task"
    CONTROL = "control"


class AuthDetails(BaseModel):
    """Represents authentication details for an envelope header."""
    scheme: str
    token: str


class EnvelopeHeaders(BaseModel):
    """Defines headers for an Envelope."""
    ce_source: Optional[str] = Field(None, alias="ce-source")
    ce_time: Optional[str] = Field(None, alias="ce-time")
    traceparent: Optional[str] = None
    tracestate: Optional[str] = None
    reply_to: Optional[str] = Field(None, alias="reply-to")
    priority: Optional[int] = None
    expires_at: Optional[str] = Field(None, alias="expires-at")
    auth: Optional[AuthDetails] = None

    class Config:
        allow_population_by_field_name = True


class Envelope(BaseModel):
    """Represents an envelope that wraps a payload for actor communication."""
    version: str = "1.0"
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    kind: EnvelopeKind
    task_id: Optional[str] = None # Crucial for CONTROL kind to target specific tasks
    actor_hint: Optional[str] = None
    headers: EnvelopeHeaders = Field(default_factory=EnvelopeHeaders)
    payload: Any


class ControlAction(str, Enum):
    """Enumerates control actions that can be sent in a ControlPayload."""
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"


class ControlPayload(BaseModel):
    """Defines the payload structure for an Envelope of kind CONTROL.

    Attributes:
        task_control: The specific control action to perform.
        followup: Optional dictionary for any follow-up actions or parameters
                  related to the control signal (e.g., data needed for resume).
    """
    task_control: ControlAction # Changed from str to ControlAction Enum
    followup: Optional[dict[str, Any]] = None


class Capability(BaseModel):
    """Describes the capabilities of an actor."""
    accepts: PySet[EnvelopeKind]
    payload_schema: Optional[type] = None

    class Config:
        arbitrary_types_allowed = True


__all__ = [
    "TaskState",
    "EnvelopeKind",
    "AuthDetails",
    "EnvelopeHeaders",
    "Envelope",
    "ControlAction",
    "ControlPayload",
    "Capability",
]
