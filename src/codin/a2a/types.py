"""Defines Pydantic models for A2A (Agent-to-Agent/Actor-to-Actor) Task Payloads.

This module includes `A2ATaskPayload` which standardizes the structure of
payloads for envelopes of kind `A2A_TASK`. It also includes helper models
like `A2AParam` and `A2AResult` for structured parameters and results.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# Path to envelope_types is ..actor.envelope_types from src/codin/a2a/types.py
from ..actor.envelope_types import TaskState


class A2AParam(BaseModel):
    """Represents a structured parameter for an A2A task.

    Attributes:
        name: The name of the parameter.
        value: The value of the parameter.
        param_type: Optional type string for the parameter (e.g., "string", "number", "object").
                    Pydantic alias "type" allows using `type` in input dicts.
    """
    name: str
    value: Any
    param_type: Optional[str] = Field(None, alias="type")

    class Config:
        """Pydantic configuration."""
        allow_population_by_field_name = True


class A2AResult(BaseModel):
    """Represents a structured result from an A2A task.

    Attributes:
        name: The name of the result field.
        value: The value of the result.
        result_type: Optional type string for the result (e.g., "string", "number", "object").
                     Pydantic alias "type" allows using `type` in input dicts.
    """
    name: str
    value: Any
    result_type: Optional[str] = Field(None, alias="type")

    class Config:
        """Pydantic configuration."""
        allow_population_by_field_name = True


class A2ATaskPayload(BaseModel):
    """Defines the structure for the payload of an `Envelope` with `kind=A2A_TASK`.

    This model standardizes how A2A tasks are represented, including their
    name, parameters, status, progress, results, and any errors.

    Attributes:
        task_name: An optional name or type for the A2A task, helping the
                   receiving actor to identify and route the task.
        params: A list of structured parameters (`A2AParam`) for the task.
                Using a list allows for ordered parameters or multiple parameters
                with the same name if needed by the task definition.
                Alternatively, `Dict[str, Any]` could be used for simpler key-value params.
        current_status: Optional current `TaskState` of the task, if the payload
                        is conveying an update about an ongoing or completed task.
        progress: Optional progress indicator, typically a percentage (0-100).
        result: A list of structured results (`A2AResult`) if the task has completed
                and produced output.
        error: An optional dictionary containing error information if the task failed.
               Common fields might include "code", "message", "details".
        metadata: Optional dictionary for any additional metadata specific to this
                  A2A task payload.
    """
    task_name: Optional[str] = None
    params: Optional[List[A2AParam]] = None # Using List of A2AParam
    current_status: Optional[TaskState] = None
    progress: Optional[int] = Field(None, ge=0, le=100) # Progress as percentage
    result: Optional[List[A2AResult]] = None # Using List of A2AResult
    error: Optional[Dict[str, Any]] = None # e.g., {"code": "E404", "message": "Not found"}
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


__all__ = [
    "A2AParam",
    "A2AResult",
    "A2ATaskPayload",
]
