"""Agent system for codin.

This module provides the core agent infrastructure including base classes,
implementations, and supporting types for creating and managing AI agents
in the codin framework.
"""

# Import codin architecture components
from ..memory.base import MemMemoryService, Memory
from ..model.base import BaseLLM
from ..tool.base import Tool

# Core interfaces and implementations
from .base import Agent, Planner
from .types import (
    StepType,
    Step,
    MessageStep,
    ToolCallStep,
    PlanStep,
    FinishStep,
    Plan,
    State,
    Message,
    Role,
)
from .session import Session
from .runner import AgentRunner as Runner
from .base_agent import BaseAgent
from .planners import BasicPlanner, ReactivePlanner, CodingAssistantPlanner
from .factory import AgentFactory, create_agent, create_local_agent, create_remote_agent
from .config import AgentEndpointConfig




__all__ = [
    # Core interfaces
    "Agent",
    "Planner",
    # Core types
    "StepType",
    "Step",
    "MessageStep",
    "ToolCallStep",
    "PlanStep",
    "FinishStep",
    "Plan",
    "State",
    "Message",
    "Role",
    # Core implementations
    "Session",
    "Runner",
    "BaseAgent",
    # Planners
    "BasicPlanner",
    "ReactivePlanner",
    "CodingAssistantPlanner",
    # Services
    "AgentFactory",
    "AgentEndpointConfig",
    "create_agent",
    "create_local_agent", 
    "create_remote_agent",
    # Legacy exports
    "Memory",
    "MemMemoryService",
    "BaseLLM",
    "Tool",
]


