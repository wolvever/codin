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
    TaskControl,
    AgentRunInput,
    AgentRunOutput,
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
from .agent import BasicAgent
from .planners import BasicPlanner, ReactivePlanner, CodingAssistantPlanner
from .discovery import AgentRegistry
from .distributed import RayAgent, AgentFactory


# Lazy imports to avoid circular dependencies
def get_base_agent():
    """Lazy import BaseAgent to avoid circular imports."""
    from .base_agent import BaseAgent

    return BaseAgent


def get_base_planner():
    """Lazy import BasePlanner to avoid circular imports."""
    from .base_planner import BasePlanner, BasePlannerConfig

    return BasePlanner, BasePlannerConfig


def get_code_agent():
    """Lazy import CodeAgent to avoid circular imports."""
    from .code_agent import CodeAgent

    return CodeAgent

  
def get_codeact_planner():
    """Lazy import CodeActPlanner to avoid circular imports."""
    from .codeact_planner import CodeActPlanner

    return CodeActPlanner

def get_search_agent():
    """Lazy import SearchAgent to avoid circular imports."""
    from .search_agent import SearchAgent

    return SearchAgent


__all__ = [
    # Core interfaces
    "Agent",
    "Planner",
    # Core types
    "TaskControl",
    "AgentRunInput",
    "AgentRunOutput",
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
    "BasicAgent",
    # Planners
    "BasicPlanner",
    "ReactivePlanner",
    "CodingAssistantPlanner",
    # Services
    "AgentRegistry",
    "RayAgent",
    "AgentFactory",
    # Legacy exports
    "Memory",
    "MemMemoryService",
    "BaseLLM",
    "Tool",
    "get_base_agent",
    "get_base_planner",
    "get_code_agent",
    "get_codeact_planner",
    "get_search_agent",
]


