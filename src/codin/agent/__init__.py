"""Agent system for codin.

This module provides the core agent infrastructure including base classes,
implementations, and supporting types for creating and managing AI agents
in the codin framework.
"""

# Import codin architecture components
from ..memory.base import MemMemoryService, Memory
from ..model.base import BaseLLM
from ..tool.base import Tool
from .base import Agent, Planner
from .concurrent_runner import ConcurrentRunner
from .runner import AgentRunner

# Import new architecture types
from .types import (
    AgentRunInput,
    AgentRunOutput,
    Event,
    EventStep,
    EventType,
    FinishStep,
    # A2A Compatible types
    Message,
    MessageStep,
    # Configuration and metrics
    Metrics,
    RunConfig,
    RunEvent,
    # State and Steps
    State,
    Step,
    StepType,
    Task,
    ThinkStep,
    ToolCallStep,
)


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
    # Base agent interface
    'Agent',
    'Planner',
    # Input/Output types
    'AgentRunInput',
    'AgentRunOutput',
    # Core architecture types
    'State',
    'Step',
    'StepType',
    'MessageStep',
    'EventStep',
    'ToolCallStep',
    'ThinkStep',
    'FinishStep',
    'EventType',
    # A2A Compatible types
    'Message',
    'Task',
    'Event',
    'RunEvent',
    # Configuration and metrics
    'Metrics',
    'RunConfig',
    # Codin architecture components
    'Memory',
    'MemMemoryService',
    'BaseLLM',
    'Tool',
    'AgentRunner',
    'ConcurrentRunner',
    # Lazy access functions
    'get_base_agent',
    'get_base_planner',
    'get_code_agent',
    'get_codeact_planner',
    'get_search_agent',
]

