"""Agent adapters subpackage."""

# Import existing components
from .base import Agent, Planner
from .types import AgentRunInput, AgentRunOutput

# Import new architecture types
from .types import (
    # State and Steps
    State,
    Step, 
    StepType, 
    MessageStep,
    EventStep,
    ToolCallStep,
    ThinkStep, 
    FinishStep, 
    EventType,
    
    # A2A Compatible types
    Message,
    Task,
    Event,
    RunEvent,
    
    # Configuration and metrics
    Metrics,
    RunConfig,
)

# Import codin architecture components
from ..memory.base import Memory, MemMemoryService
from ..model.base import BaseLLM
from ..tool.base import Tool
from ..actor.mailbox import Mailbox

# Lazy imports to avoid circular dependencies
def get_base_agent():
    """Lazy import BaseAgent to avoid circular imports."""
    from .base_agent import BaseAgent
    return BaseAgent

def get_code_planner():
    """Lazy import CodePlanner to avoid circular imports."""
    from .code_planner import CodePlanner, CodePlannerConfig
    return CodePlanner, CodePlannerConfig

def get_code_agent():
    """Lazy import CodeAgent to avoid circular imports.""" 
    from .code_agent import CodeAgent
    return CodeAgent

__all__ = [
    # Base agent interface
    "Agent",
    "Planner",
    
    # Input/Output types
    "AgentRunInput",
    "AgentRunOutput",
    
    # Core architecture types
    "State",
    "Step",
    "StepType",
    "MessageStep",
    "EventStep",
    "ToolCallStep",
    "ThinkStep",
    "FinishStep",
    "EventType",
    
    # A2A Compatible types
    "Message",
    "Task",
    "Event",
    "RunEvent",
    
    # Configuration and metrics  
    "Metrics",
    "RunConfig",
    
    # Codin architecture components
    "Memory",
    "MemMemoryService",
    "BaseLLM",
    "Tool",
    "Mailbox",
    
    # Lazy access functions
    "get_base_agent",
    "get_code_planner", 
    "get_code_agent",
] 