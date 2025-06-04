"""Agent adapters subpackage."""

# Import existing components
from .base import Agent
from .types import AgentRunInput, AgentRunOutput, ToolCall, ToolCallResult

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
    
    # Task management
    TaskStatus,
    TaskInfo,
    
    # Configuration and metrics
    Metrics,
    AgentConfig,
)

# Import service interfaces from their respective modules
from ..memory import Memory, MemoryWriter, InMemoryService
from ..artifact import ArtifactService

# Import service implementations from new locations
from ..artifact import InMemoryArtifactService

# Import core architecture components
from .planner import Planner

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
    
    # Input/Output types
    "AgentRunInput",
    "AgentRunOutput", 
    "ToolCall",
    "ToolCallResult",
    
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
    
    # Task management
    "TaskStatus",
    "TaskInfo",
    
    # Configuration and metrics  
    "Metrics",
    "AgentConfig",
    
    # Service interfaces
    "Memory",
    "MemoryWriter", 
    "ArtifactService",
    
    # Service implementations
    "InMemoryService", 
    "InMemoryArtifactService", 
    
    # Core components
    "Planner",
    
    # Lazy access functions
    "get_base_agent",
    "get_code_planner", 
    "get_code_agent",
] 