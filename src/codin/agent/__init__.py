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
from ..memory import ChatHistory, MemoryService
from ..artifact import ArtifactService

# Import service implementations from new locations
# Note: MemorySystemService not imported here to avoid circular imports
from ..artifact import InMemoryArtifactService
from ..session import SessionService, ReplayService, TaskService

# Import core architecture components
from .planner import Planner
from .session import Session, SessionManager
from .base_agent import BaseAgent

# Import concrete implementations
from .code_planner import CodePlanner, CodePlannerConfig
from .code_agent import CodeAgent

__all__ = [
    # Base agent interface
    "Agent", 
    "CodeAgent",
    
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
    "ChatHistory",
    "MemoryService", 
    "ArtifactService",
    
    # Service implementations
    # Note: MemorySystemService not exported here to avoid circular imports
    "InMemoryArtifactService", 
    "SessionService",
    "ReplayService",
    "TaskService",
    
    # Core components
    "Planner",
    "Session",
    "SessionManager", 
    "BaseAgent",
    
    # Concrete planners
    "CodePlanner",
    "CodePlannerConfig",
] 