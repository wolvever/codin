import abc
import typing as _t

from .types import (
    Step,
    StepType, 
    ThinkStep,
    MessageStep,
    ToolCallStep,
    FinishStep,
    State
)

__all__ = [
    "Step",
    "StepType", 
    "ThinkStep",
    "MessageStep",
    "ToolCallStep",
    "FinishStep",
    "State",
    "Planner",
]


class Planner(abc.ABC):
    """Abstract planner that generates execution steps from state."""
    
    @abc.abstractmethod
    async def next(self, state: State) -> _t.AsyncGenerator[Step, None]:
        """Generate the next execution steps based on current state.
        
        Args:
            state: Current conversation and execution state
            
        Yields:
            Step objects representing what the agent should do next
        """
        ...
    
    @abc.abstractmethod 
    def get_config(self) -> dict[str, _t.Any]:
        """Get planner configuration."""
        ...
    
    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Cleanup planner resources."""
        ... 