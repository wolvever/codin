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
    """Stateless planner that generates execution steps from state.
    
    The planner is stateless and only READS from State - it never modifies it.
    All state changes are handled by the Agent that orchestrates the planner.
    """
    
    @abc.abstractmethod
    async def next(self, state: State) -> _t.AsyncGenerator[Step, None]:
        """Generate the next execution steps based on current state.
        
        Args:
            state: Current comprehensive execution state (READ-ONLY)
            
        Yields:
            Step objects representing what the agent should do next
            
        Note:
            The planner must ONLY read from state - any modifications should
            be yielded as Steps for the Agent to execute and apply to state.
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