"""Base interfaces for agent and planner components within CoDIN.

This module defines the core `Agent` Abstract Base Class (ABC) and the
`Planner` ABC. The `Agent` interface is designed to be compatible with the
`CallableActor` protocol from `codin.actor.types`, enabling agents to be
managed by the actor system.
"""

import abc
import typing as _t # For _t.AsyncIterator, _t.AsyncGenerator
from ..actor.types import ActorRunInput, ActorRunOutput
from .types import (
    State,
    Step,
    Plan,
)
from ..tool.base import Tool
from ..id import new_id


__all__ = [
    'Agent',
    'Planner',
]


class Agent(abc.ABC):
    """Abstract Base Class for Agent.

    This class defines the essential methods and properties for an agent.

    Attributes:
        id: Unique identifier for the agent.
        name: Name of the agent.
        description: A brief description of the agent's purpose.
        version: The version of the agent.
        tools: A list of `Tool` instances available to the agent.
    """

    id: str
    name: str
    description: str
    version: str
    tools: list[Tool]

    def __init__(
        self,
        *,
        id: str | None = None,
        name: str,
        description:str,
        version: str = '1.0.0',
        tools: list[Tool] | None = None,
    ) -> None:
        """Initializes the base agent attributes.

        Args:
            id: Unique identifier for the agent. If None, a new ID is generated
                based on the agent's name.
            name: Name of the agent.
            description: Description of the agent's capabilities or purpose.
            version: Version string for the agent (e.g., "1.0.0").
            tools: A list of `Tool` objects that the agent can utilize.
        """
        self.name = name
        self.description = description
        self.version = version
        self.id = id or new_id(prefix=self.name) # Automatically generate ID if not provided
        self.tools = tools or []

    @abc.abstractmethod
    async def run(self, input: ActorRunInput) -> _t.AsyncIterator[ActorRunOutput]:
        """Executes the agent's core logic and asynchronously yields outputs.

        This method takes an `ActorRunInput` object, processes it, and should
        yield `ActorRunOutput` items as results become available. This makes
        it suitable for streaming outputs. It must be implemented by concrete
        subclasses.

        Args:
            input: An `ActorRunInput` instance containing the data and metadata
                   for the agent to process.

        Yields:
            `ActorRunOutput`: Results from the agent's processing.
        """
        ... # Ellipsis for abstract async generator method

    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Performs any necessary cleanup before the agent is discarded.

        This method should be implemented by subclasses to release any resources
        (e.g., network connections, temporary files) held by the agent.
        It is called by the actor management system when the agent is released.
        """
        ... # Ellipsis for abstract method


class Planner(abc.ABC):
    """Abstract Base Class for a Planner.

    A Planner is responsible for determining the next sequence of operations (Steps)
    an agent should take based on its current state. It typically interacts with
    Language Models (LLMs) or other reasoning engines.
    """

    @abc.abstractmethod
    async def next(self, state: State) -> _t.AsyncGenerator[Step, None]:
        """Generates the next execution step(s) based on the current state.

        Planners should analyze the provided `State` (which is read-only)
        and yield one or more `Step` objects that direct the agent's actions.

        Args:
            state: The current comprehensive execution state of the agent.

        Yields:
            `Step`: An object representing the next action or decision
                   for the agent to execute.
        """
        ... # Ellipsis for abstract async generator method


    @abc.abstractmethod
    async def reset(self, state: State) -> None:
        """Resets the planner's internal state, if any, based on the provided agent state.

        Args:
            state: The current agent state, which might be used to re-initialize
                   the planner.
        """
        ...
