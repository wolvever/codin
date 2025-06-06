"""Lifecycle management for tools and services."""

import abc
import enum
import logging

from contextlib import asynccontextmanager


__all__ = [
    'LifecycleManager',
    'LifecycleMixin',
    'LifecycleState',
    'lifecycle_context',
]

logger = logging.getLogger(__name__)


class LifecycleState(str, enum.Enum):
    """Lifecycle states for tools and services."""

    DOWN = 'down'  # Not started or shut down
    STARTING = 'starting'  # In the process of starting up
    UP = 'up'  # Running and ready
    STOPPING = 'stopping'  # In the process of shutting down
    ERROR = 'error'  # In error state
    DISCONNECTED = 'disconnected'  # Temporarily disconnected but can reconnect


class LifecycleMixin(abc.ABC):
    """Base mixin for lifecycle management.

    Provides common lifecycle methods for setting up and cleaning up resources.
    This replaces the old initialize/cleanup pattern with a more robust up/down pattern.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = LifecycleState.DOWN
        self._logger = logging.getLogger(f'{self.__class__.__module__}.{self.__class__.__name__}')

    @property
    def state(self) -> LifecycleState:
        """Get the current lifecycle state."""
        return self._state

    @property
    def is_up(self) -> bool:
        """Check if the resource is currently up and ready."""
        return self._state == LifecycleState.UP

    @property
    def is_down(self) -> bool:
        """Check if the resource is currently down."""
        return self._state == LifecycleState.DOWN

    @property
    def is_error(self) -> bool:
        """Check if the resource is in an error state."""
        return self._state == LifecycleState.ERROR

    async def up(self) -> None:
        """Bring the resource up.

        This is the main entry point for starting a resource. It handles
        state transitions and error handling.
        """
        if self._state == LifecycleState.UP:
            self._logger.debug('Resource already up')
            return

        if self._state in (LifecycleState.STARTING, LifecycleState.STOPPING):
            raise RuntimeError(f'Cannot start resource in state {self._state}')

        self._logger.debug('Starting resource')
        self._state = LifecycleState.STARTING

        try:
            await self._up()
            self._state = LifecycleState.UP
            self._logger.debug('Resource started successfully')
        except Exception as e:
            self._state = LifecycleState.ERROR
            self._logger.error(f'Failed to start resource: {e}')
            raise

    async def down(self) -> None:
        """Bring the resource down.

        This is the main entry point for stopping a resource. It handles
        state transitions and ensures cleanup even if errors occur.
        """
        if self._state == LifecycleState.DOWN:
            self._logger.debug('Resource already down')
            return

        if self._state == LifecycleState.STOPPING:
            self._logger.debug('Resource already stopping')
            return

        self._logger.debug('Stopping resource')
        self._state = LifecycleState.STOPPING

        try:
            await self._down()
            self._state = LifecycleState.DOWN
            self._logger.debug('Resource stopped successfully')
        except Exception as e:
            self._state = LifecycleState.ERROR
            self._logger.error(f'Error stopping resource: {e}')
            # Don't re-raise - we want to ensure cleanup happens

    async def restart(self) -> None:
        """Restart the resource by bringing it down then up."""
        await self.down()
        await self.up()

    @abc.abstractmethod
    async def _up(self) -> None:
        """Implementation-specific startup logic.

        This method should be implemented by subclasses to establish their specific
        resources (e.g. connections, tools, services, etc).
        """

    @abc.abstractmethod
    async def _down(self) -> None:
        """Implementation-specific shutdown logic.

        This method should be implemented by subclasses to properly clean up their
        specific resources and close any connections.
        """


class LifecycleManager:
    """Manager for coordinating lifecycle of multiple resources."""

    def __init__(self):
        self._resources: list[LifecycleMixin] = []
        self._logger = logging.getLogger(__name__)

    def add_resource(self, resource: LifecycleMixin) -> None:
        """Add a resource to be managed."""
        self._resources.append(resource)

    def remove_resource(self, resource: LifecycleMixin) -> None:
        """Remove a resource from management."""
        if resource in self._resources:
            self._resources.remove(resource)

    async def up_all(self) -> None:
        """Bring up all managed resources."""
        self._logger.debug(f'Starting {len(self._resources)} resources')

        for resource in self._resources:
            try:
                await resource.up()
            except Exception as e:
                self._logger.error(f'Failed to start resource {resource}: {e}')
                # Continue with other resources

    async def down_all(self) -> None:
        """Bring down all managed resources in reverse order."""
        self._logger.debug(f'Stopping {len(self._resources)} resources')

        # Stop in reverse order
        for resource in reversed(self._resources):
            try:
                await resource.down()
            except Exception as e:
                self._logger.error(f'Error stopping resource {resource}: {e}')
                # Continue with other resources

    async def restart_all(self) -> None:
        """Restart all managed resources."""
        await self.down_all()
        await self.up_all()

    @property
    def all_up(self) -> bool:
        """Check if all resources are up."""
        return all(resource.is_up for resource in self._resources)

    @property
    def any_error(self) -> bool:
        """Check if any resource is in error state."""
        return any(resource.is_error for resource in self._resources)


@asynccontextmanager
async def lifecycle_context(*resources: LifecycleMixin):
    """Context manager for automatic lifecycle management.

    Usage:
        async with lifecycle_context(tool1, tool2, toolset) as manager:
            # Resources are automatically brought up
            await do_work()
            # Resources are automatically brought down on exit
    """
    manager = LifecycleManager()
    for resource in resources:
        manager.add_resource(resource)

    try:
        await manager.up_all()
        yield manager
    finally:
        await manager.down_all()
