"""Comprehensive tests for the codin.lifecycle module."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add the src directory to the path to import the module directly
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from codin.lifecycle import (
    LifecycleManager,
    LifecycleMixin,
    LifecycleState,
    lifecycle_context,
)


class MockResource(LifecycleMixin):
    """Mock resource for testing lifecycle functionality."""

    def __init__(self, name: str = "mock", fail_on_up: bool = False, fail_on_down: bool = False):
        super().__init__()
        self.name = name
        self.fail_on_up = fail_on_up
        self.fail_on_down = fail_on_down
        self.up_called = False
        self.down_called = False
        self.up_call_count = 0
        self.down_call_count = 0

    async def _up(self) -> None:
        """Mock implementation of _up."""
        self.up_called = True
        self.up_call_count += 1
        if self.fail_on_up:
            raise RuntimeError("Simulated startup failure")

    async def _down(self) -> None:
        """Mock implementation of _down."""
        self.down_called = True
        self.down_call_count += 1
        if self.fail_on_down:
            raise RuntimeError("Simulated shutdown failure")

    def __repr__(self):
        return f"MockResource({self.name})"


class TestLifecycleState:
    """Test cases for the LifecycleState enum."""

    def test_all_states_defined(self):
        """Test that all expected states are defined."""
        expected_states = {
            'DOWN', 'STARTING', 'UP', 'STOPPING', 'ERROR', 'DISCONNECTED'
        }
        actual_states = {state.name for state in LifecycleState}
        assert actual_states == expected_states

    def test_state_values(self):
        """Test that state values are as expected."""
        assert LifecycleState.DOWN == 'down'
        assert LifecycleState.STARTING == 'starting'
        assert LifecycleState.UP == 'up'
        assert LifecycleState.STOPPING == 'stopping'
        assert LifecycleState.ERROR == 'error'
        assert LifecycleState.DISCONNECTED == 'disconnected'

    def test_state_comparison(self):
        """Test state comparison operations."""
        assert LifecycleState.DOWN == LifecycleState.DOWN
        assert LifecycleState.UP != LifecycleState.DOWN
        assert LifecycleState.UP == 'up'
        assert str(LifecycleState.UP) == 'up'


class TestLifecycleMixin:
    """Test cases for the LifecycleMixin class."""

    def test_initial_state(self):
        """Test that resources start in DOWN state."""
        resource = MockResource()
        assert resource.state == LifecycleState.DOWN
        assert resource.is_down
        assert not resource.is_up
        assert not resource.is_error

    def test_state_properties(self):
        """Test state property methods."""
        resource = MockResource()
        
        # Test DOWN state
        resource._state = LifecycleState.DOWN
        assert resource.is_down
        assert not resource.is_up
        assert not resource.is_error
        
        # Test UP state
        resource._state = LifecycleState.UP
        assert not resource.is_down
        assert resource.is_up
        assert not resource.is_error
        
        # Test ERROR state
        resource._state = LifecycleState.ERROR
        assert not resource.is_down
        assert not resource.is_up
        assert resource.is_error

    @pytest.mark.asyncio
    async def test_successful_startup(self):
        """Test successful resource startup."""
        resource = MockResource()
        
        await resource.up()
        
        assert resource.state == LifecycleState.UP
        assert resource.is_up
        assert resource.up_called
        assert resource.up_call_count == 1

    @pytest.mark.asyncio
    async def test_startup_already_up(self):
        """Test that calling up() on an already up resource is a no-op."""
        resource = MockResource()
        await resource.up()
        
        # Call up again
        await resource.up()
        
        assert resource.state == LifecycleState.UP
        assert resource.up_call_count == 1  # Should not be called again

    @pytest.mark.asyncio
    async def test_startup_failure(self):
        """Test resource startup failure handling."""
        resource = MockResource(fail_on_up=True)
        
        with pytest.raises(RuntimeError, match="Simulated startup failure"):
            await resource.up()
        
        assert resource.state == LifecycleState.ERROR
        assert resource.is_error
        assert resource.up_called

    @pytest.mark.asyncio
    async def test_startup_from_error_state(self):
        """Test that resource can start up from error state."""
        resource = MockResource(fail_on_up=True)
        
        # First attempt fails
        with pytest.raises(RuntimeError):
            await resource.up()
        assert resource.state == LifecycleState.ERROR
        
        # Fix the failure condition and try again
        resource.fail_on_up = False
        await resource.up()
        assert resource.state == LifecycleState.UP
        assert resource.up_call_count == 2

    @pytest.mark.asyncio
    async def test_startup_from_invalid_states(self):
        """Test that startup fails from STARTING and STOPPING states."""
        resource = MockResource()
        
        # Test from STARTING state
        resource._state = LifecycleState.STARTING
        with pytest.raises(RuntimeError, match="Cannot start resource in state starting"):
            await resource.up()
        
        # Test from STOPPING state
        resource._state = LifecycleState.STOPPING
        with pytest.raises(RuntimeError, match="Cannot start resource in state stopping"):
            await resource.up()

    @pytest.mark.asyncio
    async def test_successful_shutdown(self):
        """Test successful resource shutdown."""
        resource = MockResource()
        await resource.up()
        
        await resource.down()
        
        assert resource.state == LifecycleState.DOWN
        assert resource.is_down
        assert resource.down_called
        assert resource.down_call_count == 1

    @pytest.mark.asyncio
    async def test_shutdown_already_down(self):
        """Test that calling down() on an already down resource is a no-op."""
        resource = MockResource()
        
        await resource.down()
        
        assert resource.state == LifecycleState.DOWN
        assert not resource.down_called  # Should not call _down() if already down

    @pytest.mark.asyncio
    async def test_shutdown_already_stopping(self):
        """Test that calling down() on a stopping resource is a no-op."""
        resource = MockResource()
        resource._state = LifecycleState.STOPPING
        
        await resource.down()
        
        assert resource.state == LifecycleState.STOPPING
        assert not resource.down_called

    @pytest.mark.asyncio
    async def test_shutdown_failure(self):
        """Test resource shutdown failure handling."""
        resource = MockResource(fail_on_down=True)
        await resource.up()
        
        # Shutdown should not raise exception, but should log error
        with patch.object(resource._logger, 'error') as mock_log:
            await resource.down()
        
        assert resource.state == LifecycleState.ERROR
        assert resource.is_error
        assert resource.down_called
        mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart(self):
        """Test resource restart functionality."""
        resource = MockResource()
        await resource.up()
        
        await resource.restart()
        
        assert resource.state == LifecycleState.UP
        assert resource.up_call_count == 2  # Once for initial up, once for restart
        assert resource.down_call_count == 1

    @pytest.mark.asyncio
    async def test_restart_from_down(self):
        """Test restart from down state."""
        resource = MockResource()
        
        await resource.restart()
        
        assert resource.state == LifecycleState.UP
        assert resource.up_call_count == 1
        assert not resource.down_called  # down() is no-op when already down

    def test_logger_initialization(self):
        """Test that logger is properly initialized."""
        resource = MockResource()
        assert hasattr(resource, '_logger')
        assert resource._logger.name.endswith('MockResource')

    @pytest.mark.asyncio
    async def test_state_transitions_during_startup(self):
        """Test state transitions during startup process."""
        resource = MockResource()
        states = []
        
        original_up = resource._up
        async def tracking_up():
            states.append(resource.state)
            await original_up()
        
        resource._up = tracking_up
        await resource.up()
        
        assert LifecycleState.STARTING in states
        assert resource.state == LifecycleState.UP

    @pytest.mark.asyncio
    async def test_state_transitions_during_shutdown(self):
        """Test state transitions during shutdown process."""
        resource = MockResource()
        await resource.up()
        
        states = []
        original_down = resource._down
        async def tracking_down():
            states.append(resource.state)
            await original_down()
        
        resource._down = tracking_down
        await resource.down()
        
        assert LifecycleState.STOPPING in states
        assert resource.state == LifecycleState.DOWN


class TestLifecycleManager:
    """Test cases for the LifecycleManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = LifecycleManager()
        assert manager._resources == []
        assert hasattr(manager, '_logger')

    def test_add_remove_resource(self):
        """Test adding and removing resources."""
        manager = LifecycleManager()
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2")
        
        # Add resources
        manager.add_resource(resource1)
        manager.add_resource(resource2)
        assert len(manager._resources) == 2
        assert resource1 in manager._resources
        assert resource2 in manager._resources
        
        # Remove resource
        manager.remove_resource(resource1)
        assert len(manager._resources) == 1
        assert resource1 not in manager._resources
        assert resource2 in manager._resources
        
        # Remove non-existent resource (should not raise)
        manager.remove_resource(resource1)
        assert len(manager._resources) == 1

    @pytest.mark.asyncio
    async def test_up_all_success(self):
        """Test bringing up all resources successfully."""
        manager = LifecycleManager()
        resources = [MockResource(f"resource{i}") for i in range(3)]
        
        for resource in resources:
            manager.add_resource(resource)
        
        await manager.up_all()
        
        for resource in resources:
            assert resource.state == LifecycleState.UP
            assert resource.up_called

    @pytest.mark.asyncio
    async def test_up_all_with_failure(self):
        """Test that up_all continues with other resources when one fails."""
        manager = LifecycleManager()
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2", fail_on_up=True)
        resource3 = MockResource("resource3")
        
        manager.add_resource(resource1)
        manager.add_resource(resource2)
        manager.add_resource(resource3)
        
        with patch.object(manager._logger, 'error') as mock_log:
            await manager.up_all()
        
        # First and third resources should be up
        assert resource1.state == LifecycleState.UP
        assert resource3.state == LifecycleState.UP
        # Second resource should be in error state
        assert resource2.state == LifecycleState.ERROR
        
        # Error should be logged
        mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_down_all_success(self):
        """Test bringing down all resources successfully."""
        manager = LifecycleManager()
        resources = [MockResource(f"resource{i}") for i in range(3)]
        
        for resource in resources:
            manager.add_resource(resource)
            await resource.up()
        
        await manager.down_all()
        
        for resource in resources:
            assert resource.state == LifecycleState.DOWN
            assert resource.down_called

    @pytest.mark.asyncio
    async def test_down_all_reverse_order(self):
        """Test that down_all shuts down resources in reverse order."""
        manager = LifecycleManager()
        shutdown_order = []
        
        for i in range(3):
            resource = MockResource(f"resource{i}")
            original_down = resource._down
            
            async def make_tracking_down(resource_name):
                async def tracking_down():
                    shutdown_order.append(resource_name)
                    await original_down()
                return tracking_down
            
            resource._down = await make_tracking_down(f"resource{i}")
            manager.add_resource(resource)
            await resource.up()
        
        await manager.down_all()
        
        # Should shutdown in reverse order
        assert shutdown_order == ["resource2", "resource1", "resource0"]

    @pytest.mark.asyncio
    async def test_down_all_with_failure(self):
        """Test that down_all continues with other resources when one fails."""
        manager = LifecycleManager()
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2", fail_on_down=True)
        resource3 = MockResource("resource3")
        
        for resource in [resource1, resource2, resource3]:
            manager.add_resource(resource)
            await resource.up()
        
        with patch.object(manager._logger, 'error') as mock_log:
            await manager.down_all()
        
        # All resources should be processed
        assert resource1.down_called
        assert resource2.down_called
        assert resource3.down_called
        
        # First and third should be down, second in error
        assert resource1.state == LifecycleState.DOWN
        assert resource2.state == LifecycleState.ERROR
        assert resource3.state == LifecycleState.DOWN
        
        # Error should be logged
        mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_all(self):
        """Test restarting all resources."""
        manager = LifecycleManager()
        resources = [MockResource(f"resource{i}") for i in range(2)]
        
        for resource in resources:
            manager.add_resource(resource)
            await resource.up()
        
        await manager.restart_all()
        
        for resource in resources:
            assert resource.state == LifecycleState.UP
            assert resource.up_call_count == 2  # Once for initial, once for restart
            assert resource.down_call_count == 1

    def test_all_up_property(self):
        """Test the all_up property."""
        manager = LifecycleManager()
        
        # Empty manager should return True
        assert manager.all_up
        
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2")
        manager.add_resource(resource1)
        manager.add_resource(resource2)
        
        # All down
        assert not manager.all_up
        
        # One up, one down
        resource1._state = LifecycleState.UP
        assert not manager.all_up
        
        # All up
        resource2._state = LifecycleState.UP
        assert manager.all_up

    def test_any_error_property(self):
        """Test the any_error property."""
        manager = LifecycleManager()
        
        # Empty manager should return False
        assert not manager.any_error
        
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2")
        manager.add_resource(resource1)
        manager.add_resource(resource2)
        
        # No errors
        assert not manager.any_error
        
        # One error
        resource1._state = LifecycleState.ERROR
        assert manager.any_error
        
        # No errors again
        resource1._state = LifecycleState.UP
        assert not manager.any_error


class TestLifecycleContext:
    """Test cases for the lifecycle_context function."""

    @pytest.mark.asyncio
    async def test_context_manager_success(self):
        """Test successful use of lifecycle context manager."""
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2")
        
        async with lifecycle_context(resource1, resource2) as manager:
            assert isinstance(manager, LifecycleManager)
            assert resource1.state == LifecycleState.UP
            assert resource2.state == LifecycleState.UP
            assert manager.all_up
        
        # Resources should be down after context exit
        assert resource1.state == LifecycleState.DOWN
        assert resource2.state == LifecycleState.DOWN

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self):
        """Test that resources are cleaned up even when exception occurs."""
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2")
        
        with pytest.raises(ValueError):
            async with lifecycle_context(resource1, resource2) as manager:
                assert resource1.state == LifecycleState.UP
                assert resource2.state == LifecycleState.UP
                raise ValueError("Test exception")
        
        # Resources should still be down after exception
        assert resource1.state == LifecycleState.DOWN
        assert resource2.state == LifecycleState.DOWN

    @pytest.mark.asyncio
    async def test_context_manager_startup_failure(self):
        """Test context manager behavior when startup fails."""
        resource1 = MockResource("resource1")
        resource2 = MockResource("resource2", fail_on_up=True)
        
        # Context manager should not raise, but resources should be cleaned up
        async with lifecycle_context(resource1, resource2) as manager:
            assert resource1.state == LifecycleState.UP
            assert resource2.state == LifecycleState.ERROR
            assert not manager.all_up
            assert manager.any_error
        
        # Working resource should be down, failed resource should remain in error
        assert resource1.state == LifecycleState.DOWN

    @pytest.mark.asyncio
    async def test_context_manager_empty(self):
        """Test context manager with no resources."""
        async with lifecycle_context() as manager:
            assert isinstance(manager, LifecycleManager)
            assert manager.all_up  # Empty manager returns True
            assert not manager.any_error

    @pytest.mark.asyncio
    async def test_context_manager_single_resource(self):
        """Test context manager with single resource."""
        resource = MockResource("single")
        
        async with lifecycle_context(resource) as manager:
            assert resource.state == LifecycleState.UP
        
        assert resource.state == LifecycleState.DOWN


class TestIntegration:
    """Integration tests for lifecycle management."""

    @pytest.mark.asyncio
    async def test_complex_lifecycle_scenario(self):
        """Test a complex scenario with multiple resources and failures."""
        # Create resources with different behaviors
        good_resource = MockResource("good")
        startup_failure = MockResource("startup_fail", fail_on_up=True)
        shutdown_failure = MockResource("shutdown_fail", fail_on_down=True)
        
        manager = LifecycleManager()
        manager.add_resource(good_resource)
        manager.add_resource(startup_failure)
        manager.add_resource(shutdown_failure)
        
        # Start up (one will fail)
        await manager.up_all()
        assert good_resource.state == LifecycleState.UP
        assert startup_failure.state == LifecycleState.ERROR
        assert shutdown_failure.state == LifecycleState.UP
        
        # Shutdown (one will fail)
        await manager.down_all()
        assert good_resource.state == LifecycleState.DOWN
        assert shutdown_failure.state == LifecycleState.ERROR
        
        # Restart working resource
        await good_resource.restart()
        assert good_resource.state == LifecycleState.UP
        assert good_resource.up_call_count == 2
        assert good_resource.down_call_count == 2

    @pytest.mark.asyncio
    async def test_nested_context_managers(self):
        """Test nested lifecycle context managers."""
        outer_resource = MockResource("outer")
        inner_resource = MockResource("inner")
        
        async with lifecycle_context(outer_resource):
            assert outer_resource.state == LifecycleState.UP
            
            async with lifecycle_context(inner_resource):
                assert inner_resource.state == LifecycleState.UP
                assert outer_resource.state == LifecycleState.UP
            
            # Inner should be down, outer still up
            assert inner_resource.state == LifecycleState.DOWN
            assert outer_resource.state == LifecycleState.UP
        
        # Both should be down
        assert outer_resource.state == LifecycleState.DOWN
        assert inner_resource.state == LifecycleState.DOWN

    @pytest.mark.asyncio
    async def test_logging_behavior(self):
        """Test that appropriate log messages are generated."""
        resource = MockResource("logged")
        
        with patch.object(resource._logger, 'debug') as mock_debug:
            with patch.object(resource._logger, 'error') as mock_error:
                # Test successful startup
                await resource.up()
                assert mock_debug.call_count >= 2  # Starting and started messages
                
                # Test calling up again (should log already up)
                await resource.up()
                
                # Test successful shutdown
                await resource.down()
                
                # No errors should be logged for successful operations
                mock_error.assert_not_called()