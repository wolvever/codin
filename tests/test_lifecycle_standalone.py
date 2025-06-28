#!/usr/bin/env python3
"""Standalone test script for the lifecycle module."""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import directly to avoid circular imports
from codin.lifecycle import LifecycleManager, LifecycleMixin, LifecycleState, lifecycle_context


class TestResource(LifecycleMixin):
    """Test resource implementation."""
    
    def __init__(self, name: str, fail_on_up: bool = False):
        super().__init__()
        self.name = name
        self.fail_on_up = fail_on_up
        self.startup_called = False
        self.shutdown_called = False
    
    async def _up(self) -> None:
        """Start the resource."""
        if self.fail_on_up:
            raise RuntimeError(f"Failed to start {self.name}")
        self.startup_called = True
        print(f"✓ {self.name} started")
    
    async def _down(self) -> None:
        """Stop the resource."""
        self.shutdown_called = True
        print(f"✓ {self.name} stopped")


async def test_basic_lifecycle():
    """Test basic lifecycle functionality."""
    print("Testing basic lifecycle...")
    
    # Test single resource
    resource = TestResource("test-resource")
    assert resource.state == LifecycleState.DOWN
    assert resource.is_down
    
    await resource.up()
    assert resource.state == LifecycleState.UP
    assert resource.is_up
    assert resource.startup_called
    
    await resource.down()
    assert resource.state == LifecycleState.DOWN
    assert resource.is_down
    assert resource.shutdown_called
    
    print("✓ Basic lifecycle test passed")


async def test_lifecycle_manager():
    """Test lifecycle manager."""
    print("Testing lifecycle manager...")
    
    # Create manager and resources
    manager = LifecycleManager()
    resource1 = TestResource("resource-1")
    resource2 = TestResource("resource-2")
    
    manager.add_resource(resource1)
    manager.add_resource(resource2)
    
    # Start all
    await manager.up_all()
    assert manager.all_up
    assert resource1.is_up
    assert resource2.is_up
    
    # Stop all
    await manager.down_all()
    assert not manager.all_up
    assert resource1.is_down
    assert resource2.is_down
    
    print("✓ Lifecycle manager test passed")


async def test_context_manager():
    """Test lifecycle context manager."""
    print("Testing context manager...")
    
    resource1 = TestResource("context-1")
    resource2 = TestResource("context-2")
    
    async with lifecycle_context(resource1, resource2) as manager:
        assert resource1.is_up
        assert resource2.is_up
        assert manager.all_up
        print("✓ Resources are up inside context")
    
    # Resources should be down after context
    assert resource1.is_down
    assert resource2.is_down
    print("✓ Resources are down after context")
    
    print("✓ Context manager test passed")


async def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    
    # Test startup failure
    failing_resource = TestResource("failing", fail_on_up=True)
    
    try:
        await failing_resource.up()
        assert False, "Expected exception"
    except RuntimeError:
        assert failing_resource.state == LifecycleState.ERROR
        assert failing_resource.is_error
        print("✓ Startup failure handled correctly")
    
    # Test recovery from error
    failing_resource.fail_on_up = False
    await failing_resource.up()
    assert failing_resource.is_up
    print("✓ Recovery from error state works")
    
    print("✓ Error handling test passed")


async def test_restart():
    """Test restart functionality."""
    print("Testing restart...")
    
    resource = TestResource("restart-test")
    await resource.up()
    
    await resource.restart()
    assert resource.is_up
    assert resource.shutdown_called
    assert resource.startup_called
    
    print("✓ Restart test passed")


async def main():
    """Run all tests."""
    print("Running lifecycle module tests...\n")
    
    await test_basic_lifecycle()
    await test_lifecycle_manager()
    await test_context_manager()
    await test_error_handling()
    await test_restart()
    
    print("\nAll lifecycle tests passed! ✓")


if __name__ == "__main__":
    asyncio.run(main())