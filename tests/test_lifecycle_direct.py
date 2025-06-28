#!/usr/bin/env python3
"""Direct test of lifecycle module functionality."""

import asyncio
import importlib.util
from pathlib import Path

# Load the lifecycle module directly
src_path = Path(__file__).parent / "src"
lifecycle_path = src_path / "codin" / "lifecycle.py"

spec = importlib.util.spec_from_file_location("lifecycle", lifecycle_path)
lifecycle_module = importlib.util.module_from_spec(spec)

# Execute the module
spec.loader.exec_module(lifecycle_module)

# Get the classes
LifecycleState = lifecycle_module.LifecycleState
LifecycleMixin = lifecycle_module.LifecycleMixin
LifecycleManager = lifecycle_module.LifecycleManager
lifecycle_context = lifecycle_module.lifecycle_context


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


async def test_basic_functionality():
    """Test basic lifecycle functionality."""
    print("Testing basic lifecycle functionality...")
    
    # Test LifecycleState enum
    assert LifecycleState.DOWN == 'down'
    assert LifecycleState.UP == 'up'
    assert LifecycleState.ERROR == 'error'
    print("✓ LifecycleState enum works")
    
    # Test single resource lifecycle
    resource = TestResource("test-resource")
    assert resource.state == LifecycleState.DOWN
    assert resource.is_down
    assert not resource.is_up
    
    await resource.up()
    assert resource.state == LifecycleState.UP
    assert resource.is_up
    assert resource.startup_called
    
    await resource.down()
    assert resource.state == LifecycleState.DOWN
    assert resource.is_down
    assert resource.shutdown_called
    
    print("✓ Basic resource lifecycle works")


async def test_manager():
    """Test lifecycle manager."""
    print("Testing lifecycle manager...")
    
    manager = LifecycleManager()
    resource1 = TestResource("resource-1")
    resource2 = TestResource("resource-2")
    
    manager.add_resource(resource1)
    manager.add_resource(resource2)
    
    await manager.up_all()
    assert manager.all_up
    assert resource1.is_up
    assert resource2.is_up
    
    await manager.down_all()
    assert not manager.all_up
    assert resource1.is_down
    assert resource2.is_down
    
    print("✓ Lifecycle manager works")


async def test_context():
    """Test context manager."""
    print("Testing context manager...")
    
    resource1 = TestResource("context-1")
    resource2 = TestResource("context-2")
    
    async with lifecycle_context(resource1, resource2) as manager:
        assert resource1.is_up
        assert resource2.is_up
        assert manager.all_up
    
    assert resource1.is_down
    assert resource2.is_down
    
    print("✓ Context manager works")


async def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    
    failing_resource = TestResource("failing", fail_on_up=True)
    
    try:
        await failing_resource.up()
        assert False, "Expected exception"
    except RuntimeError:
        assert failing_resource.state == LifecycleState.ERROR
        assert failing_resource.is_error
    
    # Test recovery
    failing_resource.fail_on_up = False
    await failing_resource.up()
    assert failing_resource.is_up
    
    print("✓ Error handling works")


async def main():
    """Run all tests."""
    print("Testing lifecycle module directly...\n")
    
    await test_basic_functionality()
    await test_manager()
    await test_context()
    await test_error_handling()
    
    print("\nAll lifecycle tests passed! ✓")


if __name__ == "__main__":
    asyncio.run(main())