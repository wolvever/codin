"""
Demonstration of the lifecycle-based executor pattern.

This example shows how executors now follow the LifecycleMixin pattern
with proper up/down lifecycle management.
"""

import asyncio
from codin.tool.specs.base import ToolSpec, ToolType, ToolMetadata
from codin.tool.executors.python import PythonExecutor
from codin.tool.base import ToolContext
from codin.lifecycle import lifecycle_context


async def demo_executor_lifecycle():
    """Demonstrate executor lifecycle management."""
    
    print("=== Executor Lifecycle Demo ===")
    
    # Create a simple tool spec
    tool_spec = ToolSpec(
        name="multiply",
        description="Multiply two numbers",
        tool_type=ToolType.PYTHON,
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        },
        implementation_config={
            "function": lambda a, b: a * b
        },
        metadata=ToolMetadata(category="math")
    )
    
    # Create executor
    executor = PythonExecutor()
    print(f"Executor initial state: {executor.state}")
    print(f"Is up: {executor.is_up}")
    print(f"Is down: {executor.is_down}")
    
    # Bring executor up
    print("\nBringing executor up...")
    await executor.up()
    print(f"Executor state after up(): {executor.state}")
    print(f"Is up: {executor.is_up}")
    
    # Setup tool
    print(f"\nSetting up tool: {tool_spec.name}")
    await executor.setup_tool(tool_spec)
    print("Tool setup completed")
    
    # Execute tool
    print("\nExecuting tool...")
    context = ToolContext(session_id="demo")
    result = await executor.execute(tool_spec, {"a": 6, "b": 7}, context)
    print(f"Result: {result}")
    
    # Teardown tool
    print(f"\nTearing down tool: {tool_spec.name}")
    await executor.teardown_tool(tool_spec)
    print("Tool teardown completed")
    
    # Bring executor down
    print("\nBringing executor down...")
    await executor.down()
    print(f"Executor state after down(): {executor.state}")
    print(f"Is down: {executor.is_down}")


async def demo_lifecycle_context():
    """Demonstrate automatic lifecycle management with context manager."""
    
    print("\n=== Lifecycle Context Demo ===")
    
    # Create executors
    python_executor = PythonExecutor()
    
    # Use lifecycle context for automatic management
    print("Using lifecycle_context for automatic up/down management...")
    
    async with lifecycle_context(python_executor) as manager:
        print(f"Inside context - executor is up: {python_executor.is_up}")
        print(f"Manager reports all up: {manager.all_up}")
        
        # Setup and execute tool
        tool_spec = ToolSpec(
            name="add_one",
            description="Add 1 to a number",
            tool_type=ToolType.PYTHON,
            input_schema={
                "type": "object",
                "properties": {
                    "value": {"type": "number"}
                },
                "required": ["value"]
            },
            implementation_config={
                "function": lambda value: value + 1
            }
        )
        
        await python_executor.setup_tool(tool_spec)
        result = await python_executor.execute(
            tool_spec, 
            {"value": 42}, 
            ToolContext(session_id="context_demo")
        )
        print(f"Tool result: {result}")
    
    # After context, executor should be down
    print(f"After context - executor is down: {python_executor.is_down}")


async def demo_error_handling():
    """Demonstrate error handling in lifecycle management."""
    
    print("\n=== Error Handling Demo ===")
    
    # Create executor that will fail during setup
    class FailingExecutor(PythonExecutor):
        async def _up(self):
            raise RuntimeError("Simulated setup failure")
    
    executor = FailingExecutor()
    
    try:
        await executor.up()
    except RuntimeError as e:
        print(f"Expected error during up(): {e}")
        print(f"Executor state after failed up(): {executor.state}")
        print(f"Is error: {executor.is_error}")
    
    # Demonstrate that down() still works even in error state
    await executor.down()
    print(f"Executor state after down(): {executor.state}")


async def main():
    """Run all lifecycle demonstrations."""
    print("Executor Lifecycle Management Demonstration")
    print("=" * 50)
    
    await demo_executor_lifecycle()
    await demo_lifecycle_context()
    await demo_error_handling()
    
    print("\n" + "=" * 50)
    print("Lifecycle demo completed!")
    print("Key benefits of lifecycle management:")
    print("1. Consistent up/down pattern across all components")
    print("2. Automatic state tracking and validation")
    print("3. Proper error handling and cleanup")
    print("4. Context manager for automatic resource management")
    print("5. Clear separation between global (executor) and tool-specific setup")


if __name__ == "__main__":
    asyncio.run(main())