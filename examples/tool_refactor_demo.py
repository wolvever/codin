"""
Demonstration of the refactored tool system architecture.

This example shows how the new design separates tool specifications from implementations
and provides a unified execution interface with extension support.
"""

import asyncio
import json
from pathlib import Path

# Import the new tool system components
from codin.tool.specs.base import ToolSpec, ToolType, ExecutionMode, ToolMetadata, ToolSpecRegistry
from codin.tool.executors.base import ExecutorRegistry
from codin.tool.extensions.base import ExtensionManager
from codin.tool.extensions.logging import LoggingExtension
from codin.tool.extensions.approval import ApprovalExtension, ApprovalMode
from codin.tool.extensions.metrics import MetricsExtension
from codin.tool.extensions.auth import AuthExtension, AuthPolicy
from codin.tool.unified_executor import UnifiedToolExecutor
from codin.tool.base import ToolContext


async def create_sample_tool_specs() -> list[ToolSpec]:
    """Create sample tool specifications."""
    
    # 1. Python function tool
    python_tool = ToolSpec(
        name="calculate_sum",
        description="Calculate the sum of two numbers",
        tool_type=ToolType.PYTHON,
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "result": {"type": "number", "description": "Sum of a and b"}
            }
        },
        execution_mode=ExecutionMode.ASYNC,
        implementation_config={
            "function": lambda a, b: {"result": a + b}
        },
        metadata=ToolMetadata(
            category="math",
            tags=["calculator", "arithmetic"],
            estimated_duration=0.1
        )
    )
    
    # 2. Shell command tool  
    shell_tool = ToolSpec(
        name="list_files",
        description="List files in a directory",
        tool_type=ToolType.SHELL,
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path", "default": "."},
                "show_hidden": {"type": "boolean", "description": "Show hidden files", "default": False}
            }
        },
        execution_mode=ExecutionMode.ASYNC,
        timeout=10.0,
        implementation_config={
            "command": "ls {'-la' if show_hidden else '-l'} {path}",
            "sandbox_type": "local"
        },
        metadata=ToolMetadata(
            category="filesystem",
            tags=["files", "directory"],
            requires_approval=False,
            estimated_duration=1.0
        )
    )
    
    # 3. Dangerous tool requiring approval
    dangerous_tool = ToolSpec(
        name="delete_file",
        description="Delete a file from the filesystem",
        tool_type=ToolType.SHELL,
        input_schema={
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to file to delete"}
            },
            "required": ["filepath"]
        },
        execution_mode=ExecutionMode.ASYNC,
        implementation_config={
            "command": "rm {filepath}",
            "sandbox_type": "local"
        },
        metadata=ToolMetadata(
            category="filesystem",
            tags=["delete", "dangerous"],
            requires_approval=True,
            is_dangerous=True,
            estimated_duration=0.5
        )
    )
    
    # 4. MCP tool (placeholder - would connect to real MCP server)
    mcp_tool = ToolSpec(
        name="mcp_file_read",
        description="Read file contents via MCP server",
        tool_type=ToolType.MCP,
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"]
        },
        execution_mode=ExecutionMode.ASYNC,
        implementation_config={
            "server_command": ["python", "-m", "mcp_file_server"],
            "server_args": []
        },
        metadata=ToolMetadata(
            category="files",
            tags=["mcp", "read"],
            estimated_duration=2.0
        )
    )
    
    return [python_tool, shell_tool, dangerous_tool, mcp_tool]


async def demo_agent_planner_usage():
    """Demonstrate how Agents and Planners would use tool specifications."""
    
    print("=== Agent/Planner Usage Demo ===")
    
    # Create spec registry (this is what agents/planners see)
    spec_registry = ToolSpecRegistry()
    
    # Load tool specifications
    specs = await create_sample_tool_specs()
    for spec in specs:
        spec_registry.register_spec(spec)
    
    # Agent/Planner can now work with tool specifications without knowing implementations
    print("\nAvailable tools for planning:")
    for spec in spec_registry.list_specs():
        print(f"  - {spec.name}: {spec.description}")
        print(f"    Type: {spec.tool_type.value}, Mode: {spec.execution_mode.value}")
        print(f"    Category: {spec.metadata.category}, Tags: {spec.metadata.tags}")
        print(f"    Est. duration: {spec.metadata.estimated_duration}s")
        print()
    
    # Convert to different formats for LLM function calling
    print("OpenAI function format:")
    openai_tools = spec_registry.to_openai_tools()
    print(json.dumps(openai_tools[0], indent=2))
    
    print("\nMCP format:")
    mcp_tools = spec_registry.to_mcp_tools()
    print(json.dumps(mcp_tools[0], indent=2))


async def demo_unified_executor():
    """Demonstrate the unified executor with extensions."""
    
    print("\n=== Unified Executor Demo ===")
    
    # Setup registries
    spec_registry = ToolSpecRegistry()
    specs = await create_sample_tool_specs()
    for spec in specs:
        spec_registry.register_spec(spec)
    
    # Create extension manager with all extensions
    extension_manager = ExtensionManager()
    
    # Add extensions in priority order
    extension_manager.register(AuthExtension(priority=1000))
    extension_manager.register(ApprovalExtension(
        approval_mode=ApprovalMode.UNSAFE_ONLY,
        priority=750
    ))
    extension_manager.register(LoggingExtension(priority=500))
    extension_manager.register(MetricsExtension(priority=250))
    
    # Create unified executor
    executor = UnifiedToolExecutor(
        spec_registry=spec_registry,
        extension_manager=extension_manager,
        max_concurrency=5
    )
    
    # Demo 1: Simple calculation (no approval needed)
    print("\n1. Executing safe calculation tool:")
    try:
        result = await executor.execute(
            "calculate_sum",
            {"a": 5, "b": 3},
            ToolContext(session_id="demo_session")
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Demo 2: List files (shell command)
    print("\n2. Executing shell command tool:")
    try:
        result = await executor.execute(
            "list_files", 
            {"path": ".", "show_hidden": False},
            ToolContext(session_id="demo_session")
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Demo 3: Dangerous operation (would require approval)
    print("\n3. Attempting dangerous operation (delete file):")
    try:
        # This would normally prompt for approval
        result = await executor.execute(
            "delete_file",
            {"filepath": "/tmp/nonexistent_file.txt"},
            ToolContext(session_id="demo_session")
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error (expected): {e}")


async def demo_custom_extensions():
    """Demonstrate creating custom extensions."""
    
    print("\n=== Custom Extension Demo ===")
    
    from codin.tool.extensions.base import Extension, ExtensionContext, ExtensionPriority
    from codin.tool.executors.base import ExecutionResult
    
    class RateLimitExtension(Extension):
        """Custom extension that implements rate limiting."""
        
        def __init__(self, max_calls_per_minute: int = 60, **kwargs):
            super().__init__(**kwargs)
            self.max_calls = max_calls_per_minute
            self.call_history: dict[str, list[float]] = {}
        
        @property
        def name(self) -> str:
            return "rate_limit"
        
        async def before_execute(self, ctx: ExtensionContext) -> None:
            import time
            
            tool_name = ctx.spec.name
            current_time = time.time()
            
            # Clean old entries
            if tool_name in self.call_history:
                self.call_history[tool_name] = [
                    t for t in self.call_history[tool_name] 
                    if current_time - t < 60  # Last minute
                ]
            else:
                self.call_history[tool_name] = []
            
            # Check rate limit
            if len(self.call_history[tool_name]) >= self.max_calls:
                raise Exception(f"Rate limit exceeded for tool {tool_name}")
            
            # Record this call
            self.call_history[tool_name].append(current_time)
    
    class AuditExtension(Extension):
        """Custom extension that logs all tool executions to an audit trail."""
        
        @property
        def name(self) -> str:
            return "audit"
        
        async def before_execute(self, ctx: ExtensionContext) -> None:
            print(f"AUDIT: Starting execution of {ctx.spec.name}")
        
        async def after_execute(self, ctx: ExtensionContext, result: ExecutionResult) -> ExecutionResult:
            print(f"AUDIT: Completed {ctx.spec.name} with status {result.status.value}")
            return result
    
    # Create executor with custom extensions
    spec_registry = ToolSpecRegistry()
    specs = await create_sample_tool_specs()
    for spec in specs:
        spec_registry.register_spec(spec)
    
    extension_manager = ExtensionManager()
    extension_manager.register(RateLimitExtension(max_calls_per_minute=3, priority=ExtensionPriority.HIGH))
    extension_manager.register(AuditExtension(priority=ExtensionPriority.NORMAL))
    
    executor = UnifiedToolExecutor(
        spec_registry=spec_registry,
        extension_manager=extension_manager
    )
    
    # Test rate limiting
    print("Testing rate limiting (max 3 calls per minute):")
    for i in range(5):
        try:
            result = await executor.execute(
                "calculate_sum",
                {"a": i, "b": 1},
                ToolContext(session_id="rate_test")
            )
            print(f"Call {i+1}: Success - {result}")
        except Exception as e:
            print(f"Call {i+1}: Failed - {e}")


async def demo_configuration_files():
    """Demonstrate loading tool specs from configuration files."""
    
    print("\n=== Configuration File Demo ===")
    
    # Create sample configuration
    config = {
        "tools": [
            {
                "name": "weather_check",
                "description": "Check weather for a location",
                "tool_type": "http",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                },
                "implementation_config": {
                    "url": "https://api.weather.com/v1/weather",
                    "method": "GET",
                    "headers": {"API-Key": "your-api-key"}
                },
                "metadata": {
                    "category": "external_api",
                    "tags": ["weather", "api"],
                    "estimated_duration": 3.0
                }
            }
        ]
    }
    
    # Save to file
    config_path = Path("/tmp/tool_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Tool configuration saved to: {config_path}")
    print("Configuration content:")
    print(json.dumps(config, indent=2))
    
    # In practice, you would load this with:
    # spec_registry = await ToolSpecRegistry.from_config(config_path)


async def main():
    """Run all demonstrations."""
    print("Tool System Refactor Demonstration")
    print("=" * 50)
    
    await demo_agent_planner_usage()
    await demo_unified_executor()
    await demo_custom_extensions()
    await demo_configuration_files()
    
    print("\n" + "=" * 50)
    print("Demo completed! Key benefits of the refactor:")
    print("1. Clear separation of tool specs from implementations")
    print("2. Unified execution interface for all tool types")
    print("3. Extensible middleware system for cross-cutting concerns")
    print("4. Easy development and testing workflow")
    print("5. Agent/Planner decoupling from execution details")


if __name__ == "__main__":
    asyncio.run(main())