"""
Demonstration of the simplified tool system.

This shows how tools now own their own specifications and the registry
simply aggregates them, making the system much simpler.
"""

import asyncio
import json
from typing import Any
import pydantic as pyd

from codin.tool.base import Tool, ToolType, ExecutionMode, ToolContext
from codin.tool.registry import ToolRegistry
from codin.tool.unified_executor import UnifiedToolExecutor


# Example tool implementations
class CalculatorTool(Tool):
    """Simple calculator tool."""
    
    class CalculatorInput(pyd.BaseModel):
        a: float = pyd.Field(..., description="First number")
        b: float = pyd.Field(..., description="Second number")
        operation: str = pyd.Field(..., description="Operation: add, subtract, multiply, divide")
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform basic arithmetic operations",
            tool_type=ToolType.PYTHON,
            input_schema=self.CalculatorInput,
            execution_mode=ExecutionMode.ASYNC,
            timeout=5.0,
            metadata={
                'category': 'math',
                'tags': ['arithmetic', 'calculator'],
                'estimated_duration': 0.1,
                'requires_approval': False
            }
        )
    
    async def run(self, args: dict[str, Any], tool_context: ToolContext) -> dict[str, Any]:
        """Execute the calculator operation."""
        a = args['a']
        b = args['b']
        operation = args['operation']
        
        if operation == 'add':
            result = a + b
        elif operation == 'subtract':
            result = a - b
        elif operation == 'multiply':
            result = a * b
        elif operation == 'divide':
            if b == 0:
                raise ValueError("Division by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return {'result': result, 'operation': f"{a} {operation} {b} = {result}"}


class FileListTool(Tool):
    """Tool to list files in a directory."""
    
    class FileListInput(pyd.BaseModel):
        path: str = pyd.Field(".", description="Directory path to list")
        show_hidden: bool = pyd.Field(False, description="Show hidden files")
    
    def __init__(self):
        super().__init__(
            name="list_files",
            description="List files in a directory",
            tool_type=ToolType.SHELL,
            input_schema=self.FileListInput,
            execution_mode=ExecutionMode.ASYNC,
            timeout=10.0,
            metadata={
                'category': 'filesystem',
                'tags': ['files', 'directory'],
                'estimated_duration': 2.0,
                'requires_approval': False
            }
        )
    
    async def run(self, args: dict[str, Any], tool_context: ToolContext) -> dict[str, Any]:
        """List files in the specified directory."""
        import os
        from pathlib import Path
        
        path = args['path']
        show_hidden = args.get('show_hidden', False)
        
        try:
            dir_path = Path(path)
            if not dir_path.exists():
                return {'error': f"Directory does not exist: {path}"}
            
            files = []
            for item in dir_path.iterdir():
                if not show_hidden and item.name.startswith('.'):
                    continue
                
                files.append({
                    'name': item.name,
                    'type': 'directory' if item.is_dir() else 'file',
                    'size': item.stat().st_size if item.is_file() else None
                })
            
            return {
                'path': str(dir_path.resolve()),
                'files': files,
                'count': len(files)
            }
        
        except Exception as e:
            return {'error': str(e)}


class DangerousDeleteTool(Tool):
    """Dangerous tool that requires approval."""
    
    class DeleteInput(pyd.BaseModel):
        filepath: str = pyd.Field(..., description="Path to file to delete")
    
    def __init__(self):
        super().__init__(
            name="delete_file",
            description="Delete a file from the filesystem",
            tool_type=ToolType.SHELL,
            input_schema=self.DeleteInput,
            execution_mode=ExecutionMode.ASYNC,
            timeout=5.0,
            metadata={
                'category': 'filesystem',
                'tags': ['delete', 'dangerous'],
                'estimated_duration': 0.5,
                'requires_approval': True,
                'is_dangerous': True
            }
        )
    
    async def run(self, args: dict[str, Any], tool_context: ToolContext) -> dict[str, Any]:
        """Delete the specified file."""
        import os
        from pathlib import Path
        
        filepath = args['filepath']
        
        try:
            file_path = Path(filepath)
            if not file_path.exists():
                return {'error': f"File does not exist: {filepath}"}
            
            # In a real implementation, this would actually delete the file
            # For demo purposes, we'll just simulate it
            return {
                'message': f"File {filepath} would be deleted",
                'filepath': str(file_path.resolve()),
                'simulated': True
            }
        
        except Exception as e:
            return {'error': str(e)}


async def demo_simplified_system():
    """Demonstrate the simplified tool system."""
    
    print("=== Simplified Tool System Demo ===\n")
    
    # Create tools
    calculator = CalculatorTool()
    file_lister = FileListTool()
    file_deleter = DangerousDeleteTool()
    
    # Create registry and register tools
    registry = ToolRegistry()
    registry.register_tool(calculator)
    registry.register_tool(file_lister)
    registry.register_tool(file_deleter)
    
    print("1. Registry aggregates specs from tools:")
    specs = registry.list_specs()
    for spec in specs:
        print(f"   - {spec.name}: {spec.description}")
        print(f"     Type: {spec.tool_type.value}, Category: {spec.metadata.category}")
        print(f"     Dangerous: {spec.metadata.is_dangerous}, Approval: {spec.metadata.requires_approval}")
        print()
    
    print("2. Registry provides OpenAI tool format:")
    openai_tools = registry.to_openai_tools()
    print(json.dumps(openai_tools[0], indent=2))
    print()
    
    print("3. Registry provides MCP tool format:")
    mcp_tools = registry.to_mcp_tools()
    print(json.dumps(mcp_tools[0], indent=2))
    print()
    
    print("4. Filter specs by tool type:")
    python_specs = registry.list_specs(tool_type="python")
    shell_specs = registry.list_specs(tool_type="shell")
    print(f"   Python tools: {[s.name for s in python_specs]}")
    print(f"   Shell tools: {[s.name for s in shell_specs]}")
    print()
    
    # Demonstrate execution (simplified - without actual executors)
    print("5. Execute tools directly (bypassing executors for demo):")
    
    # Calculator
    calc_result = await calculator.run(
        {'a': 10, 'b': 5, 'operation': 'multiply'},
        ToolContext(session_id="demo")
    )
    print(f"   Calculator: {calc_result}")
    
    # File lister
    files_result = await file_lister.run(
        {'path': '.', 'show_hidden': False},
        ToolContext(session_id="demo")
    )
    print(f"   Files (count): {files_result.get('count', 'error')}")
    
    # Dangerous operation
    delete_result = await file_deleter.run(
        {'filepath': '/tmp/fake_file.txt'},
        ToolContext(session_id="demo")
    )
    print(f"   Delete: {delete_result.get('message', delete_result.get('error'))}")
    
    print("\n=== Benefits of Simplified Design ===")
    print("✓ Tools own their specifications")
    print("✓ Registry just aggregates specs")
    print("✓ Single source of truth per tool")
    print("✓ No complex spec/implementation separation")
    print("✓ Backward compatible with existing Tool class")
    print("✓ Easy to understand and maintain")


async def demo_spec_consistency():
    """Demonstrate that tool specs are consistent across different formats."""
    
    print("\n=== Spec Consistency Demo ===\n")
    
    calculator = CalculatorTool()
    spec = calculator.get_spec()
    
    print("Tool specification:")
    print(f"Name: {spec.name}")
    print(f"Description: {spec.description}")
    print(f"Tool Type: {spec.tool_type}")
    print(f"Input Schema: {json.dumps(spec.input_schema, indent=2)}")
    print()
    
    print("OpenAI format (from tool):")
    openai_direct = calculator.to_openai_schema()
    print(json.dumps(openai_direct, indent=2))
    print()
    
    print("OpenAI format (from spec):")
    openai_from_spec = spec.to_openai_schema()
    print(json.dumps(openai_from_spec, indent=2))
    print()
    
    print("Formats are consistent:", openai_direct == openai_from_spec)


if __name__ == "__main__":
    asyncio.run(demo_simplified_system())
    asyncio.run(demo_spec_consistency())