# Tool System Design

## Overview

The Tool System provides a plugin architecture for integrating external functionality into the CoDIN platform. It supports multiple tool backends including filesystem tools, HTTP services, and MCP (Model Context Protocol) servers, with unified discovery, validation, and execution capabilities.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Tool System                                │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Tool (ABC)    │    │   Toolset       │    │ ToolRegistry    │ │
│  │                 │    │ (Collection)    │    │ (Discovery)     │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│           │                       │                       │         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ ToolExecutor    │    │  ToolContext    │    │  ToolExtension  │ │
│  │ (Execution)     │    │  (Runtime)      │    │  (Middleware)   │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│           │                       │                       │         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   ToolSpec      │    │   MCPClient     │    │ SchemaConverter │ │
│  │ (Definition)    │    │ (Protocol)      │    │ (Formats)       │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Interfaces

### Tool Protocol

```python
class Tool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, spec: ToolSpec):
        self.spec = spec
    
    @property
    def name(self) -> str:
        """Tool name from specification."""
        return self.spec.get_name()
    
    @abstractmethod
    async def execute(self, context: ToolContext, **kwargs) -> Any:
        """Execute tool with given parameters."""
        pass
    
    def validate_input(self, parameters: Dict[str, Any]) -> bool:
        """Validate input parameters against tool specification."""
        return self.spec.validate_parameters(parameters)
    
    def get_spec(self) -> ToolSpec:
        """Get tool specification."""
        return self.spec
    
    def to_definition(self) -> Dict[str, Any]:
        """Convert to tool definition format."""
        return {
            "name": self.name,
            "description": self.spec.get_description(),
            "parameters": self.spec.get_parameters()
        }
    
    def to_mcp_schema(self) -> Dict[str, Any]:
        """Convert to MCP schema format."""
        return {
            "name": self.name,
            "description": self.spec.get_description(),
            "inputSchema": self.spec.get_parameters()
        }
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.spec.get_description(),
                "parameters": self.spec.get_parameters()
            }
        }
```

### ToolSpec Interface

```python
class ToolSpec(ABC):
    """Abstract specification for tool definition."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get tool name."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get tool description."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema (JSON Schema format)."""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters against schema."""
        try:
            jsonschema.validate(parameters, self.get_parameters())
            return True
        except jsonschema.ValidationError:
            return False
```

## Implementation Details

### Toolset

Container for managing related tools:

```python
class Toolset:
    def __init__(self, tools: List[Tool] = None):
        self._tools: Dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.add(tool)
    
    def add(self, tool: Tool) -> None:
        """Add tool to toolset."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """List all tools in toolset."""
        return list(self._tools.values())
    
    def to_definitions(self) -> List[Dict[str, Any]]:
        """Convert all tools to definition format."""
        return [tool.to_definition() for tool in self._tools.values()]
    
    def to_mcp_schemas(self) -> List[Dict[str, Any]]:
        """Convert all tools to MCP schema format."""
        return [tool.to_mcp_schema() for tool in self._tools.values()]
    
    def to_openai_schemas(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI function format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]
    
    def __contains__(self, name: str) -> bool:
        """Check if tool exists in toolset."""
        return name in self._tools
    
    def __len__(self) -> int:
        """Get number of tools in toolset."""
        return len(self._tools)
    
    def __iter__(self):
        """Iterate over tools."""
        return iter(self._tools.values())
```

### ToolRegistry

Centralized tool discovery and management:

```python
class ToolRegistry:
    def __init__(self):
        self._toolsets: Dict[str, Toolset] = {}
        self._endpoint_configs: Dict[str, EndpointConfig] = {}
        self._tool_cache: Dict[str, Tool] = {}
    
    async def from_config(self, config: Dict[str, Any]) -> None:
        """Load tools from configuration."""
        for toolset_name, toolset_config in config.get("toolsets", {}).items():
            endpoint = toolset_config.get("endpoint")
            if endpoint:
                await self.from_endpoint(toolset_name, endpoint)
    
    async def from_endpoint(self, name: str, endpoint: str) -> None:
        """Load tools from endpoint URL."""
        if endpoint.startswith("fs://"):
            await self._load_from_filesystem(name, endpoint)
        elif endpoint.startswith("http://") or endpoint.startswith("https://"):
            await self._load_from_http(name, endpoint)
        elif endpoint.startswith("mcp://"):
            await self._load_from_mcp(name, endpoint)
        else:
            raise ValueError(f"Unsupported endpoint scheme: {endpoint}")
    
    def register_toolset(self, name: str, toolset: Toolset) -> None:
        """Register toolset with registry."""
        self._toolsets[name] = toolset
        
        # Update tool cache
        for tool in toolset.list_tools():
            self._tool_cache[tool.name] = tool
    
    def register_tool(self, tool: Tool, toolset_name: str = "default") -> None:
        """Register individual tool."""
        if toolset_name not in self._toolsets:
            self._toolsets[toolset_name] = Toolset()
        
        self._toolsets[toolset_name].add(tool)
        self._tool_cache[tool.name] = tool
    
    def get_toolset(self, name: str) -> Optional[Toolset]:
        """Get toolset by name."""
        return self._toolsets.get(name)
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._tool_cache.get(name)
    
    def get_tools(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tool_cache.values())
    
    def get_toolsets(self) -> Dict[str, Toolset]:
        """Get all registered toolsets."""
        return self._toolsets.copy()
    
    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI format."""
        tools = []
        for toolset in self._toolsets.values():
            tools.extend(toolset.to_openai_schemas())
        return tools
    
    def to_mcp_tools(self) -> List[Dict[str, Any]]:
        """Convert all tools to MCP format."""
        tools = []
        for toolset in self._toolsets.values():
            tools.extend(toolset.to_mcp_schemas())
        return tools
    
    def get_tools_with_executor(self) -> Tuple[Toolset, ToolExecutor]:
        """Get combined toolset and executor."""
        combined_toolset = Toolset()
        for toolset in self._toolsets.values():
            for tool in toolset.list_tools():
                combined_toolset.add(tool)
        
        executor = ToolExecutor(registry=self)
        return combined_toolset, executor
```

### ToolExecutor

Executes tools with extension support:

```python
class ToolExecutor:
    def __init__(
        self, 
        registry: ToolRegistry,
        extensions: List[ToolExtension] = None
    ):
        self.registry = registry
        self.extensions = extensions or []
    
    async def execute(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        context: ToolContext
    ) -> ToolCallResult:
        """Execute tool with extensions pipeline."""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return ToolCallResult(
                call_id=context.call_id,
                result=None,
                error=f"Tool not found: {tool_name}"
            )
        
        try:
            # Pre-execution extensions
            for extension in self.extensions:
                if hasattr(extension, 'before_execute'):
                    parameters = await extension.before_execute(
                        tool, parameters, context
                    )
            
            # Validate parameters
            if not tool.validate_input(parameters):
                return ToolCallResult(
                    call_id=context.call_id,
                    result=None,
                    error="Invalid parameters"
                )
            
            # Execute tool
            start_time = time.time()
            result = await tool.execute(context, **parameters)
            execution_time = time.time() - start_time
            
            # Post-execution extensions
            for extension in self.extensions:
                if hasattr(extension, 'after_execute'):
                    result = await extension.after_execute(
                        tool, result, context, execution_time
                    )
            
            return ToolCallResult(
                call_id=context.call_id,
                result=result,
                error=None
            )
            
        except Exception as e:
            # Error handling extensions
            for extension in self.extensions:
                if hasattr(extension, 'on_error'):
                    await extension.on_error(tool, e, context)
            
            return ToolCallResult(
                call_id=context.call_id,
                result=None,
                error=str(e)
            )
```

## Tool Extensions

### Extension Interface

```python
class ToolExtension(ABC):
    """Base class for tool execution extensions."""
    
    async def before_execute(
        self, 
        tool: Tool, 
        parameters: Dict[str, Any], 
        context: ToolContext
    ) -> Dict[str, Any]:
        """Called before tool execution."""
        return parameters
    
    async def after_execute(
        self, 
        tool: Tool, 
        result: Any, 
        context: ToolContext,
        execution_time: float
    ) -> Any:
        """Called after successful tool execution."""
        return result
    
    async def on_error(
        self, 
        tool: Tool, 
        error: Exception, 
        context: ToolContext
    ) -> None:
        """Called when tool execution fails."""
        pass
```

### Built-in Extensions

#### Approval Extension

```python
class ApprovalExtension(ToolExtension):
    def __init__(self, approval_mode: ApprovalMode = ApprovalMode.NONE):
        self.approval_mode = approval_mode
        self.approval_callback: Optional[Callable] = None
    
    async def before_execute(
        self, 
        tool: Tool, 
        parameters: Dict[str, Any], 
        context: ToolContext
    ) -> Dict[str, Any]:
        """Request approval before execution if required."""
        if self.approval_mode == ApprovalMode.ALL:
            await self._request_approval(tool, parameters, context)
        elif self.approval_mode == ApprovalMode.DANGEROUS:
            if self._is_dangerous_tool(tool):
                await self._request_approval(tool, parameters, context)
        
        return parameters
    
    async def _request_approval(
        self, 
        tool: Tool, 
        parameters: Dict[str, Any], 
        context: ToolContext
    ) -> None:
        """Request user approval for tool execution."""
        if self.approval_callback:
            approved = await self.approval_callback(tool, parameters, context)
            if not approved:
                raise ToolExecutionDeniedError("Tool execution denied by user")
        else:
            # Default: require explicit approval
            raise ToolExecutionDeniedError("Approval required but no callback provided")
```

#### Logging Extension

```python
class LoggingExtension(ToolExtension):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    async def before_execute(
        self, 
        tool: Tool, 
        parameters: Dict[str, Any], 
        context: ToolContext
    ) -> Dict[str, Any]:
        """Log tool execution start."""
        self.logger.info(
            f"Executing tool: {tool.name}",
            extra={
                "tool_name": tool.name,
                "parameters": parameters,
                "context": context.to_dict()
            }
        )
        return parameters
    
    async def after_execute(
        self, 
        tool: Tool, 
        result: Any, 
        context: ToolContext,
        execution_time: float
    ) -> Any:
        """Log successful tool execution."""
        self.logger.info(
            f"Tool executed successfully: {tool.name}",
            extra={
                "tool_name": tool.name,
                "execution_time": execution_time,
                "result_type": type(result).__name__
            }
        )
        return result
    
    async def on_error(
        self, 
        tool: Tool, 
        error: Exception, 
        context: ToolContext
    ) -> None:
        """Log tool execution error."""
        self.logger.error(
            f"Tool execution failed: {tool.name}",
            extra={
                "tool_name": tool.name,
                "error": str(error),
                "error_type": type(error).__name__
            },
            exc_info=True
        )
```

#### Metrics Extension

```python
class MetricsExtension(ToolExtension):
    def __init__(self):
        self.metrics = {
            "executions": defaultdict(int),
            "execution_times": defaultdict(list),
            "errors": defaultdict(int)
        }
    
    async def after_execute(
        self, 
        tool: Tool, 
        result: Any, 
        context: ToolContext,
        execution_time: float
    ) -> Any:
        """Record execution metrics."""
        self.metrics["executions"][tool.name] += 1
        self.metrics["execution_times"][tool.name].append(execution_time)
        return result
    
    async def on_error(
        self, 
        tool: Tool, 
        error: Exception, 
        context: ToolContext
    ) -> None:
        """Record error metrics."""
        self.metrics["errors"][tool.name] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return {
            "total_executions": sum(self.metrics["executions"].values()),
            "tool_executions": dict(self.metrics["executions"]),
            "average_execution_times": {
                tool: sum(times) / len(times)
                for tool, times in self.metrics["execution_times"].items()
                if times
            },
            "error_rates": {
                tool: self.metrics["errors"][tool] / max(1, self.metrics["executions"][tool])
                for tool in self.metrics["executions"]
            }
        }
```

## MCP Integration

### MCP Client

```python
class MCPClient:
    def __init__(self, server_url: str, session_id: str = None):
        self.server_url = server_url
        self.session_id = session_id or str(uuid.uuid4())
        self.client = httpx.AsyncClient()
        self.tools_cache: Dict[str, MCPTool] = {}
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from MCP server."""
        response = await self.client.post(
            f"{self.server_url}/tools/list",
            json={"method": "tools/list", "params": {}}
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("result", {}).get("tools", [])
        else:
            raise MCPError(f"Failed to list tools: {response.status_code}")
    
    async def call_tool(
        self, 
        name: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call tool on MCP server."""
        payload = {
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        
        response = await self.client.post(
            f"{self.server_url}/tools/call",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("result", {})
        else:
            raise MCPError(f"Tool call failed: {response.status_code}")
    
    async def get_tool_schema(self, name: str) -> Dict[str, Any]:
        """Get schema for specific tool."""
        tools = await self.list_tools()
        for tool in tools:
            if tool.get("name") == name:
                return tool
        
        raise MCPError(f"Tool not found: {name}")
```

### MCP Tool Wrapper

```python
class MCPTool(Tool):
    def __init__(self, mcp_client: MCPClient, tool_schema: Dict[str, Any]):
        self.mcp_client = mcp_client
        self.tool_schema = tool_schema
        
        # Create spec from MCP schema
        spec = MCPToolSpec(tool_schema)
        super().__init__(spec)
    
    async def execute(self, context: ToolContext, **kwargs) -> Any:
        """Execute tool via MCP client."""
        try:
            result = await self.mcp_client.call_tool(self.name, kwargs)
            return result.get("content", result)
        except Exception as e:
            raise ToolExecutionError(f"MCP tool execution failed: {str(e)}")

class MCPToolSpec(ToolSpec):
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
    
    def get_name(self) -> str:
        return self.schema["name"]
    
    def get_description(self) -> str:
        return self.schema.get("description", "")
    
    def get_parameters(self) -> Dict[str, Any]:
        return self.schema.get("inputSchema", {})
```

## Built-in Tools

### File Operations

```python
class ReadFileTool(Tool):
    def __init__(self, sandbox: Sandbox = None):
        self.sandbox = sandbox
        spec = FunctionToolSpec(
            name="read_file",
            description="Read contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        )
        super().__init__(spec)
    
    async def execute(self, context: ToolContext, **kwargs) -> str:
        """Read file contents."""
        path = kwargs["path"]
        
        if self.sandbox:
            return await self.sandbox.read_file(path)
        else:
            # Direct file system access (less secure)
            async with aiofiles.open(path, 'r') as f:
                return await f.read()

class WriteFileTool(Tool):
    def __init__(self, sandbox: Sandbox = None):
        self.sandbox = sandbox
        spec = FunctionToolSpec(
            name="write_file",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        )
        super().__init__(spec)
    
    async def execute(self, context: ToolContext, **kwargs) -> str:
        """Write content to file."""
        path = kwargs["path"]
        content = kwargs["content"]
        
        if self.sandbox:
            await self.sandbox.write_file(path, content)
        else:
            async with aiofiles.open(path, 'w') as f:
                await f.write(content)
        
        return f"File written successfully: {path}"
```

### Shell Commands

```python
class RunCommandTool(Tool):
    def __init__(self, sandbox: Sandbox = None):
        self.sandbox = sandbox
        spec = FunctionToolSpec(
            name="run_command",
            description="Execute shell command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds", "default": 30}
                },
                "required": ["command"]
            }
        )
        super().__init__(spec)
    
    async def execute(self, context: ToolContext, **kwargs) -> Dict[str, Any]:
        """Execute shell command."""
        command = kwargs["command"]
        timeout = kwargs.get("timeout", 30)
        
        if self.sandbox:
            request = SandboxRequest(
                code=command,
                language="bash",
                timeout=timeout
            )
            result = await self.sandbox.run(request)
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "execution_time": result.execution_time
            }
        else:
            # Direct execution (less secure)
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                return {
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "exit_code": process.returncode
                }
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ToolExecutionError("Command timed out")
```

## Configuration

### Tool Configuration

```python
@dataclass
class ToolConfig:
    toolsets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extensions: List[str] = field(default_factory=list)
    approval_mode: str = "none"
    timeout: float = 30.0
    max_retries: int = 3
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolConfig":
        """Create configuration from dictionary."""
        return cls(
            toolsets=data.get("toolsets", {}),
            extensions=data.get("extensions", []),
            approval_mode=data.get("approval_mode", "none"),
            timeout=data.get("timeout", 30.0),
            max_retries=data.get("max_retries", 3)
        )
```

## Error Handling

### Exception Types

```python
class ToolError(Exception):
    """Base exception for tool errors."""
    pass

class ToolNotFoundError(ToolError):
    """Raised when tool is not found."""
    pass

class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""
    pass

class ToolExecutionDeniedError(ToolError):
    """Raised when tool execution is denied."""
    pass

class ToolValidationError(ToolError):
    """Raised when tool parameter validation fails."""
    pass

class MCPError(ToolError):
    """Raised when MCP operation fails."""
    pass
```

## Performance Optimizations

### Caching

- Cache tool schemas and definitions
- Cache MCP server connections
- Cache execution results for idempotent tools

### Connection Pooling

- Reuse HTTP connections for MCP servers
- Pool sandbox instances for tool execution
- Maintain persistent connections where possible

### Lazy Loading

- Load tools on demand from endpoints
- Defer expensive validation until execution
- Lazy initialization of tool extensions

## Usage Examples

### Example 1: Development Tool Suite

```python
import asyncio
import json
from typing import Any, Dict
from pathlib import Path
import pydantic as pyd

from codin.tool.base import Tool, ToolType, ExecutionMode, ToolContext
from codin.tool.registry import ToolRegistry
from codin.sandbox.local import LocalSandbox

# Custom development tools using the simplified system
class CodeFormatterTool(Tool):
    """Tool for formatting code using black, prettier, etc."""
    
    class FormatterInput(pyd.BaseModel):
        file_path: str = pyd.Field(..., description="Path to file to format")
        language: str = pyd.Field(..., description="Programming language (python, javascript, etc.)")
        config_path: str = pyd.Field("", description="Optional path to formatter config")
    
    def __init__(self, sandbox: LocalSandbox):
        self.sandbox = sandbox
        super().__init__(
            name="format_code",
            description="Format code files using appropriate formatters",
            tool_type=ToolType.SHELL,
            input_schema=self.FormatterInput,
            execution_mode=ExecutionMode.ASYNC,
            timeout=30.0,
            metadata={
                'category': 'development',
                'tags': ['formatting', 'code-quality'],
                'estimated_duration': 5.0,
                'requires_approval': False
            }
        )
    
    async def run(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
        """Format code using language-specific formatters."""
        file_path = args['file_path']
        language = args['language'].lower()
        config_path = args.get('config_path', '')
        
        try:
            # Check if file exists
            if not Path(file_path).exists():
                return {'error': f'File not found: {file_path}'}
            
            # Choose formatter based on language
            if language == 'python':
                cmd = f"black {file_path}"
                if config_path:
                    cmd += f" --config {config_path}"
            elif language == 'javascript' or language == 'typescript':
                cmd = f"prettier --write {file_path}"
                if config_path:
                    cmd += f" --config {config_path}"
            elif language == 'rust':
                cmd = f"rustfmt {file_path}"
            else:
                return {'error': f'Unsupported language: {language}'}
            
            # Execute formatter via sandbox
            result = await self.sandbox.run_command(cmd)
            
            return {
                'file_path': file_path,
                'language': language,
                'formatted': result.exit_code == 0,
                'output': result.stdout,
                'errors': result.stderr
            }
            
        except Exception as e:
            return {'error': str(e)}

class CodeLinterTool(Tool):
    """Tool for linting code using various linters."""
    
    class LinterInput(pyd.BaseModel):
        file_path: str = pyd.Field(..., description="Path to file to lint")
        language: str = pyd.Field(..., description="Programming language")
        fix_issues: bool = pyd.Field(False, description="Automatically fix issues where possible")
    
    def __init__(self, sandbox: LocalSandbox):
        self.sandbox = sandbox
        super().__init__(
            name="lint_code",
            description="Lint code files for style and quality issues",
            tool_type=ToolType.SHELL,
            input_schema=self.LinterInput,
            execution_mode=ExecutionMode.ASYNC,
            timeout=60.0,
            metadata={
                'category': 'development',
                'tags': ['linting', 'code-quality'],
                'estimated_duration': 10.0,
                'requires_approval': False
            }
        )
    
    async def run(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
        """Lint code using language-specific linters."""
        file_path = args['file_path']
        language = args['language'].lower()
        fix_issues = args.get('fix_issues', False)
        
        try:
            if not Path(file_path).exists():
                return {'error': f'File not found: {file_path}'}
            
            # Choose linter based on language
            if language == 'python':
                cmd = f"flake8 {file_path}"
                if fix_issues:
                    # Use autopep8 for fixes
                    fix_cmd = f"autopep8 --in-place {file_path}"
                    await self.sandbox.run_command(fix_cmd)
            elif language == 'javascript' or language == 'typescript':
                cmd = f"eslint {file_path}"
                if fix_issues:
                    cmd += " --fix"
            elif language == 'rust':
                cmd = f"cargo clippy -- -D warnings"
            else:
                return {'error': f'Unsupported language: {language}'}
            
            # Execute linter
            result = await self.sandbox.run_command(cmd)
            
            # Parse linter output
            issues = self._parse_linter_output(result.stdout, language)
            
            return {
                'file_path': file_path,
                'language': language,
                'issues_found': len(issues),
                'issues': issues,
                'fixed': fix_issues and result.exit_code == 0,
                'output': result.stdout
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _parse_linter_output(self, output: str, language: str) -> list:
        """Parse linter output into structured issues."""
        # Simplified parsing - real implementation would be more sophisticated
        lines = output.strip().split('\n')
        issues = []
        
        for line in lines:
            if ':' in line and any(word in line.lower() for word in ['error', 'warning']):
                issues.append({
                    'line': line,
                    'severity': 'error' if 'error' in line.lower() else 'warning'
                })
        
        return issues

class TestRunnerTool(Tool):
    """Tool for running tests with various frameworks."""
    
    class TestInput(pyd.BaseModel):
        test_path: str = pyd.Field(".", description="Path to test files or directory")
        framework: str = pyd.Field("auto", description="Test framework (pytest, jest, cargo, auto)")
        coverage: bool = pyd.Field(True, description="Generate coverage report")
        verbose: bool = pyd.Field(False, description="Verbose output")
    
    def __init__(self, sandbox: LocalSandbox):
        self.sandbox = sandbox
        super().__init__(
            name="run_tests",
            description="Run tests using appropriate test framework",
            tool_type=ToolType.SHELL,
            input_schema=self.TestInput,
            execution_mode=ExecutionMode.ASYNC,
            timeout=300.0,
            metadata={
                'category': 'development',
                'tags': ['testing', 'quality-assurance'],
                'estimated_duration': 60.0,
                'requires_approval': False
            }
        )
    
    async def run(self, args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
        """Run tests using the appropriate framework."""
        test_path = args['test_path']
        framework = args['framework']
        coverage = args.get('coverage', True)
        verbose = args.get('verbose', False)
        
        try:
            # Auto-detect framework if needed
            if framework == "auto":
                framework = await self._detect_test_framework(test_path)
            
            # Build test command
            if framework == "pytest":
                cmd = f"pytest {test_path}"
                if coverage:
                    cmd += " --cov=."
                if verbose:
                    cmd += " -v"
            elif framework == "jest":
                cmd = f"npm test"
                if coverage:
                    cmd += " -- --coverage"
            elif framework == "cargo":
                cmd = f"cargo test"
                if verbose:
                    cmd += " --verbose"
            else:
                return {'error': f'Unsupported test framework: {framework}'}
            
            # Execute tests
            result = await self.sandbox.run_command(cmd)
            
            # Parse test results
            test_summary = self._parse_test_output(result.stdout, framework)
            
            return {
                'test_path': test_path,
                'framework': framework,
                'success': result.exit_code == 0,
                'summary': test_summary,
                'coverage_enabled': coverage,
                'output': result.stdout,
                'errors': result.stderr
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _detect_test_framework(self, test_path: str) -> str:
        """Auto-detect the appropriate test framework."""
        # Check for common test framework indicators
        if Path("pytest.ini").exists() or Path("setup.cfg").exists():
            return "pytest"
        elif Path("package.json").exists():
            return "jest"
        elif Path("Cargo.toml").exists():
            return "cargo"
        else:
            return "pytest"  # Default fallback
    
    def _parse_test_output(self, output: str, framework: str) -> dict:
        """Parse test output to extract summary information."""
        summary = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'duration': '0s'
        }
        
        # Framework-specific parsing
        if framework == "pytest":
            # Parse pytest output for test counts
            for line in output.split('\n'):
                if 'passed' in line and 'failed' in line:
                    # Extract numbers from lines like "5 passed, 2 failed in 1.23s"
                    import re
                    matches = re.findall(r'(\d+)\s+(passed|failed|skipped)', line)
                    for count, status in matches:
                        summary[status] = int(count)
                        summary['total_tests'] += int(count)
        
        return summary

# Set up development tool suite
async def setup_development_tools():
    """Set up a comprehensive development tool suite."""
    
    # Initialize sandbox
    sandbox = LocalSandbox()
    await sandbox.up()
    
    # Create tool registry
    registry = ToolRegistry()
    
    # Register development tools
    formatter = CodeFormatterTool(sandbox)
    linter = CodeLinterTool(sandbox)
    test_runner = TestRunnerTool(sandbox)
    
    registry.register_tool(formatter)
    registry.register_tool(linter)
    registry.register_tool(test_runner)
    
    print(f"✅ Registered {len(registry.list_tools())} development tools")
    
    # Display available tools
    print("\n📋 Available Tools:")
    for tool in registry.list_tools():
        spec = tool.get_spec()
        print(f"  • {spec.name}: {spec.description}")
        print(f"    Category: {spec.metadata.category}, Type: {spec.tool_type.value}")
    
    return registry, sandbox

# Usage example
async def run_development_workflow():
    """Demonstrate a complete development workflow using tools."""
    
    registry, sandbox = await setup_development_tools()
    
    try:
        # Create tool context
        context = ToolContext(
            session_id="dev_workflow",
            call_id="workflow_001"
        )
        
        # Example Python file for demonstration
        test_file = "example.py"
        
        # Create a sample Python file
        sample_code = '''
def hello_world( name ):
    print(f"Hello, {name}!")
    return f"Hello, {name}!"

def add_numbers(a,b):
    return a+b

# This is a comment
if __name__ == "__main__":
    hello_world("World")
    result = add_numbers(5, 3)
    print(f"5 + 3 = {result}")
'''
        
        # Write sample file
        with open(test_file, 'w') as f:
            f.write(sample_code)
        
        print(f"📝 Created sample file: {test_file}")
        
        # Step 1: Format the code
        print("\n🎨 Step 1: Formatting code...")
        formatter = registry.get_tool("format_code")
        format_result = await formatter.run({
            'file_path': test_file,
            'language': 'python'
        }, context)
        
        print(f"Format result: {format_result}")
        
        # Step 2: Lint the code
        print("\n🔍 Step 2: Linting code...")
        linter = registry.get_tool("lint_code")
        lint_result = await linter.run({
            'file_path': test_file,
            'language': 'python',
            'fix_issues': True
        }, context)
        
        print(f"Lint result: {lint_result}")
        
        # Step 3: Run tests (if test files exist)
        print("\n🧪 Step 3: Running tests...")
        test_runner = registry.get_tool("run_tests")
        test_result = await test_runner.run({
            'test_path': '.',
            'framework': 'pytest',
            'coverage': True,
            'verbose': True
        }, context)
        
        print(f"Test result: {test_result}")
        
        print("\n✅ Development workflow completed!")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
    
    finally:
        # Clean up
        if Path(test_file).exists():
            Path(test_file).unlink()
        await sandbox.down()

# Tool registry integration example
async def demonstrate_tool_formats():
    """Demonstrate how tools integrate with different formats."""
    
    registry, sandbox = await setup_development_tools()
    
    try:
        print("🔧 Tool Registry Integration Examples")
        print("=" * 50)
        
        # Get all tools in different formats
        tools = registry.list_tools()
        
        print(f"\n📊 Registry contains {len(tools)} tools")
        
        # OpenAI function calling format
        print("\n🤖 OpenAI Function Format:")
        openai_tools = registry.to_openai_tools()
        for tool in openai_tools[:1]:  # Show first tool as example
            print(json.dumps(tool, indent=2))
        
        # MCP format
        print("\n🔌 MCP Tool Format:")
        mcp_tools = registry.to_mcp_tools()
        for tool in mcp_tools[:1]:  # Show first tool as example
            print(json.dumps(tool, indent=2))
        
        # Filter tools by category
        print("\n📂 Tools by Category:")
        dev_tools = [t for t in tools if t.get_spec().metadata.category == 'development']
        print(f"Development tools: {[t.name for t in dev_tools]}")
        
        # Filter by tool type
        print("\n⚙️ Tools by Type:")
        shell_tools = [t for t in tools if t.get_spec().tool_type == ToolType.SHELL]
        print(f"Shell tools: {[t.name for t in shell_tools]}")
        
    finally:
        await sandbox.down()

if __name__ == "__main__":
    # Run both examples
    print("🚀 Starting Development Tool Examples")
    asyncio.run(run_development_workflow())
    print("\n" + "="*50)
    asyncio.run(demonstrate_tool_formats())
```

### Example 2: Sandbox Tool Integration

```python
from codin.tool import SandboxToolset
from codin.sandbox.local import LocalSandbox
from codin.tool.registry import ToolRegistry

async def setup_sandbox_tools():
    """Set up sandbox-integrated tools for secure code execution."""
    
    # Initialize sandbox
    sandbox = LocalSandbox()
    await sandbox.up()
    
    # Create sandbox toolset
    sandbox_toolset = SandboxToolset(sandbox)
    await sandbox_toolset.up()
    
    # Create registry and register toolset
    registry = ToolRegistry()
    registry.register_toolset(sandbox_toolset)
    
    print(f"🏗️ Sandbox toolset initialized with {len(registry.list_tools())} tools")
    
    return registry, sandbox

async def execute_code_safely():
    """Demonstrate safe code execution using sandbox tools."""
    
    registry, sandbox = await setup_sandbox_tools()
    
    try:
        context = ToolContext(session_id="sandbox_demo")
        
        # Get sandbox tools
        write_tool = registry.get_tool("write_file")
        read_tool = registry.get_tool("read_file") 
        run_tool = registry.get_tool("run_command")
        
        # Step 1: Write a Python script
        print("📝 Writing Python script...")
        await write_tool.run({
            'path': 'hello.py',
            'content': '''
import sys
import math

def calculate_fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    print("Fibonacci Calculator")
    print("=" * 20)
    
    for i in range(10):
        fib = calculate_fibonacci(i)
        print(f"F({i}) = {fib}")

if __name__ == "__main__":
    main()
'''
        }, context)
        
        # Step 2: Execute the script
        print("🚀 Executing Python script...")
        result = await run_tool.run({
            'command': 'python hello.py',
            'timeout': 30
        }, context)
        
        print(f"Execution result: {result}")
        
        # Step 3: Read output file (if created)
        print("📖 Reading script content...")
        content = await read_tool.run({
            'path': 'hello.py'
        }, context)
        
        print("Script content verified")
        
    finally:
        await sandbox.down()

if __name__ == "__main__":
    asyncio.run(execute_code_safely())
```

### Example 3: CLI Integration Example

```python
#!/usr/bin/env python3
"""
CLI tool demonstrating CoDIN tool system integration.

Usage:
    python cli_example.py format --file main.py --language python
    python cli_example.py lint --file main.py --language python --fix
    python cli_example.py test --path tests/ --framework pytest --coverage
"""

import asyncio
import click
from pathlib import Path

from codin.tool.registry import ToolRegistry
from codin.tool.base import ToolContext
from codin.sandbox.local import LocalSandbox

class CLIToolRunner:
    """CLI wrapper for CoDIN tools."""
    
    def __init__(self):
        self.registry = None
        self.sandbox = None
    
    async def initialize(self):
        """Initialize tools and sandbox."""
        self.sandbox = LocalSandbox()
        await self.sandbox.up()
        
        self.registry = ToolRegistry()
        
        # Register development tools (from previous example)
        from __main__ import CodeFormatterTool, CodeLinterTool, TestRunnerTool
        
        formatter = CodeFormatterTool(self.sandbox)
        linter = CodeLinterTool(self.sandbox)
        test_runner = TestRunnerTool(self.sandbox)
        
        self.registry.register_tool(formatter)
        self.registry.register_tool(linter)
        self.registry.register_tool(test_runner)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.sandbox:
            await self.sandbox.down()

# CLI Commands
@click.group()
@click.pass_context
def cli(ctx):
    """CoDIN Development Tools CLI"""
    ctx.ensure_object(dict)

@cli.command()
@click.option('--file', required=True, help='File to format')
@click.option('--language', required=True, help='Programming language')
@click.option('--config', default='', help='Config file path')
@click.pass_context
def format(ctx, file, language, config):
    """Format code files."""
    asyncio.run(_run_format(file, language, config))

@cli.command()
@click.option('--file', required=True, help='File to lint')
@click.option('--language', required=True, help='Programming language')
@click.option('--fix/--no-fix', default=False, help='Auto-fix issues')
@click.pass_context
def lint(ctx, file, language, fix):
    """Lint code files."""
    asyncio.run(_run_lint(file, language, fix))

@cli.command()
@click.option('--path', default='.', help='Test path')
@click.option('--framework', default='auto', help='Test framework')
@click.option('--coverage/--no-coverage', default=True, help='Generate coverage')
@click.option('--verbose/--quiet', default=False, help='Verbose output')
@click.pass_context
def test(ctx, path, framework, coverage, verbose):
    """Run tests."""
    asyncio.run(_run_tests(path, framework, coverage, verbose))

# CLI command implementations
async def _run_format(file_path, language, config_path):
    """Run code formatting."""
    runner = CLIToolRunner()
    await runner.initialize()
    
    try:
        context = ToolContext(session_id="cli_format")
        formatter = runner.registry.get_tool("format_code")
        
        result = await formatter.run({
            'file_path': file_path,
            'language': language,
            'config_path': config_path
        }, context)
        
        if 'error' in result:
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        if result.get('formatted'):
            click.echo(f"✅ Formatted {file_path}")
        else:
            click.echo(f"⚠️ Formatting failed for {file_path}")
            
        if result.get('errors'):
            click.echo(f"Errors: {result['errors']}")
    
    finally:
        await runner.cleanup()

async def _run_lint(file_path, language, fix_issues):
    """Run code linting."""
    runner = CLIToolRunner()
    await runner.initialize()
    
    try:
        context = ToolContext(session_id="cli_lint")
        linter = runner.registry.get_tool("lint_code")
        
        result = await linter.run({
            'file_path': file_path,
            'language': language,
            'fix_issues': fix_issues
        }, context)
        
        if 'error' in result:
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        issues_count = result.get('issues_found', 0)
        if issues_count == 0:
            click.echo(f"✅ No issues found in {file_path}")
        else:
            click.echo(f"⚠️ Found {issues_count} issues in {file_path}")
            
            for issue in result.get('issues', []):
                severity_icon = "🔴" if issue['severity'] == 'error' else "🟡"
                click.echo(f"  {severity_icon} {issue['line']}")
        
        if fix_issues and result.get('fixed'):
            click.echo("🔧 Auto-fixes applied")
    
    finally:
        await runner.cleanup()

async def _run_tests(test_path, framework, coverage, verbose):
    """Run tests."""
    runner = CLIToolRunner()
    await runner.initialize()
    
    try:
        context = ToolContext(session_id="cli_test")
        test_runner = runner.registry.get_tool("run_tests")
        
        result = await test_runner.run({
            'test_path': test_path,
            'framework': framework,
            'coverage': coverage,
            'verbose': verbose
        }, context)
        
        if 'error' in result:
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        # Display test summary
        summary = result.get('summary', {})
        total = summary.get('total_tests', 0)
        passed = summary.get('passed', 0)
        failed = summary.get('failed', 0)
        
        if result.get('success'):
            click.echo(f"✅ All tests passed ({passed}/{total})")
        else:
            click.echo(f"❌ Tests failed ({failed} failed, {passed} passed)")
        
        if verbose and result.get('output'):
            click.echo("\nTest Output:")
            click.echo(result['output'])
    
    finally:
        await runner.cleanup()

if __name__ == '__main__':
    cli()
```

This tool system design provides a flexible, extensible framework for integrating diverse external functionality while maintaining security, performance, and ease of use.