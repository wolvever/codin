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

This tool system design provides a flexible, extensible framework for integrating diverse external functionality while maintaining security, performance, and ease of use.