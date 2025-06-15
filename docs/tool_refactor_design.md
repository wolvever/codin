# Tool System Refactor Design

## Overview

This document describes the refactored tool system architecture that separates tool specifications from implementations and provides a unified execution interface with extension support.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent / Planner Layer                    │
├─────────────────────────────────────────────────────────────────┤
│  • Tool Planning                                               │
│  • Function Call Generation                                    │
│  • Only sees ToolSpec (descriptions + schemas)                 │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Tool Specification Layer                    │
├─────────────────────────────────────────────────────────────────┤
│  ToolSpecRegistry                                              │
│  ├── ToolSpec (name, description, input/output schemas)        │
│  ├── ToolMetadata (version, category, permissions)             │
│  └── Loaders (file, HTTP, database)                           │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Unified Executor Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  UnifiedToolExecutor                                           │
│  ├── Finds appropriate executor for tool type                  │
│  ├── Handles timeouts, retries, concurrency                   │
│  └── Orchestrates extension pipeline                          │
└─────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│   Extension Layer    │ │   Extension Layer    │ │   Extension Layer    │
├──────────────────────┤ ├──────────────────────┤ ├──────────────────────┤
│  • Auth Extension    │ │  • Approval Ext.     │ │  • Logging Ext.      │
│  • Metrics Extension │ │  • Rate Limit Ext.   │ │  • Custom Extensions │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│  Python Executor     │ │   MCP Executor       │ │  Sandbox Executor    │
├──────────────────────┤ ├──────────────────────┤ ├──────────────────────┤
│  • Function calls    │ │  • MCP servers       │ │  • Shell commands    │
│  • Module imports    │ │  • Session mgmt      │ │  • Sandbox methods   │
│  • Async/sync        │ │  • Protocol handling │ │  • Environment mgmt  │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
```

## Key Components

### 1. Tool Specification System

**Purpose**: Define what tools do, not how they do it.

```python
# Tool specifications are data, not code
tool_spec = ToolSpec(
    name="calculate_sum",
    description="Add two numbers together",
    tool_type=ToolType.PYTHON,
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        }
    },
    implementation_config={
        "module": "math_tools",
        "function_name": "add"
    }
)
```

**Benefits**:
- Agents/Planners only see tool capabilities
- Specifications can be loaded from files/endpoints
- Easy to version and manage
- Implementation details are hidden

### 2. Unified Executor Interface

**Purpose**: Handle all tool types through a single interface.

```python
# All tool types executed the same way
result = await executor.execute(
    tool_name="any_tool",
    args={"param": "value"},
    context=ToolContext(session_id="123")
)
```

**Supported Tool Types**:
- **Python**: Function calls, module imports
- **MCP**: Model Context Protocol servers
- **Sandbox**: Shell commands, containerized execution
- **HTTP**: REST API calls
- **Custom**: Extensible for new types

### 3. Extension System

**Purpose**: Handle cross-cutting concerns as middleware.

```python
# Extensions process all tool executions
class CustomExtension(Extension):
    async def before_execute(self, ctx):
        # Auth, validation, logging
        pass
    
    async def after_execute(self, ctx, result):
        # Metrics, audit, post-processing
        return result
```

**Built-in Extensions**:
- **Authentication**: Role-based access control
- **Approval**: User approval for dangerous operations
- **Logging**: Comprehensive execution logging
- **Metrics**: Prometheus + OpenTelemetry metrics
- **Rate Limiting**: Prevent abuse
- **Audit**: Compliance and security trails

## Usage Examples

### Agent/Planner Usage

```python
# Agents only work with specifications
spec_registry = ToolSpecRegistry()
await spec_registry.load_all()

# Get tool schemas for LLM function calling
openai_tools = spec_registry.to_openai_tools()
mcp_tools = spec_registry.to_mcp_tools()

# Plan without knowing implementations
for spec in spec_registry.list_specs(tool_type="python"):
    print(f"Available: {spec.name} - {spec.description}")
```

### Executor Usage

```python
# Single interface for all tool types
executor = UnifiedToolExecutor(spec_registry)

# Execute any tool type the same way
await executor.execute("python_function", args)
await executor.execute("mcp_tool", args) 
await executor.execute("shell_command", args)
```

### Extension Usage

```python
# Add custom cross-cutting logic
extension_manager = ExtensionManager()
extension_manager.register(AuthExtension())
extension_manager.register(LoggingExtension())
extension_manager.register(MyCustomExtension())

executor = UnifiedToolExecutor(
    spec_registry=spec_registry,
    extension_manager=extension_manager
)
```

## Configuration

### Tool Specification Files

```yaml
# tools.yaml
tools:
  - name: file_reader
    description: Read file contents
    tool_type: python
    input_schema:
      type: object
      properties:
        path: {type: string}
    implementation_config:
      module: file_tools
      function_name: read_file
    metadata:
      category: filesystem
      requires_approval: false

  - name: dangerous_delete
    description: Delete files
    tool_type: shell
    input_schema:
      type: object
      properties:
        path: {type: string}
    implementation_config:
      command: "rm {path}"
    metadata:
      is_dangerous: true
      requires_approval: true
```

### Extension Configuration

```python
# Configure extension chain
extension_manager = ExtensionManager()

# Auth first (highest priority)
extension_manager.register(AuthExtension(
    priority=ExtensionPriority.HIGHEST
))

# Then approval for dangerous ops
extension_manager.register(ApprovalExtension(
    approval_mode=ApprovalMode.UNSAFE_ONLY,
    priority=ExtensionPriority.HIGH
))

# Logging and metrics
extension_manager.register(LoggingExtension())
extension_manager.register(MetricsExtension())
```

## Migration Strategy

### Phase 1: Add New System Alongside Existing

1. Implement new tool specification system
2. Create unified executor 
3. Add extension framework
4. Keep existing tool system running

### Phase 2: Migrate Tools Gradually

1. Convert high-value tools to new system
2. Create adapter layer for backward compatibility
3. Update agents to use new specifications
4. Validate functionality parity

### Phase 3: Replace Old System

1. Migrate remaining tools
2. Remove old tool infrastructure
3. Clean up deprecated code
4. Update documentation

## Benefits Summary

### For Developers
- **Easier tool development**: Clear separation of concerns
- **Better testing**: Mock executors independently from specs
- **Flexible deployment**: Different executor configs per environment

### For Agents/Planners  
- **Simplified interface**: Only see tool capabilities, not implementations
- **Better planning**: Rich metadata for tool selection
- **Consistent format**: Same schema regardless of implementation

### For Operations
- **Centralized control**: Extensions handle security, logging, metrics
- **Easy monitoring**: Built-in observability
- **Flexible policies**: Configure approval/auth per tool

### For the System
- **Scalability**: Unified concurrency and resource management  
- **Extensibility**: Easy to add new tool types and extensions
- **Maintainability**: Clear boundaries between components