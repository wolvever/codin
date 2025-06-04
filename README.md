# Codin

A modern coding agent framework with an elegant prompt system and iterative execution model, designed for simplicity and power.

## Features

- **Elegant Prompt System**: Simple API with `prompt_run(name, [version], [variables], [tools], stream=True/False)`
- **Iterative Code Agent**: True iterative execution following codex.rs logic
- **A2A Protocol**: Full Agent-to-Agent communication protocol compliance
- **Local/Remote Storage**: `fs://./path` or `http://host/path` template endpoints
- **Complete Tool System**: 12 core tools with exact argument specifications
- **Enhanced Memory System**: Title-weighted search with multiple chunk types
- **Enhanced File Reading**: Line range support for efficient reading of large files
- **DAG-based Planning**: Break down complex tasks into interdependent subtasks
- **Flexible LLM Support**: Any OpenAI-compatible LLM API
- **Real-time Streaming CLI**: Live LLM output with enhanced tool calling
- **Cross-platform Execution**: Windows PowerShell + Unix bash support
- **MCP Integration**: Model Context Protocol server support
- **Enhanced Output Formatting**: Better CLI display with configurable limits
- **Generic Tool Formatting**: Intelligent argument display with semantic icons for any tool

## Requirements

- Python 3.13 or higher
- Redis (optional, for distributed event handling)

## Installation

```bash
# Make sure you have Python 3.13 installed
python --version

# Install from source
git clone https://github.com/your-username/codin.git
cd codin
pip install -e .

# Alternative: Use uv (faster package manager)
uv venv
uv pip install -e .
```

## Quick Start

### Elegant Prompt System

```python
from codin.prompt import prompt_run, set_endpoint

# Set template storage location
set_endpoint("fs://./prompt_templates")

# Simple usage
response = await prompt_run("summarize", text="Long text to summarize...")

# With tools and conditions
response = await prompt_run(
    "code_assistant",
    tools=[execute_tool, read_file_tool],
    conditions={"model_family": "claude"},
    user_input="Help me debug this code"
)
```

### Iterative Code Agent

```python
import asyncio
from src.codin.agent.code_agent import CodeAgent, ApprovalPolicy
from src.codin.agent.base import AgentRunInput
from a2a.types import Message, TextPart, Role

async def main():
    # Create agent with iterative execution
    agent = CodeAgent(
        name="CodingAssistant",
        approval_policy=ApprovalPolicy.NEVER,
        max_turns=5
    )
    
    # Create task
    message = Message(
        messageId="user_1",
        role=Role.user,
        parts=[TextPart(text="Create a Python script to calculate fibonacci numbers")]
    )
    task = AgentRunInput(message=message)
    
    # Run task with automatic tool execution
    result = await agent.run(task)
    print(f"Success: {result.success}")
    print(f"Response: {result.message}")

asyncio.run(main())
```

## CLI Usage

Codin provides a simple command-line interface with two modes:

### Basic Usage

```bash
# Interactive mode (default) - starts REPL
codin

# Interactive mode with initial prompt
codin "write a python hello world script and run it"

# Quiet mode (single execution, no REPL)
codin -q "create a function"
```

### Advanced Options

```bash
# Run in full-auto mode (no approval required)
codin --approval-mode full-auto "Create a simple Flask application"

# Enable verbose output
codin --verbose "Create a simple Flask application"

# Combine flags
codin -q -v --approval-mode auto_edit "help me debug this"
```

### Interactive Mode (Default)

```bash
# Start REPL session
codin

# Start REPL with initial prompt
codin "Create a web scraper for news articles"

# Verbose interactive mode
codin -v "Help me optimize this code"
```

The interactive mode provides:
- **Conversational interface** with the AI assistant
- **Real-time streaming** responses
- **Tool execution** with approval controls
- **Session history** and context preservation
- **Project documentation** integration (`codin_rules.md`)
- **REPL commands**: `/help`, `/clear`, `/config`, `/tools`, `/exit`

### Quiet Mode

```bash
# Single execution, minimal output
codin -q "Fix this bug in my code"

# Verbose quiet mode (shows initialization details)
codin -q -v "Create unit tests for this function"
```

Quiet mode provides:
- **Single execution** - runs once and exits
- **No REPL interface** - direct prompt to response
- **Minimal output** - focused on results
- **Scriptable** - suitable for automation and CI/CD

## Complete Tool System

Codin now features a comprehensive tool system with **12 core tools** that match exact argument specifications:

### All 12 Core Tools

#### 1. **codebase_search**
- **Arguments**: 
  - `query` (string, required) - The search query to find relevant code
  - `explanation` (string, required) - One sentence explanation of why this tool is being used
  - `target_directories` (array of strings, optional) - Glob patterns for directories to search over

#### 2. **read_file**
- **Arguments**:
  - `target_file` (string, required) - The path of the file to read
  - `should_read_entire_file` (boolean, required) - Whether to read the entire file
  - `start_line_one_indexed` (integer, required) - The one-indexed line number to start reading from
  - `end_line_one_indexed_inclusive` (integer, required) - The one-indexed line number to end reading at
  - `explanation` (string, required) - One sentence explanation of why this tool is being used

#### 3. **run_shell** (unified from run_terminal_cmd, shell, sandbox_exec, container_exec)
- **Arguments**:
  - `command` (string, required) - The terminal command to execute
  - `is_background` (boolean, required) - Whether the command should be run in the background
  - `explanation` (string, required) - One sentence explanation of why this command needs to be run

#### 4. **list_dir**
- **Arguments**:
  - `relative_workspace_path` (string, required) - Path to list contents of, relative to the workspace root
  - `explanation` (string, required) - One sentence explanation of why this tool is being used

#### 5. **grep_search**
- **Arguments**:
  - `query` (string, required) - The regex pattern to search for
  - `explanation` (string, required) - One sentence explanation of why this tool is being used
  - `case_sensitive` (boolean, optional) - Whether the search should be case sensitive
  - `include_pattern` (string, optional) - Glob pattern for files to include
  - `exclude_pattern` (string, optional) - Glob pattern for files to exclude

#### 6. **edit_file**
- **Arguments**:
  - `target_file` (string, required) - The target file to modify
  - `instructions` (string, required) - A single sentence instruction describing what you are going to do
  - `code_edit` (string, required) - Specify ONLY the precise lines of code that you wish to edit

#### 7. **search_replace**
- **Arguments**:
  - `file_path` (string, required) - The path to the file you want to search and replace in
  - `old_string` (string, required) - The text to replace (must be unique within the file)
  - `new_string` (string, required) - The edited text to replace the old_string

#### 8. **file_search**
- **Arguments**:
  - `query` (string, required) - Fuzzy filename to search for
  - `explanation` (string, required) - One sentence explanation of why this tool is being used

#### 9. **delete_file**
- **Arguments**:
  - `target_file` (string, required) - The path of the file to delete, relative to the workspace root
  - `explanation` (string, required) - One sentence explanation of why this tool is being used

#### 10. **reapply**
- **Arguments**:
  - `target_file` (string, required) - The relative path to the file to reapply the last edit to

#### 11. **web_search**
- **Arguments**:
  - `search_term` (string, required) - The search term to look up on the web
  - `explanation` (string, required) - One sentence explanation of why this tool is being used

#### 12. **mcp_fetch_fetch**
- **Arguments**:
  - `url` (string, required) - URL to fetch
  - `max_length` (integer, optional, default: 5000) - Maximum number of characters to return
  - `raw` (boolean, optional, default: false) - Get the actual HTML content without simplification
  - `start_index` (integer, optional, default: 0) - Starting character index for output

### Tool System Usage

```python
from codin.tool.core_tools import CoreToolset
from codin.sandbox.base import LocalSandbox

# Initialize
sandbox = LocalSandbox()
toolset = CoreToolset(sandbox)
await toolset.initialize()

# Use any tool
read_tool = toolset.get_tool("read_file")
result = await read_tool.run({
    "target_file": "example.py",
    "should_read_entire_file": False,
    "start_line_one_indexed": 1,
    "end_line_one_indexed_inclusive": 10,
    "explanation": "Reading file header"
}, context)
```

### Enhanced Tool Registry

```python
from codin.tool import ToolRegistry, CoreToolset

registry = ToolRegistry()
registry.register_toolset(CoreToolset(sandbox))

# Both names available due to prefix removal:
# registry.get_tool("sandbox_read_file")  # Original name
# registry.get_tool("read_file")          # Simplified name

# Get all tools (with executor if configured)
tools = registry.get_tools_with_executor()
```

## Enhanced Memory System

The memory system has been significantly enhanced with better structure, search capabilities, and content management:

### Key Memory Features

#### 1. **Enhanced MemoryChunk Structure**

The MemoryChunk now includes:
- **`doc_id`**: Document identifier for grouping related chunks
- **`chunk_id`**: Unique identifier for the specific chunk
- **`chunk_type`**: Enum-based type classification (MEMORY_SUMMARY, MEMORY_ENTITY, MEMORY_ID_MAPPING)
- **`content`**: Flexible content storage (string or dictionary)
- **`title`**: Human-readable title with higher search weight
- **`metadata`**: Additional metadata storage

#### 2. **Chunk Types**

Three distinct chunk types are supported:

```python
class ChunkType(Enum):
    MEMORY_SUMMARY = "memory_summary"      # Conversation summaries
    MEMORY_ENTITY = "memory_entity"        # Extracted entities
    MEMORY_ID_MAPPING = "memory_id_mapping" # ID to name mappings
```

#### 3. **Title-Weighted Search**

Search functionality prioritizes matches in titles:
- **Title Match**: +5 points
- **Content Match**: +2 points  
- **Dictionary Key/Value Match**: +1 point
- **Exact Title Match**: +3 bonus points
- **Chunk Type Match**: +1 bonus point

#### 4. **Memory Usage Examples**

```python
# Summary chunk
summary_chunk = MemoryChunk(
    doc_id="doc-001",
    chunk_id="summary-001", 
    session_id="session-1",
    chunk_type=ChunkType.MEMORY_SUMMARY,
    content="User discussed Flask web development...",
    title="Flask Web App Discussion"
)

# Entity chunk with dictionary content
entities = {
    "framework": "Flask",
    "language": "Python", 
    "features": ["authentication", "REST API"]
}
entity_chunk = MemoryChunk(
    doc_id="doc-001",
    chunk_id="entities-001",
    session_id="session-1", 
    chunk_type=ChunkType.MEMORY_ENTITY,
    content=entities,
    title="Technical Entities"
)

# Search with title weighting
results = await memory_store.search_memory_chunks(
    "session-1", 
    "flask", 
    limit=5
)
# Returns chunks with "Flask" in title first
```

## Real-time Streaming CLI

The CLI now features real-time LLM streaming for immediate responsiveness:

### Streaming Features

#### **Real-time Text Display**
- Text appears character-by-character as generated by the LLM
- No buffering - immediate display of content
- Proper event coordination between streaming and tool execution

#### **Enhanced Output Formatting**
- **Better truncation**: Increased from 5,000 to 20,000 characters
- **Line count display**: Shows "first X lines of Y total lines"
- **Emoji indicators**: 📝 for writing, 📖 for reading, 🏃 for execution
- **Configurable limits**: `--full-output` and `--max-output-lines` options

#### **Duplicate Prevention**
- **Tool call tracking**: Prevents duplicate tool calls in same turn
- **Smart intervention**: Detects loops and suggests next steps
- **Configuration option**: `prevent_duplicate_tools` setting

### CLI Configuration Options

```bash
# Basic streaming (default)
codin "Write a Python function"

# Full output without truncation
codin --full-output "Generate a large file"

# Verbose mode with streaming
codin --verbose "Create a web scraper"

# Quiet mode (no streaming)
codin --quiet "Fix this bug"

# Custom output limits
codin --max-output-lines 100 "Show me a long file"
```

### Cross-Platform Support

The `LocalSandbox` has been enhanced to:
- **Auto-detect platform**: Automatically detects Windows vs Unix and uses appropriate shell
- **Cross-platform Python execution**: Handles both PowerShell and bash environments
- **UTF-8 encoding support**: Properly handles Unicode characters on Windows
- **Direct command execution**: Avoids shell escaping issues by using subprocess directly

### Enhanced Tool Call Parsing

The `CodeAgent` now supports multiple tool call formats:

#### Format 1: Claude-style Multiple Tool Calls
```xml
<function_calls>
[
  {"name": "file_write", "arguments": {"path": "hello.py", "content": "print('Hello!')"}},
  {"name": "shell", "arguments": {"command": ["python", "hello.py"]}}
]
</function_calls>
```

#### Format 2: Individual Tool Calls
```xml
<function_call name="file_read">
{"path": "example.py"}
</function_call>
```

#### Format 3: Thinking with Tool Calls
```xml
I need to create a file first:
<function_call name="file_write">
{"path": "test.py", "content": "print('test')"}
</function_call>
```

## Configuration

### Basic Configuration

Create `~/.codin/config.yaml` or `config.yaml` in your project:

```yaml
# Core settings
model: "claude-3-5-sonnet-20241022"
provider: "openai"
approval_mode: "suggest"

# Output settings (NEW!)
verbose: true
debug: true
max_output_lines: 50
max_output_chars: 20000
show_full_output: false
prevent_duplicate_tools: true

# Provider configurations with custom base URLs
providers:
  openai:
    name: "OpenAI"
    base_url: "https://aiproxy.usw.sealos.io/v1"  # Custom proxy URL
    env_key: "OPENAI_API_KEY"
    models: ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022"]
  
  anthropic:
    name: "Anthropic"
    base_url: "https://api.anthropic.com"
    env_key: "ANTHROPIC_API_KEY"
    models: ["claude-3.5-sonnet", "claude-3-opus"]

# MCP server configurations
mcp_servers:
  fetch:
    command: "uvx"
    args: ["mcp-server-fetch"]
    description: "Fetch content from URLs and web pages"
  
  baidu_search:
    url: "http://appbuilder.baidu.com/v2/ai_search/mcp/sse?api_key=your-key"
    description: "Baidu search MCP server via SSE"

# Tool registry configuration (NEW!)
tool_prefix_removal: true
auto_initialize: true
```

### Environment Variables

Configure Codin using environment variables:

```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key

# CLI Configuration
CODIN_APPROVAL_MODE=suggest
LLM_MODEL=claude-3-5-sonnet
LLM_PROVIDER=openai
CODIN_RULE=1

# Prompt Template Storage
CODIN_PROMPT_ENDPOINT=fs://./prompt_templates
```

### View Configuration

```bash
# Show current configuration
codin config

# Show available providers
codin providers
```

## Elegant Prompt System

A concise and powerful prompt template system with A2A protocol support.

### Key Features

- **Simple API**: `prompt_run(name, [version], [variables], [tools], stream=True/False)`
- **Dict-based Conditions**: Simple condition matching for template variants
- **Auto LLM Detection**: Automatic model capability detection
- **Template Versions**: Support for versioned templates with automatic selection
- **Tool Integration**: Seamless tool definition and conversion
- **History Support**: Conversation history integration

### Template Structure

```yaml
name: code_assistant
version: latest
variants:
  - text: |
      You are a coding assistant...
      {% if has_tools %}Available tools: {{ tool_names|join(', ') }}{% endif %}
    conditions:
      model_family: claude
      tool_support: true
  - text: |
      You are a helpful coding assistant...
    conditions: {}  # Default variant
```

### Simple Condition Matching

```python
conditions = {
    "model_family": "claude",      # Prefix matching: "claude" matches "claude-3-5-sonnet"
    "tool_support": True,          # Exact boolean matching
    "locale": "en",                # Prefix matching: "en" matches "en-US"
    "custom_field": "value"        # Exact string matching
}
```

### API Reference

#### `prompt_run(name, /, *, version=None, variables=None, tools=None, history=None, conditions=None, stream=False, **kwargs)`

Execute a prompt template with full LLM integration.

**Parameters:**
- `name`: Template name (required)
- `version`: Template version (optional, defaults to "latest")
- `variables`: Template variables (dict)
- `tools`: Available tools (list of ToolDefinition or dict)
- `history`: Conversation history (list of A2AMessage or dict)
- `conditions`: Template selection conditions (dict)
- `stream`: Stream response (bool)
- `**kwargs`: Additional template variables

**Returns:** `A2AResponse` with message and content

#### `render_only(name, /, *, version=None, variables=None, conditions=None, **kwargs)`

Render a template without executing LLM.

**Returns:** Rendered prompt text (str)

#### `set_endpoint(endpoint)`

Set the global template storage endpoint.

**Endpoint formats:**
- `fs://./prompt_templates` - Local filesystem
- `http://host:port/path` - Remote HTTP with caching

### Automatic Variables

The system automatically provides these variables:

- `model`: Current model name
- `model_family`: Model family (claude, openai, google)
- `model_provider`: Provider name
- `streaming`: Whether streaming is enabled
- `has_tools`: Boolean indicating if tools are available
- `tools`: List of tool dictionaries
- `tool_names`: List of tool names
- `tool_descriptions`: List of formatted tool descriptions
- `has_history`: Boolean indicating if history exists
- `history`: List of A2A messages
- `history_text`: Formatted history text

## Iterative Code Agent System

The CodeAgent implements a true iterative execution model following the logic from `codex.rs`.

### Key Features

1. **Iterative Execution Loop** - User input → Agent analysis → Tool calls → Results → Continue
2. **Tool Call Parsing** - Automatic parsing and execution of structured tool calls
3. **Event System** - Comprehensive monitoring and debugging events
4. **Approval System** - Configurable safety policies
5. **Conversation Persistence** - Full A2A protocol message history
6. **Prompt Template Integration** - Dynamic template selection based on model capabilities
7. **Duplicate Prevention** - Prevents repeated tool calls in same turn
8. **Enhanced Output** - Better formatting and configurable limits
9. **Real-time Streaming** - Live LLM output with proper event coordination

### Tool Call Format

```xml
<function_call name="tool_name">
{"parameter": "value", "other_param": "value"}
</function_call>
```

### Event Monitoring

```python
async def event_monitor(event: AgentEvent) -> None:
    print(f"Event: {event.event_type} - {event.data}")

agent.add_event_callback(event_monitor)
```

**Event Types:**
- `task_start/task_complete` - Task lifecycle
- `turn_start/turn_complete` - Conversation turns
- `llm_response` - LLM generations
- `llm_text_delta` - Real-time streaming text chunks (NEW!)
- `tool_call_start/end` - Tool execution
- `approval_requested` - User approval needed
- `task_error` - Error handling

### Approval Policies

```python
from src.codin.agent.code_agent import ApprovalPolicy

# Auto-approve all safe commands
agent = CodeAgent(approval_policy=ApprovalPolicy.NEVER)

# Ask for approval on every tool call
agent = CodeAgent(approval_policy=ApprovalPolicy.ALWAYS)

# Only ask for potentially dangerous operations
agent = CodeAgent(approval_policy=ApprovalPolicy.UNSAFE_ONLY)
```

### Loop Logic (following codex.rs)

```
┌─────────────────┐
│   User Input    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Load Template  │
│  & Call LLM     │ ← Real-time streaming
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Parse Response  │
│ for Tool Calls  │
└─────────┬───────┘
          │
    ┌─────▼─────┐
    │ Any Tools? │
    └─────┬─────┘
          │
    No    │    Yes
    ┌─────▼─────┐
    │   Done    │
    └───────────┘
          │
          ▼
┌─────────────────┐
│ Execute Tools   │
│ & Get Results   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Add Results    │
│ to Conversation │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Continue Loop  │
│ (up to max_turns)│
└─────────────────┘
```

## MCP Integration

Codin supports Model Context Protocol (MCP) servers for extended functionality.

### Configuration

```yaml
mcp_servers:
  fetch:
    command: "uvx"
    args: ["mcp-server-fetch"]
    description: "Fetch content from URLs and web pages"
  
  baidu_search:
    url: "http://appbuilder.baidu.com/v2/ai_search/mcp/sse?api_key=your-key"
    description: "Baidu search MCP server via SSE"
```

### Viewing MCP Servers

```bash
# Show MCP servers in configuration
codin config
```

## Host Usage

### Single Agent Host

```python
import asyncio
from codin.model.openai_llm import OpenAILLM
from codin.host.single import create_dag_agent_host

async def main():
    # Create an LLM
    llm = OpenAILLM()  # Will use LLM_MODEL, LLM_BASE_URL, and LLM_API_KEY from environment
    await llm.prepare()
    
    # Create a host with a DAG-based agent
    host = await create_dag_agent_host(
        llm=llm,
        workspace_dir="./workspace",
        interactive=True,
    )
    
    # Run the agent interactively
    await host.run_interactive()

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Agent Host

```python
import asyncio
from codin.model.openai_llm import OpenAILLM
from codin.host.multi import create_multi_agent_host

async def main():
    # Create an LLM
    llm = OpenAILLM()
    await llm.prepare()
    
    # Define agent specifications
    agent_specs = [
        {
            "id": "planner",
            "name": "Planning Agent",
            "description": "Creates plans for coding tasks",
        },
        {
            "id": "executor",
            "name": "Execution Agent",
            "description": "Executes coding tasks",
        },
    ]
    
    # Create a multi-agent host
    host = await create_multi_agent_host(
        llm=llm,
        agent_specs=agent_specs,
        workspace_dir="./workspace",
    )
    
    # Run a conversation
    conversation = await host.run_conversation(
        initial_prompt="Create a simple Flask application with a home page",
        coordinator_agent_id="planner",
    )
    
    # Save the conversation history
    host.save_history("conversation.json")

if __name__ == "__main__":
    asyncio.run(main())
```

## A2A Protocol Support

Full compliance with Agent-to-Agent communication protocol:

### Message Structure

```python
from codin.prompt.base import A2AMessage, A2ARole, A2ATextPart

message = A2AMessage(
    message_id="unique-id",
    role=A2ARole.USER,
    parts=[A2ATextPart(text="Hello")],
    context_id="conversation-id",
    task_id="task-id"
)
```

### Response Structure

```python
response = await prompt_run("template", user_input="Hello")

# Access response
print(response.content)           # String content
print(response.message.parts[0])  # A2A message parts
print(response.streaming)         # Streaming flag
```

## Tool Integration

### Tool Definition

```python
from codin.prompt.base import ToolDefinition

tool = ToolDefinition(
    name="execute_code",
    description="Execute Python code",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code"}
        },
        "required": ["code"]
    }
)
```

### Automatic Conversion

```python
# Dict format (automatically converted)
tools = [{
    "name": "read_file",
    "description": "Read file contents",
    "parameters": {"type": "object", "properties": {...}}
}]

response = await prompt_run("template", tools=tools)
```

### Tool Decorators

```python
from codin.tool import tool

@tool(name="my_tool", description="Custom tool")
async def my_function(
    arg1: str,
    arg2: int = 42,
    tool_context: ToolContext = None
) -> dict:
    return {"result": f"{arg1}_{arg2}"}
```

## Template Development

Templates are stored in `prompt_templates/` directory using YAML format:

```yaml
name: code_helper
version: latest
variants:
  - text: |
      You are {{ model_family }} coding assistant.
      
      {% if has_history %}
      Previous conversation:
      {{ history_text }}
      {% endif %}
      
      {% if has_tools %}
      Available tools:
      {% for tool in tools %}
      - {{ tool.name }}: {{ tool.description }}
      {% endfor %}
      {% endif %}
      
      User request: {{ user_input }}
    conditions:
      tool_support: true
```

## Using External LLM APIs

Codin supports any OpenAI-compatible LLM API:

```python
import asyncio
import os
from dotenv import load_dotenv
from codin.model.openai_llm import OpenAILLM
from codin.host.single import create_dag_agent_host

async def main():
    # Load environment variables
    load_dotenv()
    
    # Set up environment variables for the LLM
    os.environ["LLM_BASE_URL"] = "https://your-llm-api-url.com/v1"
    os.environ["LLM_API_KEY"] = "your-api-key"
    os.environ["LLM_MODEL"] = "gpt-4o"  # Or any model supported by your API
    
    # Create an LLM
    llm = OpenAILLM()
    await llm.prepare()
    
    # Create a host with a DAG-based agent
    host = await create_dag_agent_host(
        llm=llm,
        workspace_dir="./workspace",
        interactive=True,
    )
    
    # Run the agent interactively
    await host.run_interactive()

if __name__ == "__main__":
    asyncio.run(main())
```

## Examples

### Basic Usage with Prompt System

```python
import asyncio
from codin.prompt import prompt_run, set_endpoint

async def main():
    set_endpoint("fs://./templates")
    
    # Simple text generation
    response = await prompt_run(
        "summarize",
        text="Long article text here...",
        max_length=100
    )
    print(response.content)

asyncio.run(main())
```

### Coding Assistant with Tools and History

```python
async def coding_assistant():
    tools = [
        {
            "name": "execute_python",
            "description": "Execute Python code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"}
                }
            }
        }
    ]
    
    history = [
        {"role": "user", "content": "I need help with Python"},
        {"role": "agent", "content": "I'd be happy to help!"}
    ]
    
    response = await prompt_run(
        "code_assistant",
        tools=tools,
        history=history,
        conditions={"model_family": "claude"},
        user_input="Write a factorial function"
    )
    
    return response.content
```

### Streaming Response

```python
async def stream_example():
    response = await prompt_run(
        "creative_writer",
        topic="space exploration",
        stream=True
    )
    
    if response.streaming:
        async for chunk in response.content:
            print(chunk, end="", flush=True)
    else:
        print(response.content)
```

### Code Review and Debugging

```python
from src.codin.agent.code_agent import CodeAgent

agent = CodeAgent()

# Code review
review = await agent.review_code(
    code="def hello(): print('world')",
    language="python"
)

# Debugging
debug_help = await agent.debug_code(
    code="buggy_code_here",
    error_description="NameError: name 'x' is not defined",
    expected_behavior="Should print numbers 1-10",
    language="python"
)
```

## Error Handling

The system provides graceful error handling:

```python
response = await prompt_run("nonexistent_template")

if response.error:
    print(f"Error: {response.error['message']}")
    print(f"Type: {response.error['type']}")
    print(f"Time: {response.error['timestamp']}")
```

## Performance Features

- **Template Compilation**: Jinja2 templates are pre-compiled for performance
- **In-Memory Caching**: Templates cached in memory after first load
- **HTTP Caching**: Remote templates cached with ETag support
- **Lazy Loading**: LLMs created only when needed
- **Capability Detection**: Automatic model capability detection
- **Output Optimization**: Configurable truncation and formatting
- **Real-time Streaming**: Immediate LLM output display
- **Memory Optimization**: Title-weighted search with efficient indexing

## Best Practices

1. **Use Simple Conditions**: Keep condition dictionaries simple and focused
2. **Set Reasonable max_turns**: Prevent infinite loops in iterative execution
3. **Version Templates**: Use semantic versioning for template versions
4. **Monitor Events**: Use event callbacks for debugging and observability
5. **Handle Errors Gracefully**: Implement proper error handling in production
6. **Reset Conversations**: Periodically reset for long sessions
7. **Configure Output Limits**: Use `--full-output` and `--max-output-lines` appropriately
8. **Use MCP Servers**: Extend functionality with Model Context Protocol servers
9. **Leverage Streaming**: Enable streaming for better user experience
10. **Optimize Memory**: Use title-weighted search for better relevance

## Troubleshooting

### Template Issues

```python
# Check available templates
from codin.prompt import get_registry

registry = get_registry()
templates = await registry.list()
print("Available templates:", templates)
```

### Storage Issues

```python
# Test storage backend
from codin.prompt.storage import get_storage_backend

storage = get_storage_backend("fs://./templates")
templates = await storage.list_templates()
print("Storage templates:", templates)
```

### Agent Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add event monitoring
agent.add_event_callback(print)
```

### CLI Issues

```bash
# Check configuration
codin config

# Test with verbose output
codin --verbose "simple task"

# Use full output for debugging
codin --full-output --verbose "complex task"

# Test streaming
codin --no-stream "test without streaming"
```

### Tool Issues

```python
# Test core toolset
from codin.tool.core_tools import CoreToolset
from codin.sandbox.base import LocalSandbox

sandbox = LocalSandbox()
toolset = CoreToolset(sandbox)
await toolset.initialize()

# Check tool availability
tools = toolset.get_all_tools()
print(f"Available tools: {[t.name for t in tools]}")
```

### Memory Issues

```python
# Test memory system
from codin.memory import MemoryChunk, ChunkType

chunk = MemoryChunk(
    doc_id="test",
    chunk_id="test-1",
    session_id="test-session",
    chunk_type=ChunkType.MEMORY_SUMMARY,
    content="Test content",
    title="Test Title"
)

# Test search functionality
results = await memory_store.search_memory_chunks("test-session", "test")
print(f"Search results: {len(results)}")
```

## Migration Notes

### From Stream Subcommand

**Old way (deprecated):**
```bash
codin stream "write hello world"
```

**New way (default streaming):**
```bash
codin "write hello world"              # Streaming by default
codin --no-stream "write hello world"  # Disable streaming
```

### Configuration Updates

The configuration system now supports:
- Enhanced output settings (`max_output_lines`, `max_output_chars`, `show_full_output`)
- Duplicate prevention (`prevent_duplicate_tools`)
- MCP server configurations
- Provider-specific settings
- Tool registry configuration (`tool_prefix_removal`, `auto_initialize`)

### Tool System Updates

- **Unified Shell Tool**: `run_terminal_cmd`, `shell`, `sandbox_exec`, `container_exec` are now unified as `run_shell`
- **Enhanced Registry**: Automatic prefix removal with conflict detection
- **Exact Specifications**: All 12 tools now match exact argument specifications
- **Comprehensive Testing**: Full test coverage for all tools

## Testing

The system includes comprehensive testing:

### Core Tool Tests
- ✅ **17/17 core tool tests** passing
- ✅ All 12 tools verified with exact argument specifications
- ✅ Error handling and edge cases covered

### Registry Tests
- ✅ **14/14 registry tests** passing
- ✅ Prefix removal logic verified
- ✅ Tool initialization and cleanup tested

### Memory Tests
- ✅ **16/16 memory system tests** passing
- ✅ Title-weighted search functionality
- ✅ Multiple chunk types and content handling

### CLI Tests
- ✅ Streaming functionality verified
- ✅ Output formatting and truncation
- ✅ Cross-platform compatibility

## Comparison with codex.rs

| Feature | codex.rs | Python Codin |
|---------|----------|--------------|
| Iterative Loop | ✅ | ✅ |
| Tool Execution | ✅ | ✅ |
| Approval System | ✅ | ✅ |
| Event Emission | ✅ | ✅ |
| Conversation History | ✅ | ✅ |
| Function Calling | ✅ | ✅ |
| A2A Protocol | ✅ | ✅ |
| Template System | ❌ | ✅ |
| MCP Integration | ✅ | ✅ |
| Streaming CLI | ✅ | ✅ |
| Output Formatting | ❌ | ✅ |
| Duplicate Prevention | ❌ | ✅ |
| Memory System | ❌ | ✅ |
| Tool Registry | ❌ | ✅ |
| Real-time Streaming | ❌ | ✅ |

## License

MIT 