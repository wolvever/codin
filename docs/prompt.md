# How to Use `prompt_run`

`prompt_run` is the main API for executing prompt templates in the codin framework. It provides a simple, elegant interface for running prompts with automatic LLM detection, template selection, and A2A protocol compliance.

## Quick Start

### 1. Set Template Storage Location

First, configure where your prompt templates are stored:

```python
from codin.prompt import set_endpoint

# Use local filesystem storage
set_endpoint("fs://./prompt_templates")

# Or use remote HTTP storage  
set_endpoint("http://your-server.com/templates")
```

### 2. Basic Usage

```python
from codin.prompt import prompt_run

# Simple text summarization
response = await prompt_run(
    "summarize", 
    text="Long article text here..."
)
print(response.content)
```

### 3. With Tools and Conditions

```python
# Define tools
tools = [
    {
        "name": "file_read",
        "description": "Read a file from the filesystem",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            }
        }
    },
    {
        "name": "shell",
        "description": "Execute shell commands",
        "parameters": {
            "type": "object", 
            "properties": {
                "command": {"type": "array", "items": {"type": "string"}}
            }
        }
    }
]

# Execute with specific model family and tools
response = await prompt_run(
    "code_agent_main",
    tools=tools,
    conditions={"model_family": "claude"},
    user_input="Help me debug this Python script",
    sandbox_type="local",
    agent_name="CodeAgent"
)
```

## API Reference

### Function Signature

```python
async def prompt_run(
    name: str,                                    # Template name (required)
    /,                                           # Positional-only separator
    version: str | None = None,                  # Template version (optional)
    variables: dict[str, Any] | None = None,     # Template variables
    tools: list[ToolDefinition | dict] | None = None,  # Available tools
    history: list[A2AMessage | dict] | None = None,    # Conversation history
    conditions: dict[str, Any] | None = None,    # Template selection conditions
    stream: bool = False,                        # Stream response
    **kwargs                                     # Additional variables
) -> A2AResponse:
```

### Parameters

#### `name` (required)
The name of the prompt template to execute. Must match a template file in your storage location.

```python
# Uses template from prompt_templates/summarize.yaml
await prompt_run("summarize", text="content")
```

#### `version` (optional)
Specific template version. Defaults to "latest".

```python
await prompt_run("code_agent", version="v2.1", user_input="Help me")
```

#### `variables` (optional)
Dictionary of variables to pass to the template for rendering.

```python
await prompt_run(
    "translate",
    variables={
        "source_language": "English",
        "target_language": "Spanish", 
        "text": "Hello world"
    }
)
```

#### `tools` (optional)
List of tools available to the LLM. Can be `ToolDefinition` objects or dictionaries.

```python
# Using dictionaries
tools = [
    {
        "name": "search_web",
        "description": "Search the internet",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }
    }
]

# Using ToolDefinition objects
from codin.prompt import ToolDefinition

tools = [
    ToolDefinition(
        name="execute_code",
        description="Execute Python code",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to run"}
            }
        }
    )
]

await prompt_run("code_assistant", tools=tools, task="Write a function")
```

#### `history` (optional)
Conversation history as A2A messages or dictionaries.

```python
# Using dictionaries
history = [
    {"role": "user", "content": "Hello"},
    {"role": "agent", "content": "Hi! How can I help you?"},
    {"role": "user", "content": "What's the weather like?"}
]

# Using A2A Message objects
from codin.prompt import A2AMessage, A2ARole, A2ATextPart

history = [
    A2AMessage(
        message_id="msg1",
        role=A2ARole.user,
        parts=[A2ATextPart(text="Hello")],
    ),
    A2AMessage(
        message_id="msg2", 
        role=A2ARole.agent,
        parts=[A2ATextPart(text="Hi! How can I help?")],
    )
]

await prompt_run("continue_conversation", history=history)
```

#### `conditions` (optional)
Template selection conditions for choosing the best variant.

```python
await prompt_run(
    "multi_model_template",
    conditions={
        "model_family": "claude",      # Prefer Claude variants
        "model_provider": "anthropic", # From Anthropic
        "use_case": "coding",          # For coding tasks
        "language": "python"           # Python-specific
    }
)
```

#### `stream` (optional)
Whether to stream the response or return complete text.

```python
# Streaming response
response = await prompt_run("generate_story", stream=True)
async for chunk in response.content:
    print(chunk, end="", flush=True)

# Complete response  
response = await prompt_run("generate_story", stream=False)
print(response.content)
```

#### `**kwargs`
Additional variables passed directly to template rendering.

```python
# These are equivalent:
await prompt_run("template", variables={"name": "John", "age": 30})
await prompt_run("template", name="John", age=30)
```

### Return Value: A2AResponse

The function returns an `A2AResponse` object with the following structure:

```python
@dataclass
class A2AResponse:
    message: A2AMessage | None = None           # Structured A2A message
    streaming: bool = False                     # Whether response is streaming
    content: str | AsyncIterator[str] | None    # Response content
    error: dict[str, Any] | None = None         # Error information if any
```

#### Accessing Results

```python
response = await prompt_run("template", text="input")

# Get the content directly
print(response.content)

# Get the structured message (includes metadata)
if response.message:
    print(f"Message ID: {response.message.message_id}")
    print(f"Role: {response.message.role}")
    for part in response.message.parts:
        if hasattr(part, 'text'):
            print(f"Text: {part.text}")

# Handle streaming responses
if response.streaming:
    async for chunk in response.content:
        print(chunk, end="")
```

## Template Structure

Templates are YAML files with multiple variants that are automatically selected based on conditions:

```yaml
# prompt_templates/code_agent_main.yaml
name: code_agent_main
version: latest
metadata:
  description: "Main coding assistant template"

variants:
  - text: |
      You are Claude, an AI coding assistant...
      {{ user_input }}
      {% if has_tools %}
      Available tools: {{ tool_names|join(', ') }}
      {% endif %}
    conditions:
      model_family: "claude"
      tool_support: true

  - text: |
      You are a helpful coding assistant...
      {{ user_input }}
      {% if has_tools %}
      Tools: {{ tool_descriptions|join('\n') }}
      {% endif %}
    conditions:
      model_family: "openai"
      tool_support: true

  - text: |
      Default coding assistant prompt...
      {{ user_input }}
    # No conditions = fallback variant
```

## Automatic Features

### Model Detection
The system automatically detects your LLM's capabilities:

```python
# These are automatically detected and passed to template conditions:
{
    "model": "gpt-4",
    "model_family": "openai", 
    "model_provider": "openai",
    "tool_support": True,
    "multimodal": False
}
```

### Tool Integration
Tools are automatically formatted for the template:

```python
# These variables are automatically added when tools are provided:
{
    "tools": [...],              # Full tool definitions
    "has_tools": True,           # Boolean flag
    "tool_names": ["read", "write"],  # List of tool names
    "tool_descriptions": ["read: Read files", "write: Write files"]
}
```

### History Context
Conversation history is automatically formatted:

```python
# These variables are automatically added when history is provided:
{
    "history": [...],            # A2A Message objects
    "has_history": True,         # Boolean flag
    "history_text": "user: Hello\nagent: Hi there!"  # Formatted text
}
```

## Complete Examples

### 1. Simple Text Processing

```python
from codin.prompt import prompt_run, set_endpoint

# Setup
set_endpoint("fs://./prompt_templates")

# Summarize text
response = await prompt_run(
    "summarize",
    text="Long article about AI developments...",
    max_length=200
)
print(response.content)
```

### 2. Code Assistant with Tools

```python
from codin.prompt import prompt_run, set_endpoint

# Setup
set_endpoint("fs://./prompt_templates")

# Define coding tools
tools = [
    {
        "name": "file_read",
        "description": "Read a file", 
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            }
        }
    },
    {
        "name": "file_write",
        "description": "Write to a file",
        "parameters": {
            "type": "object", 
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            }
        }
    }
]

# Get coding help
response = await prompt_run(
    "code_agent_main",
    tools=tools,
    conditions={"model_family": "claude"},
    user_input="Create a Python script to parse CSV files",
    sandbox_type="local",
    agent_name="CodeAgent"
)

print(response.content)
```

### 3. Conversational with History

```python
from codin.prompt import prompt_run, A2AMessage, A2ARole, A2ATextPart

# Build conversation history
history = [
    {
        "role": "user", 
        "content": "I'm learning Python"
    },
    {
        "role": "agent",
        "content": "Great! What would you like to learn about Python?"
    },
    {
        "role": "user", 
        "content": "How do I work with files?"
    }
]

# Continue conversation
response = await prompt_run(
    "python_tutor",
    history=history,
    user_input="Show me how to read a CSV file",
    difficulty="beginner"
)

print(response.content)
```

### 4. Streaming Response

```python
import asyncio
from codin.prompt import prompt_run

async def stream_story():
    response = await prompt_run(
        "creative_writer",
        stream=True,
        genre="science fiction",
        characters=["Alice", "Bob"],
        setting="Mars colony"
    )
    
    print("Streaming story:")
    async for chunk in response.content:
        print(chunk, end="", flush=True)
        await asyncio.sleep(0.01)  # Small delay for effect

# Run the streaming example
await stream_story()
```

## Best Practices

### 1. Use Specific Template Names
```python
# Good: Specific and descriptive
await prompt_run("code_review_python", code=code)

# Avoid: Too generic
await prompt_run("assistant", input=text)
```

### 2. Pass Context Variables
```python
# Good: Rich context
await prompt_run(
    "debug_code",
    code=code,
    error_message=error,
    language="python",
    framework="django"
)

# Avoid: Minimal context
await prompt_run("debug", text=code)
```

### 3. Use Conditions for Model-Specific Behavior
```python
# Good: Let template pick best variant
await prompt_run(
    "multi_model_template", 
    conditions={"model_family": "claude", "use_case": "coding"}
)

# Avoid: Hardcoded for one model
await prompt_run("claude_only_template")
```

### 4. Structure Tools Properly
```python
# Good: Complete tool definition
tools = [
    {
        "name": "search_web",
        "description": "Search the internet for current information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
]

# Avoid: Incomplete definitions
tools = [{"name": "search", "description": "Search"}]
```

### 5. Handle Errors Gracefully
```python
try:
    response = await prompt_run("template", text=input_text)
    if response.error:
        print(f"Error: {response.error}")
    else:
        print(response.content)
except Exception as e:
    print(f"Failed to run prompt: {e}")
```

## Error Handling

```python
# Check for errors in response
response = await prompt_run("template", text="input")
if response.error:
    print(f"Error: {response.error['message']}")
    print(f"Type: {response.error['type']}")
else:
    print(response.content)

# Handle exceptions
try:
    response = await prompt_run("nonexistent_template")
except ValueError as e:
    print(f"Template error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Usage

### Custom LLM Configuration
```python
from codin.prompt import create_prompt_engine
from codin.model import ModelRegistry

# Create custom LLM
llm = ModelRegistry.create_llm("gpt-4", temperature=0.7)

# Create engine with custom LLM
engine = create_prompt_engine(llm=llm)

# Use with custom engine
response = await engine.run("template", text="input")
```

### Template Debugging
```python
from codin.prompt import render_only

# Just render template without LLM execution
rendered = await render_only(
    "debug_template",
    variables={"debug": True},
    conditions={"model_family": "claude"}
)
print(f"Rendered template: {rendered}")
```

This guide covers the complete usage of `prompt_run` with the new a2a SDK integration. The system automatically handles LLM detection, template selection, and A2A protocol compliance, making it easy to build sophisticated AI applications. 