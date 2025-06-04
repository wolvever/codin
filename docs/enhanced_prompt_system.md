# Enhanced Prompt System

The enhanced prompt system in Codin provides a powerful and flexible way to manage prompts with support for:

1. **Template variables and model configuration**
2. **Sub-templates for different models, locales, and contexts**
3. **Local template storage in YAML format**
4. **Remote template fetching and caching**
5. **Global prompt execution via run() function**

## Core Components

### ModelOptions

Configure model-specific parameters for LLM generation:

```python
from codin.prompt.base import ModelOptions

options = ModelOptions(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1000,
    stop=["END"],
    presence_penalty=0.1,
    frequency_penalty=0.2
)
```

### SubTemplate

Create specialized templates for specific contexts with their own model and options:

```python
from codin.prompt.base import SubTemplate, ModelOptions

# Claude-specific template
claude_template = SubTemplate(
    text="You are Claude, an AI assistant by Anthropic. Task: {{ task }}",
    criteria={"model": "claude-3-5-sonnet-20241022"},
    model="claude-3-5-sonnet-20241022",
    model_options=ModelOptions(temperature=0.3, max_tokens=2000)
)

# Chinese locale template
chinese_template = SubTemplate(
    text="你是一个编程助手。任务：{{ task }}",
    criteria={"locale": "zh-CN"},
    model_options=ModelOptions(temperature=0.4)
)
```

### PromptTemplate

Main template with sub-template support:

```python
from codin.prompt.base import PromptTemplate, ModelOptions

# Create main template
prompt = PromptTemplate(
    name="code_assistant",
    text="You are a helpful coding assistant. Task: {{ task }}",
    model="gpt-4",
    model_options=ModelOptions(temperature=0.5),
    version="1.0"
)

# Add sub-templates
prompt.add_sub_template(claude_template)
prompt.add_sub_template(chinese_template)

# Render with criteria
result = prompt.render(
    criteria={"model": "claude-3-5-sonnet-20241022"}, 
    task="Write a Python function"
)
```

## YAML Template Format

Store templates in the `prompt_templates` directory using YAML format:

```yaml
# prompt_templates/code_assistant.yaml
name: code_assistant
version: "1.0"
model: gpt-4
model_options:
  temperature: 0.1
  max_tokens: 2000
  top_p: 0.9
text: |
  You are a helpful coding assistant.
  
  User request: {{ user_request }}
  
  {% if context %}
  Additional context: {{ context }}
  {% endif %}

sub_templates:
  - text: |
      You are Claude, a helpful coding assistant created by Anthropic.
      
      User request: {{ user_request }}
      
      {% if context %}
      Additional context: {{ context }}
      {% endif %}
    criteria:
      model: claude-3-5-sonnet-20241022
    model: claude-3-5-sonnet-20241022
    model_options:
      temperature: 0.05
      max_tokens: 4000

  - text: |
      你是一个编程助手，可以帮助用户编写、调试和理解代码。
      
      用户请求：{{ user_request }}
      
      {% if context %}
      其他上下文：{{ context }}
      {% endif %}
    criteria:
      locale: zh-CN
    model_options:
      temperature: 0.2
      max_tokens: 2000

metadata:
  author: "Codin Team"
  description: "A versatile coding assistant prompt"
  created_at: "2024-01-15"
  tags: ["coding", "assistant", "multi-model"]
```

## Registry Usage

### Loading Templates

```python
from codin.prompt.registry import PromptRegistry

# Get registry instance
registry = PromptRegistry.get_instance()

# Load all templates from prompt_templates directory
registry.load_all_local_templates()

# Register a template programmatically
template = PromptTemplate(name="test", text="Hello {{ name }}!")
registry.register(template)

# Get a template
template = registry.get("code_assistant")
template_with_version = registry.get("code_assistant", "1.0")
```

### Decorator Registration

```python
from codin.prompt.registry import PromptRegistry
from codin.prompt.base import ModelOptions

@PromptRegistry.prompt(
    "greeting", 
    version="v1", 
    model="gpt-4",
    model_options=ModelOptions(temperature=0.7)
)
def greeting_prompt():
    return "Hello, {{ name }}! How can I help you today?"
```

## Global Prompt Execution

Use the global `run()` function to execute prompts - this is the recommended way for agents:

```python
from codin.prompt.run import run

# Run prompt with sub-template selection
result = await run(
    "code_assistant",
    criteria={"model": "claude-3-5-sonnet-20241022"},
    user_request="Help me debug this code",
    context="Python function with error"
)

# Run with different criteria
result = await run(
    "code_assistant",
    criteria={"locale": "zh-CN"},
    user_request="编写Python函数"
)

# Override the model specified in template
result = await run(
    "code_assistant", 
    model_override="gpt-4-turbo",
    user_request="Write a function"
)
```

## Agent Integration

Agents now use the global `run()` function instead of managing prompts themselves:

```python
from codin.agent.code_agent import CodeAgent
from codin.prompt.run import run

# Create agent - automatically registers default prompts
agent = CodeAgent()

# Use built-in methods that leverage prompts
response = await agent.review_code("def test(): pass", "python")
response = await agent.debug_code(
    "def test(): pass", 
    "Function doesn't work", 
    "Should return True"
)

# Use run() directly in agent methods
class MyAgent(Agent):
    async def generate_response(self, query: str) -> str:
        return await run(
            "my_prompt_template",
            criteria={"model": "gpt-4"},
            query=query
        )
```

## Engine Usage (Optional)

For more control, you can use PromptEngine directly:

```python
from codin.prompt.engine import PromptEngine

# Create engine (can specify default LLM)
engine = PromptEngine("gpt-4")

# Run prompt with sub-template selection
result = await engine.run(
    "code_assistant",
    criteria={"model": "claude-3-5-sonnet-20241022"},
    user_request="Help me debug this code",
    context="Python function with error"
)

# Just render without executing
rendered = await engine.render_only(
    "code_assistant",
    criteria={"locale": "zh-CN"},
    user_request="编写Python函数"
)
```

## Environment Configuration

Configure the prompt system using environment variables:

```bash
# Local or remote mode
export PROMPT_RUN_MODE=local  # or "remote"

# Local template directory
export PROMPT_TEMPLATE_DIR=./prompt_templates

# Remote base URL for fetching templates
export PROMPT_REMOTE_BASE_URL=http://localhost:8080
```

## Remote Template Support

The system supports fetching templates from a remote server:

```python
# Set remote mode
from codin.prompt.run import set_run_mode
set_run_mode("remote")

# Templates will be fetched from PROMPT_REMOTE_BASE_URL
result = await run("code_assistant", query="Hello")  # Fetches from remote if not cached
```

Expected remote API endpoint: `GET /v1/prompt_template/{name}?version={version}`

## Best Practices

1. **Use the global run() function** for prompt execution in agents
2. **Use descriptive names** for templates and sub-templates
3. **Version your templates** for reproducible behavior
4. **Organize templates by domain** (coding, writing, analysis, etc.)
5. **Use sub-templates with criteria** for model-specific optimizations
6. **Set appropriate model options** for each use case and sub-template
7. **Test templates** with different variable combinations
8. **Document template variables** in metadata

## Migration from Previous System

### Agent Changes

**Before:**
```python
class MyAgent(Agent):
    def __init__(self, prompts: dict[str, PromptTemplate]):
        super().__init__(prompts=prompts)
    
    async def my_method(self):
        return await self.execute_prompt("my_prompt", var="value")
```

**After:**
```python
class MyAgent(Agent):
    def __init__(self):
        super().__init__()
        # Register prompts in the global registry during init
        self._setup_prompts()
    
    async def my_method(self):
        return await run("my_prompt", var="value")
```

### SubTemplate Changes

**Before:**
```yaml
sub_templates:
  - text: "Template for Claude"
    metadata:
      model: claude-3
    model_options:
      temperature: 0.5
```

**After:**
```yaml
sub_templates:
  - text: "Template for Claude"
    criteria:
      model: claude-3-5-sonnet-20241022
    model: claude-3-5-sonnet-20241022
    model_options:
      temperature: 0.5
```

## Examples

See the test files in `tests/prompt/test_enhanced_prompt_system.py` for comprehensive examples of all features. 