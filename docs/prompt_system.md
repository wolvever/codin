# Prompt System

This document describes how Codin manages prompts and how to run them using `prompt_run`.

## Features

- Template variables and model configuration
- Sub-templates for different models, locales and contexts
- Local template storage in YAML format
- Remote template fetching and caching
- Global execution via `prompt_run`

## Quick Start

Set where your prompt templates are stored:

```python
from codin.prompt import set_endpoint

# Local filesystem
set_endpoint("fs://./prompt_templates")

# Remote server
set_endpoint("http://your-server.com/templates")
```

Execute a prompt directly:

```python
from codin.prompt import prompt_run

response = await prompt_run("welcome", user="Jane")
print(response.text)
```

### Sub-templates

Templates can include variations that are automatically selected based on model or other criteria. Example:

```yaml
sub_templates:
  - text: "Template for Claude"
    criteria:
      model: claude-3-5-sonnet-20241022
    model: claude-3-5-sonnet-20241022
    model_options:
      temperature: 0.5
```

See `tests/prompt/test_enhanced_prompt_system.py` for comprehensive examples.
