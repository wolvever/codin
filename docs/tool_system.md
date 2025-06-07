# Tool and Runtime System

Codin exposes tools to agents via a registry and executes them in sandboxed runtimes.

## Core Concepts

- **ToolRegistry** stores metadata and creates tool instances.
- **ToolExecutor** runs tools and handles streaming results.
- **Sandbox runtimes** isolate code execution (local, Docker, remote).

### Example: Reading a File

```python
from codin.tool import registry

file_content = await registry.exec("sandbox_read_file", path="README.md")
print(file_content)
```

### Automatic Truncation

Large files are truncated with metadata headers so LLMs do not exceed token limits.

### Runtime Backends

Runtimes support functions, CLIs, containers and remote endpoints. The default local runtime can be swapped for Docker or other backends.
