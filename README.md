# Codin - AI Coding Agent Framework

A modern, extensible framework for building AI coding agents with support for multiple LLM providers, tool systems, and execution environments.

## Features

- **Multi-LLM Support**: OpenAI, Anthropic, Google Gemini, and more
- **Tool System**: Extensible tool registry with built-in coding tools
- **Sandbox Environments**: Local and containerized execution
- **Memory Management**: Conversation history and context management
- **Agent Architecture**: Modular agent design with planners and executors
- **Plan-Execute Agent**: Simple planner that generates a plan then executes each step
- **A2A Protocol**: Compatible with Agent-to-Agent communication standards

## Quick Start

### Hello World Example

The framework includes a complete hello world example demonstrating the BaseAgent + CodePlanner architecture:

```bash
# Run with mock LLM (no API keys required)
python examples/hello_world_test.py

# Run with real LLM (requires API keys)
python examples/hello_world.py
```

This example demonstrates:
- **BaseAgent + CodePlanner** iterative execution loop
- Tool execution (file creation, shell commands)
- Task completion logic
- Memory and state management

### Agent Logic Comparison

The framework implements the same core iterative loop logic as other coding agents:

| Component | BaseAgent + CodePlanner | CodeAgent | Codex.rs |
|-----------|------------------------|-----------|----------|
| **Loop Driver** | Planner step generation | Tool call presence | Function call presence |
| **Completion Logic** | FinishStep from planner | should_continue flag + heuristics | Empty function call list |
| **State Management** | Immutable State + Memory service | Mutable instance variables | Mutex-protected state |
| **Tool Execution** | Step-based via ToolCallStep | Turn-based with results | Stream-based function calls |

See `docs/agent_architecture.md` for a deeper explanation of the agent loop and state handling.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd codin

# Install with uv (recommended)
uv sync

# Or install with pip (development mode)
pip install -e .[dev]
```

## Environment Setup

Copy `.env-example` to `.env` and configure your API keys:

```bash
cp .env-example .env
# Edit .env with your API keys
```

## Usage

### Basic Agent Setup

```python
from codin.agent.base_agent import BaseAgent
from codin.agent.code_planner import CodePlanner, CodePlannerConfig
from codin.tool.registry import ToolRegistry
from codin.tool.sandbox import SandboxToolset
from codin.sandbox.local import LocalSandbox

# Initialize components
sandbox = LocalSandbox()
await sandbox.up()

tool_registry = ToolRegistry()
sandbox_toolset = SandboxToolset(sandbox)
await sandbox_toolset.up()
tool_registry.register_toolset(sandbox_toolset)

# Create planner and agent
planner_config = CodePlannerConfig(model="gpt-4")
planner = CodePlanner(config=planner_config, tool_registry=tool_registry)

agent = BaseAgent(
    name="CodingAgent",
    planner=planner,
    tools=tool_registry.get_tools()
)

# Execute task
from codin.agent.types import AgentRunInput, Message
from codin.agent.types import Role, TextPart

task_message = Message(
    messageId="task-1",
    role=Role.user,
    parts=[TextPart(text="Create a Python hello world script")],
    contextId="session-1",
    kind="message"
)

agent_input = AgentRunInput(
    session_id="session-1",
    message=task_message
)

async for output in agent.run(agent_input):
    print(f"Agent output: {output}")
```

### Debug Sandbox

Run commands under the same sandbox used by Codin. This mirrors the Rust debug helpers.

```bash
codin debug-sandbox --full-auto echo "hello sandbox"
codin debug-sandbox -s disk-write-cwd -s network-full-access -- python script.py
```

Use `-s/--sandbox-permission` multiple times to customize the policy or `--full-auto` for a permissive default.

## Architecture

The framework follows a modular architecture:

- **Agents**: High-level orchestrators (BaseAgent)
- **Planners**: LLM-based step generation (CodePlanner)
- **Tools**: Extensible tool system with registries
- **Sandbox**: Execution environments (Local, Docker)
- **Memory**: Conversation and context management
- **Models**: LLM abstraction layer

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Submit a pull request

## License

[License information] 
