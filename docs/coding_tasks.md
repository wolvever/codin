# Sample Coding Tasks for codin CLI

This document lists a few example tasks you can run with the `codin` CLI to evaluate the `BaseAgent` and `BasePlanner` using your environment configuration.

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=claude-3-7-sonnet-20250219
export LLM_BASE_URL=https://aiproxy.usw.sealos.io/v1
export LLM_API_KEY=<your key>
```

After setting these variables you can execute tasks in quiet mode:

```bash
codin --debug --approval-mode never -q "Write a Python hello world script"
codin --debug --approval-mode never -q "Create a function that sums a list"
codin --debug --approval-mode never -q "Add a README section describing the project"
```

Each command runs the default BaseAgent + BasePlanner loop and prints the result. These small tasks are good sanity checks for prompt formatting and tool calls.

> **Note** Ensure network access to the configured base URL. If the MCP toolset fails to initialize or the API key is missing, CLI runs may fail.
