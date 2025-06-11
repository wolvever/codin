# Omni-Agent Platform – High-Level Design

## 1. Purpose
A unified, extensible platform that allows heterogeneous AI agents (ADK, LangGraph, CrewAI, Agno etc.) to collaborate via the A2A protocol, share tools, and execute code in secure sandboxes across diverse runtimes (local, Docker/K8s, Ray, remote Daytona/E2B).

## 2. Guiding Principles
1. **Protocol-first** – all communication uses A2A JSON-RPC 2.0 over HTTP / SSE.
2. **Framework-agnostic** – a common `Agent` interface wraps concrete agent implementations from any library.
3. **Omni-Runtime** – a pluggable execution backend capable of running *functions, classes, CLIs, containers or remote endpoints* synchronously or asynchronously with streaming.
4. **Separation of Concerns** – `TaskManager` handles lifecycle & events; `MemorySystem` persists context; `ToolSystem` provides discovery & execution; `Runtime` performs code execution.
5. **Self-hosting & Self-evolving** – agents can generate new agents/tools that are auto-registered & deployed.

## 3. Component Overview
| Layer | Component | Responsibility |
|-------|-----------|----------------|
| API   | A2A Gateway | Auth, routing, JSON-RPC → internal bus |
| Orchestration | **TaskManager** | Task decomposition, event handling, agent scheduling |
| Agents | **Agent** | Uniform interface over ADK, LangGraph, etc. |
| Tools  | **ToolRegistry / ToolExecutor** | Store metadata, run OpenAPI, code, MCP, … via Runtime |
| Runtime | **OmniRuntime** | Execute workloads on Local, Docker, K8s, Ray, Daytona, Remote |
| Storage | **MemorySystem** | Chat history, task/event logs, embeddings |
| Security | Sandbox | Limits CPU, memory, network, FS access |

## 4. Key Interfaces (Python typestubs)
```python
class Agent(Protocol):
    id: str
    name: str
    description: str
    tools: list[Tool]
    async def handle(self, task: Task) -> AgentResponse: ...
```
```python
class Tool(Protocol):
    name: str
    spec: ToolSpec  # OpenAPI / fn-sig / other
    async def invoke(self, *args, **kwargs) -> Any: ...
```
```python
class Runtime(ABC):
    async def run(self, workload: Workload, /, *, stream: bool = False) -> RuntimeResult: ...
```
## 5. OmniRuntime Execution Matrix
| Workload Type | Local | Docker/K8s | Ray | Remote HTTP | Daytona |
|---------------|-------|-----------|-----|-------------|---------|
| Function | ✔ | ✔ | ✔ | N/A | N/A |
| Module/Class | ✔ | ✔ | ✔ | N/A | N/A |
| CLI Command | ✔ | ✔ | ✖ | N/A | N/A |
| Container | ✖ | ✔ | ✖ | N/A | ✔ |
| Endpoint Call | ✔(requests) | ✔ | ✔ | ✔ | N/A |

## 6. Sequence: User → System
1. User issues natural-language goal via UI.
2. A2A Gateway creates `Task` → passes to `TaskManager`.
3. `TaskManager` consults `ToolRegistry` & plans subtasks.
4. Subtasks dispatched to Agents (via A2A). Each Agent may invoke Tools ➜ `ToolExecutor`.
5. `ToolExecutor` selects proper `Runtime` backend & executes workload.
6. Results stream back (SSE) through Gateway to UI; `MemorySystem` logs everything.

## 7. Self-Evolution Flow
* Agents can emit `NewAgent` / `NewTool` events with code or OpenAPI spec.
* Gateway validates, stores artefacts, builds container (if needed), registers in `ToolRegistry/AgentRegistry`, and deploys via OmniRuntime.

## 8. Security Model
* All tool/agent executions go through Sandbox‐enabled runtimes.
* RBAC on Tool access.
* Signed AgentCards + JWT between services.

## 9. Roadmap Snippet
* MVP 0.1: Local Runtime + ToolRegistry + single ADK-wrapped Agent.
* 0.2: Docker Runtime & Sandbox; TaskManager with simple planner.
* 0.3: A2A Gateway + streaming; MemorySystem (Redis) + pgvector.
* 0.4: Self-evolution pipeline; remote Runtime (Daytona).

## 10. Sub-System Boundaries

Below are the five top-level sub-systems and their primary responsibilities.  Each maps directly to a Python package in the codebase (e.g. `codin/agent`, `codin/tool`, …).

| Sub-System | Package | Responsibility |
|------------|---------|----------------|
| **AgentSystem** | `codin.agent` | Framework-agnostic `Agent` interface, wrappers for ADK/LangGraph/CrewAI, multi-agent coordination (Planner, Router, etc.). |
| **ToolSystem** | `codin.tool` | `Tool` abstraction, `ToolRegistry`, `ToolExecutor`, adapters for OpenAPI and Python functions. |
| **Sandbox** | `codin.sandbox` | Secure, isolated execution layer for code & shell (local, Docker, Daytona, E2B). Provides FS access abstraction. |
| **MemorySystem** | `codin.memory` | Persistent conversation history, short-term "note" store, vector search over memory chunks (pluggable backends: in-mem, Redis, pgvector). |
| **OmniRuntime** | `codin.runtime` | Execution orchestration of functions, CLI, containers, endpoints, and distributed jobs across local, Docker/K8s, Ray, etc. |

Each sub-system is internally cohesive and externally communicates through well-defined interfaces (Python protocols / ABCs) and—when crossing service boundaries—via the A2A protocol.


