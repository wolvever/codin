
# Agent2Agent (A2A) Protocol: Project Analysis Report

## 1. Overview

**Agent2Agent (A2A)** is an open protocol designed to enable communication and interoperability between opaque agentic applications. It allows AI agents, built on diverse frameworks and running on separate servers, to communicate and collaborate as agents, not just as tools. The protocol is framework-agnostic and supports both Python and JavaScript/TypeScript ecosystems.

---

## 2. Core Concepts

- **AgentCard**: A JSON object describing an agent's capabilities, skills, connection info, and authentication requirements. Used for agent discovery and negotiation.
- **Skills**: Each agent exposes a set of skills (capabilities) with metadata, input/output modes, and examples.
- **Standardized Communication**: Uses JSON-RPC 2.0 over HTTP(S) for all agent-to-agent interactions.
- **Task Lifecycle**: Agents can negotiate, delegate, and manage long-running tasks.
- **Security & Opacity**: Agents collaborate without exposing internal state, memory, or proprietary tools.

#### Example AgentCard (JSON)
```json
{
  "name": "Reimbursement Agent",
  "description": "Handles reimbursement process for employees.",
  "url": "http://localhost:10002/",
  "version": "1.0.0",
  "capabilities": { "streaming": true },
  "skills": [
    {
      "id": "process_reimbursement",
      "name": "Process Reimbursement Tool",
      "description": "Helps with the reimbursement process."
    }
  ]
}
```

---

## 3. Getting Started: Code Examples

### Python Example (Expense Reimbursement Agent)
- Navigate to `samples/python/agents/google_adk`.
- Set up your environment and API key:
  ```bash
  echo "GOOGLE_API_KEY=your_api_key_here" > .env
  uv run .
  ```
- Start a CLI client to interact with the agent:
  ```bash
  cd samples/python/hosts/cli
  uv run . --agent http://localhost:10002
  ```

#### Minimal Python Example: Starting an A2A Agent
```python
from agent import ReimbursementAgent
from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill
from task_manager import AgentTaskManager

capabilities = AgentCapabilities(streaming=True)
skill = AgentSkill(
    id='process_reimbursement',
    name='Process Reimbursement Tool',
    description='Helps with the reimbursement process.',
)
agent_card = AgentCard(
    name='Reimbursement Agent',
    url='http://localhost:10002/',
    version='1.0.0',
    capabilities=capabilities,
    skills=[skill],
)
server = A2AServer(
    agent_card=agent_card,
    task_manager=AgentTaskManager(agent=ReimbursementAgent()),
    host='localhost',
    port=10002,
)
server.start()
```

### Multi-Agent Web Demo
- Navigate to `demo/ui` and run:
  ```bash
  uv run main.py
  ```
- Add remote agents via the UI by specifying their AgentCard address (e.g., `localhost:10002`).
- Interact with multiple agents and observe orchestration and delegation.

### JavaScript Example
- See `samples/js/src/agents/README.md` for Genkit-based agents (e.g., Movie Info Agent, Coder Agent).

#### Minimal JS Example: Registering a Genkit Agent
```js
const { Agent } = require('a2a-sdk');
const movieAgent = new Agent({
  name: 'Movie Info Agent',
  skills: [/* ... */],
  // ...
});
movieAgent.listen(10010);
```

---

## 4. TaskManager: Architecture, Responsibilities, and Implementation

The `TaskManager` is a core component that enables protocol-compliant, modular, and extensible agent servers.

- **Role:** Handles protocol-level logic, task lifecycle management, state/history/artifact tracking, streaming, and push notifications. It is responsible for interfacing with the A2A protocol and managing the flow of tasks.
- **Responsibilities:**
  - Protocol method handlers: `on_send_task`, `on_send_task_subscribe`, `on_get_task`, etc.
  - Task state transitions and storage (in-memory or persistent)
  - Streaming and async support (detailed below)
  - Push notification orchestration
  - Validation of protocol requests
  - Invoking the Agent for business logic and handling its results

#### Streaming and Async Support
- The method `on_send_task_subscribe` enables streaming responses for long-running or interactive tasks.
- Uses asynchronous generators and Server-Sent Events (SSE) to push incremental updates to clients.
- Maintains per-task SSE subscriber queues to manage multiple clients.

```python
async def on_send_task_subscribe(self, request: SendTaskStreamingRequest) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
    # Implementation in concrete TaskManager (see AgentTaskManager, InMemoryTaskManager)
    ...

async def setup_sse_consumer(self, task_id: str, is_resubscribe: bool = False):
    async with self.subscriber_lock:
        if task_id not in self.task_sse_subscribers:
            if is_resubscribe:
                raise ValueError('Task not found for resubscription')
            self.task_sse_subscribers[task_id] = []
        sse_event_queue = asyncio.Queue(maxsize=0)
        self.task_sse_subscribers[task_id].append(sse_event_queue)
        return sse_event_queue
```

#### Push Notification Support
- Clients can register a webhook or notification endpoint for a task using `on_set_task_push_notification`.
- The TaskManager stores notification configs per task and can later retrieve them to send updates.

#### State, History, and Artifacts Management
- All tasks are tracked in an in-memory dictionary (`self.tasks`).
- State transitions, message history, and artifacts are updated as tasks progress.
- Methods like `update_store` and `append_task_history` manage these updates.

#### Agent Interface Requirements for TaskManager Compatibility

To work with a TaskManager, an agent **should** implement a minimal interface:
- `invoke(self, query, session_id)` (sync or async)
- `async stream(self, query, session_id)`
- `SUPPORTED_CONTENT_TYPES` (class attribute)

```python
class AgentWithTaskManager(ABC):
    @abstractmethod
    def get_processing_message(self) -> str:
        pass
    def invoke(self, query, session_id) -> str:
        ...
    async def stream(self, query, session_id) -> AsyncIterable[dict[str, Any]]:
        ...
```

#### How TaskManager Uses the Interface
- The TaskManager calls `agent.invoke(...)` for synchronous tasks and `await agent.stream(...)` for streaming tasks.
- It checks `agent.SUPPORTED_CONTENT_TYPES` to validate protocol compatibility.

#### Best Practices
- Keep protocol and state management in TaskManager; keep business logic and LLM/tool use in Agent.
- This separation makes it easy to swap out agent logic, add new skills, or change storage/notification backends without rewriting protocol code.
- For advanced workflows (multi-agent, persistent state, custom streaming), extend TaskManager or Agent as needed, but keep their responsibilities clear and focused.

---

## 5. Agent Design and Implementation

- **Role:** Encapsulates the business logic, tool invocation, and LLM (or other backend) interaction. Responsible for actually performing the work required by a task.
- **Responsibilities:**
  - The core logic for handling a user query (e.g., LLM prompt, tool use, business rules)
  - Tool/plugin registration and invocation
  - Session/memory management (if needed for multi-turn)
  - Formatting the response for the TaskManager
  - Supporting streaming and async by implementing the required methods

#### Example: Google ADK Agent
```python
class ReimbursementAgent(AgentWithTaskManager):
    def _build_agent(self) -> LlmAgent:
        return LlmAgent(
            model='gemini-2.0-flash-001',
            ...,
            tools=[create_request_form, reimburse, return_form],
        )
```

#### Example: LangGraph Agent
```python
class CurrencyAgent:
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
    def invoke(self, query, sessionId) -> str:
        ...
    async def stream(self, query, sessionId) -> AsyncIterable[dict[str, Any]]:
        ...
```

#### Example: Semantic Kernel Agent
```python
class SemanticKernelTravelAgent:
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
    async def invoke(self, user_input: str, session_id: str) -> dict[str, Any]:
        ...
    async def stream(self, user_input: str, session_id: str) -> AsyncIterable[dict[str, Any]]:
        ...
```

#### How Agents Support Streaming and Async
- Implement the `stream` method to yield incremental results or status updates.
- Use async generators or similar constructs to support real-time updates.

#### How to Add Tools/Skills and Change LLM Backend
```python
new_skill = AgentSkill(
    id='currency_conversion',
    name='Currency Conversion',
    description='Converts amounts between currencies.'
)
agent_card.skills.append(new_skill)

# Change LLM backend
agent = LlmAgent(model='your-llm-model', ...)
```

#### Best Practices
- Keep business logic, tool/LLM integration, and session management in the Agent.
- Make the agent modular and extensible for new skills and backends.

---

## 6. Single-Agent and Multi-Agent Patterns

- **Single-Agent:** Each agent can run as a standalone A2A server, exposing its skills and capabilities.
- **Multi-Agent:** Host agents can orchestrate multiple remote agents, delegating tasks based on skills and capabilities.
- **Agent Discovery and Orchestration:** Agents can discover each other using AgentCards and delegate tasks accordingly.

#### Multi-Agent Orchestration Example
```python
class HostAgent:
    def __init__(self, remote_agent_addresses: list[str], ...):
        ...
        for address in remote_agent_addresses:
            card_resolver = A2ACardResolver(address)
            card = card_resolver.get_agent_card()
            remote_connection = RemoteAgentConnections(card)
            self.remote_agent_connections[card.name] = remote_connection
            self.cards[card.name] = card
```

---

## 7. Extensibility

- How to extend the system (add skills, new agents, new backends, etc.).
- Schema-driven extensibility.
- Enterprise-Ready features.

#### Adding a New Skill to an Agent
```python
new_skill = AgentSkill(
    id='currency_conversion',
    name='Currency Conversion',
    description='Converts amounts between currencies.'
)
agent_card.skills.append(new_skill)
```

---

## 8. Summary Table

| Aspect                | Details                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| **Getting Started**   | Python, JS, CLI, and Web UI samples; easy local setup                   |
| **Core Concepts**     | AgentCard, Skills, JSON-RPC, Task Lifecycle, Opacity                    |
| **Capabilities**      | Discovery, Orchestration, Streaming, Forms, Images, Security            |
| **Extensibility**     | Framework-agnostic, pluggable skills, schema-driven, host/client apps   |
| **Single/Multi-Agent**| Supports both; multi-agent orchestration via host agents                |
| **Modularity**        | Agents as modular services, skill-based, composable orchestration        |

---

## 9. References
- [A2A Protocol Documentation](https://google.github.io/A2A/)
- [A2A Protocol Specification](https://google.github.io/A2A/specification/)
- [Sample Agents](samples/python/agents/README.md)
- [Demo Web App](demo/README.md) 


# FAQ



## 1  What the protocol calls a **Message** vs. a **Task**

| Concept     | What it is                                                                                                                                                                                                           | Typical lifetime                          | Primary use-case                                                                                            |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Message** | A single conversational turn that carries one or more `Part`s (text, file, JSON, etc.) together with minimal metadata.                                                                                               | Milliseconds ➜ a few seconds.             | Prompts, quick clarifications, follow-ups, or incremental user input.                                       |
| **Task**    | A state-ful unit of work with its own `id`, `status`, `artifacts[]`, and optional message history. It progresses through states such as `submitted → working → completed/failed` and can pause for `input-required`. | Seconds ➜ minutes ➜ hours (long-running). | Anything that needs streaming progress, resumability, cancellation, push notifications, or large artifacts. |

*Definitions are formalised in §6.1 “Task object” and §6.4 “Message object” of the spec* ([Google GitHub][1], [Google GitHub][1])
*A concise summary appears again in the “Key Concepts” guide* ([Google GitHub][2])

---

## 2  RPC methods exposed by the current SDK

| Method                                                                             | What you send                            | What you get back                                                                                              | Status                                                                                                   |
| ---------------------------------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `message/send`                                                                     | **Message**                              | Either another `Message` **or** a fresh `Task` (if the server decides the request should become long-running). | **Preferred since spec 0.2.x** ([GitHub][3])                                                             |
| `message/stream`                                                                   | **Message** + automatic SSE subscription | A stream of `Task / TaskStatusUpdateEvent / TaskArtifactUpdateEvent`                                           | Preferred when you want real-time tokens or progress bars ([Google GitHub][4])                           |
| `tasks/get`, `tasks/resubscribe`, `tasks/cancel`, `tasks/pushNotificationConfig/*` | **Task ID**                              | Updated `Task` or confirmation                                                                                 | Manage an existing task                                                                                  |
| `tasks/send`, `tasks/sendSubscribe`                                                | **Task-shaped payload**                  | `Task` or SSE stream                                                                                           | **Deprecated**—kept only for backwards compatibility; use `message/*` instead ([GitHub][3], [Medium][5]) |

> **Why the change?**
> Earlier drafts (≤0.1.x) had you explicitly create a task with `tasks/send`.
> From spec 0.2.0 forward the entry point is always a *message*; if the agent concludes it needs a long-running job it simply responds with a `Task`, so clients never have to guess up-front whether something will be “quick” or “long”. ([Hugging Face][6])

---

## 3  When to send a **Message**

Send `message/send` (or `message/stream`) when:

1. **Turn-based chat** – you expect a direct answer in one or two turns (“What’s 3 × 7?”).
2. **Incremental dialogue** – you’re already inside a task and need to provide the next user reply (`taskId` can be set on the `Message`).
3. **Latency matters** – you want token-level streaming back immediately; `message/stream` keeps the connection open and the server pushes events as it works ([Google GitHub][4]).
4. **You don’t care about later polling** – the result can fit in a single HTTP response.

In these scenarios the response may be:

* another `Message` (fast path), *or*
* a short `Task` in terminal state (the agent wrapped up in one shot).

Either way you haven’t had to manage a task lifecycle yourself.

---

## 4  When to send (or continue working with) a **Task**

Choose the task workflow when any of these are true:

| Need                              | Why the Task abstraction helps                                                                                                    |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Minutes-to-hours runtime**      | Task states (`working`, `input-required`, `auth-required`, `completed`, etc.) keep everything resumable across disconnections.    |
| **Progress bars / token streams** | Subscribe with `message/stream` or re-attach later using `tasks/resubscribe`.                                                     |
| **Large or binary outputs**       | Artifacts can be chunked and streamed without re-sending the whole payload each turn.                                             |
| **Push notifications / Webhooks** | Configure once via `tasks/pushNotificationConfig/set`; the server pings your webhook on major state changes ([Google GitHub][4]). |
| **Cancellation**                  | `tasks/cancel` is available while the task is still `working`.                                                                    |
| **Audit / trace**                 | The server can persist history inside `Task.history[]` for compliance or debugging ([Google GitHub][1]).                          |

Even if you *started* with a `message/send`, the moment the server replies with a `Task` you switch to the task methods (`tasks/get`, `tasks/resubscribe`, etc.) for the remainder of the conversation.

---

## 5  Rule of thumb for client code

```python
# pseudo-code
reply = client.message_send(Message(role="user", parts=[TextPart(text=prompt)]))

if isinstance(reply, Task):          # long-running job detected
    task_id = reply.id
    # Option A – stay connected:
    for event in client.message_stream(task_id, resume=True):
        handle(event)
    # Option B – disconnect and poll later:
    while not reply.status.state in TERMINAL:
        time.sleep(5)
        reply = client.tasks_get(task_id)
else:
    print("Quick answer:", reply.parts[0].text)
```

---

## 6  Further reading & examples

* “Understanding A2A — The Protocol for Agent Collaboration” (Google Cloud Comm.) ([Google Cloud Community][7])
* “A2A Deep Dive: Getting Real-Time Updates from AI Agents” (Medium) ([Medium][5])
* Official Python quick-start: `python-a2a` tutorial section “Send Messages to an Agent” ([A2A Protocol][8])
* LangGraph demo building an A2A currency agent (dev.to) ([DEV Community][9])

---

### TL;DR

* **Send a *Message* first** – it’s the modern, spec-compliant entry point.
* **Let the server decide** whether a lightweight reply or a full-blown *Task* is appropriate.
* **Work with tasks only once you receive them**—that’s when you gain progress tracking, streaming, push notifications, cancellation, and artifact handling.

[1]: https://google.github.io/A2A/specification/ "Specification - Agent2Agent Protocol (A2A)"
[2]: https://google.github.io/A2A/topics/key-concepts/ "Key Concepts - Agent2Agent Protocol (A2A)"
[3]: https://github.com/google/A2A/issues/578 "Different tasks/send  and message/send? · Issue #578 · google/A2A · GitHub"
[4]: https://google.github.io/A2A/topics/streaming-and-async/ "Streaming & Asynchronous Operations - Agent2Agent Protocol (A2A)"
[5]: https://medium.com/google-cloud/a2a-deep-dive-getting-real-time-updates-from-ai-agents-a28d60317332?utm_source=chatgpt.com "A2A Deep Dive: Getting Real-Time Updates from AI Agents - Medium"
[6]: https://huggingface.co/blog/lynn-mikami/agent2agent?utm_source=chatgpt.com "What is The Agent2Agent Protocol (A2A) and Why You Must Learn It ..."
[7]: https://www.googlecloudcommunity.com/gc/Community-Blogs/Understanding-A2A-The-Protocol-for-Agent-Collaboration/ba-p/906323?utm_source=chatgpt.com "Understanding A2A — The Protocol for Agent Collaboration"
[8]: https://a2aprotocol.ai/blog/python-a2a-tutorial-20250513?utm_source=chatgpt.com "Python A2A Tutorial 20250513 - A2A Protocol"
[9]: https://dev.to/czmilo/building-an-a2a-currency-agent-with-langgraph-5c24?utm_source=chatgpt.com "Building an A2A Currency Agent with LangGraph - DEV Community"
