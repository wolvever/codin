Below is a self-contained design document that ties four moving parts into one coherent system:

* **A2A protocol** – the wire format and streaming rules
* **Mailbox abstraction** – pluggable local / Ray transports with back-pressure
* **Connector layer** – translates the agent’s internal `AgentEvent`s into first-class A2A frames
* **Agent** – pure async coroutine that yields tokens and events, oblivious to transport or protocol

---

## Executive summary

The design uses A2A’s native JSON-RPC + SSE envelope for every hop, so any UI or third-party agent that already “speaks A2A” can subscribe without adapters.
On a laptop the mailbox is a bounded `asyncio.Queue`; in a cluster it is a distributed `ray.util.queue.Queue`.
Because both queues share an identical async API, the Connector—the only place that knows A2A—runs unchanged in either mode.
Back-pressure is enforced hop-by-hop, Control-Aware priority queues keep cancellations responsive, and optional event-sourcing journals give at-least-once durability and replay.

---

## 1 Architecture at a glance

```
 ┌─────────────┐       tokens/events        ┌──────────────┐  A2A JSON frames  ┌──────────────┐
 │  LLM Agent  │ ─────────────────────────▶ │  Connector   │ ─────────────────▶│   Mailbox    │
 │ (pure async)│  AsyncIterator[str|Event]  │ (A2A glue)   │    publish()      │  (local/Ray) │
 └─────────────┘                            └──────────────┘◀──────────────────┘
                                                  subscribe()
```

* **Agent** yields text deltas and internal `AgentEvent`s (`tool_call_start`, `tool_call_end`, …).
* **Connector** converts those events to A2A frames and calls `mailbox.publish(evt)`.
* **Mailbox** is either `AsyncioMailbox` (in-proc) or `RayMailbox` (cluster).
* Subscribers—CLI, web UI, other agents—consume an SSE stream produced directly from the mailbox.

---

## 2 A2A protocol essentials

### 2.1 Transport & framing

* All messages are JSON-RPC 2.0 envelopes streamed over **Server-Sent Events** (`tasks/sendSubscribe`) ([medium.com][1], [googlecloudcommunity.com][2]).
* SSE runs on HTTP/2, giving ordered delivery plus built-in flow-control ([medium.com][3], [developer.mozilla.org][4]).

### 2.2 Streaming content

| A2A event                     | Purpose                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| **`TaskMessageDeltaEvent`**   | Carries incremental `parts`; last delta sets `"final":true` ([medium.com][1])            |
| **`TaskStatusUpdateEvent`**   | Lifecycle signal: `WORKING`, `COMPLETED`, `FAILED`, … ([medium.com][1], [github.com][5]) |
| **`TaskArtifactUpdateEvent`** | Announces large artefacts by URI (optional) ([a2aprotocol.ai][6])                        |

The protocol purposefully avoids ad-hoc labels; the `event` discriminator plus optional `stage` metadata are enough for every consumer ([developers.googleblog.com][7]).

---

## 3 Mailbox pattern

### 3.1 Interface

```python
class Mailbox(Protocol):
    async def publish(self, evt: dict) -> None: ...
    async def subscribe(self, task_id: str) -> AsyncIterator[dict]: ...
```

### 3.2 `AsyncioMailbox` – local dev

* Wraps `asyncio.Queue(maxsize=N)`; `put()` blocks when the queue is full, giving natural back-pressure ([discuss.python.org][8]).

### 3.3 `RayMailbox` – distributed

```python
@ray.remote
class _Q:           # FIFO queue actor on the cluster
    def __init__(self, cap=0): self.q = Queue(cap)       # ray.util.queue.Queue
    async def put(self,e): await self.q.put_async(e)
    async def pull(self):   return await self.q.get_async()
```

`ray.util.queue.Queue` deliberately mimics `asyncio.Queue`, including blocking semantics when full ([docs.ray.io][9], [docs.ray.io][10]).

### 3.4 Priority & control messages

If you need `CancelTask` to leap-frog a flood of token deltas, wrap either queue with a **Control-Aware Mailbox** (pattern borrowed from Akka `UnboundedControlAwareMailbox`) ([doc.akka.io][11], [doc.akka.io][12]).

---

## 4 Connector layer

### 4.1 Event mapping

| `AgentEvent`            | A2A frame                              | Notes                          |
| ----------------------- | -------------------------------------- | ------------------------------ |
| `tool_call_start / end` | `TaskMessageDeltaEvent` with JSON part | `"final":false`                |
| `llm_token`             | `TaskMessageDeltaEvent` with text part | tokens streamed as they arrive |
| `turn_start`            | `TaskStatusUpdateEvent` → `WORKING`    | additional `stage:"turn"` meta |
| `task_complete`         | `TaskStatusUpdateEvent` → `COMPLETED`  |                                |
| `task_error`            | `TaskStatusUpdateEvent` → `FAILED`     | attach stack trace             |

Sample delta frame produced by Connector:

```json
{
 "jsonrpc":"2.0",
 "result":{
   "event":"TaskMessageDeltaEvent",
   "task_id":"t_9",
   "message_delta":{
      "id":"d_42",
      "parts":[{"type":"text","text":"The quick "}],
      "final":false
   }
 }
}
```

---

## 5 Local vs. Ray deployment flow

| Step                                  | Local (asyncio)                                      | Ray cluster                                             |
| ------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------- |
| Agent calls `await mailbox.publish()` | Pushes to in-memory queue; back-pressure via `await` | RPC to queue actor; `put_async` blocks when cap reached |
| Connector subscribes                  | Iterator over same queue                             | Remote pulls via `get_async`                            |
| HTTP gateway                          | Reads events and relays as SSE                       | Identical code path                                     |
| Ordering & durability                 | FIFO in one process                                  | FIFO across workers; optional write-ahead journal       |

Because both mailboxes share ordering guarantees, **no changes are required in Agent or Connector** when switching deployment mode.

---

## 6 Durability & replay (optional)

Persist every published frame to an append-only log (Postgres, Kafka). If the gateway crashes it can **replay** from the last cursor and rebuild state—classic **event-sourcing** ([ramchandra-vadranam.medium.com][13], [softwareengineering.stackexchange.com][14]).

---

## 7 Back-pressure & flow control

* Producer awaits the queue; queue blocks when `maxsize` reached (both transports) ([lucumr.pocoo.org][15]).
* SSE inherits HTTP/2 window updates → slow browser automatically throttles the gateway ([medium.com][3]).

No custom credit-scheme is required.

---

## 8 Security considerations

* **Sandbox isolation** – all code execution happens in a Firecracker/gVisor sandbox, never in the agent process.
* **HMAC-signed frames** – sign every A2A JSON event to prevent spoofing.
* **Queue ACLs** – Ray actors can enable namespace-level RBAC; local mode trusts process isolation.

---

## 9 Extensibility roadmap

| Need                 | Drop-in change                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Multi-tenant metrics | Add a `monitoring.publish(evt)` side-channel; frames already include `task_id`.                                              |
| Binary artefacts     | Use `TaskArtifactUpdateEvent` with S3 presigned URI ([a2aprotocol.ai][6]).                                                   |
| gRPC streams         | Implement `GrpcMailbox` that mirrors the same `publish/subscribe` API over bidi streams (HTTP/2 flow-control still applies). |

---

## 10 Key references

1. Google Blog – A2A announcement ([developers.googleblog.com][7])
2. Medium – A2A real-time updates & `tasks/sendSubscribe` ([medium.com][1])
3. A2A official site & schema pointers ([a2aprotocol.ai][6])
4. GitHub issue showing live SSE frames (`final:true`) ([github.com][5])
5. Ray docs – `ray.util.queue.Queue` API & semantics ([docs.ray.io][9])
6. MDN – Server-Sent Events basics ([developer.mozilla.org][4])
7. Medium – SSE + HTTP/2 flow-control discussion ([medium.com][3])
8. Akka docs – classic & control-aware mailboxes ([doc.akka.io][11], [doc.akka.io][12])
9. Python discussion – back-pressure in `asyncio.Queue` ([discuss.python.org][8])
10. Lucumr essay – “async pressure” deep dive ([lucumr.pocoo.org][15])
11. Event-sourcing article – replay & versioning ([softwareengineering.stackexchange.com][14])
12. Event-sourcing vs messaging in distributed architectures ([ramchandra-vadranam.medium.com][13])
13. JSON-RPC 2.0 spec (ordering & IDs) ([docs.ray.io][10])
14. A2A community blog – SSE vs WebSocket rationale ([googlecloudcommunity.com][2])
15. Akka ControlAwareMailbox config snippet ([doc.akka.io][11])

These sources anchor every design choice in a production-tested pattern—letting you prototype on a laptop, scale to a Ray cluster, and interoperate with the broader A2A ecosystem without rewriting a line of agent code.

[1]: https://medium.com/google-cloud/a2a-deep-dive-getting-real-time-updates-from-ai-agents-a28d60317332?utm_source=chatgpt.com "A2A Deep Dive: Getting Real-Time Updates from AI Agents - Medium"
[2]: https://www.googlecloudcommunity.com/gc/Community-Blogs/Understanding-A2A-The-Protocol-for-Agent-Collaboration/ba-p/906323?utm_source=chatgpt.com "Understanding A2A — The Protocol for Agent Collaboration"
[3]: https://medium.com/%40kaitmore/server-sent-events-http-2-and-envoy-6927c70368bb?utm_source=chatgpt.com "Thoughts on Server-Sent Events, HTTP/2, and Envoy - Medium"
[4]: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events?utm_source=chatgpt.com "Using server-sent events - Web APIs | MDN"
[5]: https://github.com/google-a2a/A2A/issues/454?utm_source=chatgpt.com "tasks/sendSubscribe SSE events don't return sessionId #454 - GitHub"
[6]: https://a2aprotocol.ai/?utm_source=chatgpt.com "A2A Protocol - Agent-to-Agent Communication"
[7]: https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/?utm_source=chatgpt.com "Announcing the Agent2Agent Protocol (A2A)"
[8]: https://discuss.python.org/t/adding-cycle-buffer-to-asyncio-queue/5179?utm_source=chatgpt.com "Adding Cycle Buffer to asyncio.Queue - Python discussion forum"
[9]: https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.queue.Queue.html?utm_source=chatgpt.com "ray.util.queue.Queue — Ray 2.46.0 - Ray Docs"
[10]: https://docs.ray.io/en/latest/_modules/ray/util/queue.html?utm_source=chatgpt.com "ray.util.queue — Ray 2.46.0 - Ray Docs"
[11]: https://doc.akka.io/libraries/akka-core/current/mailboxes.html?utm_source=chatgpt.com "Classic Mailboxes - Akka Documentation"
[12]: https://doc.akka.io/libraries/akka-core/current/typed/mailboxes.html?utm_source=chatgpt.com "Mailboxes - Akka Documentation"
[13]: https://ramchandra-vadranam.medium.com/the-ultimate-guide-to-event-sourcing-and-messaging-systems-for-distributed-architectures-b05d36311c32?utm_source=chatgpt.com "The Ultimate Guide to Event Sourcing and Messaging Systems for ..."
[14]: https://softwareengineering.stackexchange.com/questions/310176/event-sourcing-replaying-and-versioning?utm_source=chatgpt.com "Event sourcing, replaying and versioning"
[15]: https://lucumr.pocoo.org/2020/1/1/async-pressure/?utm_source=chatgpt.com "I'm not feeling the async pressure"
