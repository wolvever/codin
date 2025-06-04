
## Agentic Loop


Below is an implementation-agnostic explanation of the “agent loop” that both OpenHands and Codex follow, what differs between them, and why Codex stores follow-ups in a queue whereas OpenHands keeps only one pending action.

──────────────────────────────────────────────
1 How many “actions” can arrive at once?
──────────────────────────────────────────────
OpenHands (loop.py)  
• `Agent.step(state)` returns **exactly one** `Action` object  
  (or `NullAction`).  
• The controller therefore needs to track at most one *runnable*
  action whose result (an `Observation`) is still outstanding.  
• It stores that single action in the field `_pending_action` and
  blocks further steps until the matching observation arrives.  
  Queueing would be unnecessary overhead.

Codex (codex.rs)  
• The model is queried through a streaming API.  
  A single turn can emit a sequence like  

  ```
  “assistant: …”
  FunctionCall("container.exec", …)
  Reasoning(…)
  FunctionCall("shell", …)
  …
  ```

  i.e. **many** actionable items.  
• Each actionable `ResponseItem` is executed immediately and *may*
  create a `ResponseInputItem` that must be sent back to the model on
  the next turn.  
• Those follow-ups are accumulated in `pending_response_input`
  (a `Vec<ResponseInputItem>`); after the turn finishes, they are
  prepended to the next prompt.  
  Hence a queue (vector) is required.

──────────────────────────────────────────────
2 Unified pseudo-algorithm
──────────────────────────────────────────────
The snippet below condenses the logic of both projects into one
high-level, side-effect-free algorithm.  Think of it as the contract the
future *codin* loop will have to implement.

```python
# Pseudo code – NOT tied to any concrete library / runtime
# --------------------------------------------------------
def run_agent(user_messages: list[UserMsg]) -> list[Event]:
    """
    Drive the agent until the current user request is handled.
    Returns an ordered Event log for the front-end.
    """
    history: list[Event] = []                 # conversation so far
    followups: list[FollowUp] = to_followups(user_messages)
    agent_state = RUNNING                     # or IDLE / ERROR / etc.

    while agent_state == RUNNING:
        # ── 1. Compose the next prompt ──────────────────────────
        prompt = build_prompt(history, followups)  # fn of policies, truncation, …
        followups.clear()                          # will refill below

        # ── 2. Let the model talk (stream) ──────────────────────
        for token_or_item in stream_llm(prompt):
            if is_displayable(token_or_item):
                history.append(display_event(token_or_item))      # text to UI

            elif is_action(token_or_item):
                # 2a. Execute the requested side-effect
                try:
                    result = run_action(token_or_item)            # shell, HTTP, …
                    if result.must_be_sent_back:                  # Codex path
                        followups.append(result.to_followup())
                except NeedsUserApproval as ask:
                    emit_approval_request(ask)                    # blocks task
                    pending_decision = wait_for_decision()
                    if pending_decision.denied:
                        followups.append(ask.failure_followup())
                    else:
                        redo = run_action(token_or_item, sandbox=None)
                        followups.append(redo.to_followup())

            elif is_terminal(token_or_item):       # e.g. AgentFinishAction
                agent_state = FINISHED
                break

        # ── 3. Traffic-control / stuck-detection guards ─────────
        if exceeded_iteration_limit() or exceeded_budget():
            agent_state = ERROR
            history.append(error_event("traffic-control"))
        if is_stuck(history):
            agent_state = ERROR
            history.append(error_event("stuck"))

        # ── 4. Exit condition ───────────────────────────────────
        if agent_state != RUNNING:
            break
        # If the loop continues:
        #   • OpenHands will have at most 1 follow-up originating from
        #     the pending action’s Observation.
        #   • Codex may have 0-N follow-ups gathered above.
        #   • If both followups **and** queued user messages are empty,
        #     the model produced a final answer and the turn is done.
        if not followups and no_queued_user_input():
            break

    return history
```

Legend of helper concepts  
• `FollowUp` - a wrapper that the model expects to receive on the next  
  call (Codex: `ResponseInputItem`; OpenHands: at most a single
  *Observation* with `cause = previous_action_id`).  
• `run_action` - executes a tool, shell command, patch, etc., returns an
  object containing stdout / metadata / approval state.  
• `display_event` - converts text tokens or messages into `AgentMessage`
  events for the UI.  
• `traffic-control` and `is_stuck` reflect the iteration-, cost- and
  loop-protection present in both codebases.

──────────────────────────────────────────────
3 Take-aways for future codin implementation
──────────────────────────────────────────────
1. Decide whether the internal agent API can emit *multiple* actionable
   items per turn.  
   • If **one** → a single “pending_action” slot à la OpenHands
     suffices.  
   • If **many** → follow-ups need to be stored in a queue à la Codex.

2. Regardless of (1) the outer loop always has the same shape:  
   build prompt → stream model → execute side-effects → collect
   follow-ups → repeat until no follow-ups or a terminal state.

3. Keep explicit guardrails (iteration, cost, stuck detector) out of the
   critical path; check them once per turn before calling the model
   again.

4. Separate *displayable* output (goes straight to the UI) from
   *follow-ups* (fed back to the model).  Both loops do this split, just
   with different container types.

You can now translate this abstract logic into concrete code for the new
codin project, choosing the follow-up-queue strategy that best matches
the capabilities of its own model interface.


### OpenHands Loop

Below is a step-by-step walkthrough of how a single natural-language message from the user is carried all the way to completion inside the OpenHands runtime.  
File/line citations refer to the three modules you pointed out.

--------------------------------------------------------------------
1.  The one-shot driver coroutine
--------------------------------------------------------------------
```
 1:26:reflib/coding/OpenHands/openhands/core/loop.py
async def run_agent_until_done(controller, runtime, memory, end_states): ...
```

• `run_agent_until_done` is a tiny wrapper that:

  1.  Installs a shared `status_callback` on the three core objects  
     (runtime, controller, memory).  
  2.  Enters a `while` loop that simply sleeps one second until the
     controller’s `state.agent_state` becomes one of the supplied
     `end_states` (typically `FINISHED`, `REJECTED`, `ERROR`, or `STOPPED`).  

The heavy lifting therefore happens inside the controller and the runtime;  
the loop’s only job is to stay alive long enough for them to finish.

--------------------------------------------------------------------
2.  How a user message enters the system
--------------------------------------------------------------------
a.  The front-end (GUI, CLI, or test harness) turns the user text into a
    `MessageAction` and adds it to the **EventStream**
    (`EventSource.USER`).  
b.  Because the root `AgentController` subscribed to that stream during
    construction (see next section), its `on_event` callback is invoked.

--------------------------------------------------------------------
3.  The AgentController lifecycle
--------------------------------------------------------------------
Subscription and initialisation:

```
 46:57:reflib/.../agent_controller.py
self.event_stream.subscribe(EventStreamSubscriber.AGENT_CONTROLLER,
                            self.on_event, self.id)
```

Key paths inside `AgentController` once an event arrives:

1. `on_event` (synchronous)  
   • Forwards to a delegate controller if one is active.  
   • Otherwise dispatches into `_on_event` (async) via
     `asyncio.get_event_loop().run_until_complete`.

2. `_on_event`  
   • Adds non-filtered events to an in-memory history.  
   • Routes `Action`s to `_handle_action` and `Observation`s to
     `_handle_observation`.  
   • Decides whether to “step” the agent by calling `should_step`.  
     A new user message always returns `True`.

3. `_step_with_exception_handling → _step`  
   • Performs traffic-control checks (max iterations / cost).  
   • Detects loops via `StuckDetector`.  
   • Increments iteration counters (`update_state_before_step`).  
   • Asks the `Agent` for its next `Action` (`self.agent.step(...)`).  
   • Emits that `Action` into the `EventStream` and, if the action is
     runnable (e.g. `CmdRunAction`, `BrowseURLAction`, …), stores it as
     `_pending_action` so that the controller will wait for the
     corresponding `Observation` before stepping again.  
   • Updates metrics (`update_state_after_step`).

Any exception anywhere in this chain is translated to a user-visible
error message and the state becomes `ERROR`
(`_react_to_exception`).

--------------------------------------------------------------------
4.  Executing runnable actions – the Runtime
--------------------------------------------------------------------
When the `Runtime` receives an `Action` event it can execute, the
following happens:

```
878:906:reflib/.../runtime/base.py
async def _handle_action(self, event: Action) -> None:
    ...
    observation = await call_sync_from_async(self.run_action, event)
    observation._cause = event.id
    self.event_stream.add_event(observation, source)
```

• Each specific action type (`CmdRunAction`, `FileReadAction`, ...) is
  implemented by a concrete runtime subclass (Docker, local shell,
  etc.).  
• The resulting `Observation` is tagged with the `Action`’s id (`cause`)
  and pushed back to the `EventStream`.

--------------------------------------------------------------------
5.  Closing the control loop
--------------------------------------------------------------------
Once the observation is in the stream, `AgentController._on_event`
matches it with `_pending_action`, clears that flag, updates metrics,
and returns to the main loop.  
Eventually the agent emits an `AgentFinishAction` or the controller
hits an error/stop condition, setting its state to one of the
`end_states`.  
`run_agent_until_done` notices this and simply exits, letting the caller
clean up.

--------------------------------------------------------------------
6.  Information flow at a glance
--------------------------------------------------------------------
USER → `MessageAction` → EventStream  
        ↓ (subscription)  
AgentController.on_event  
        ↓ (_step)  
Agent.step → **Action** (may be runnable) → EventStream  
        ↓ (subscription)  
Runtime.on_event → execute in sandbox → **Observation** → EventStream  
        ↓ (subscription)  
AgentController._handle_observation → clears pending_action  
(loop repeats)  
        ... → `AgentFinishAction` → state = FINISHED  
run_agent_until_done detects FINISHED → returns.

This orchestrated interaction among the event stream, agent controller,
and runtime — all coordinated by the lightweight `run_agent_until_done`
coroutine — is what allows OpenHands to accept a user query and keep
running until the task is complete or an unrecoverable error occurs.


### Codex Loop

Below is the “happy-path” journey of a single human message through the Codex agent, starting from the CLI / API boundary and ending when the turn is declared finished.  The description is implementation-centred and points at the key Rust modules that realise each step.

──────────────────────────────────────────────────────────────────
0.  Names to keep in mind
──────────────────────────────────────────────────────────────────
• Submission        = `{ id, op }` pushed into the *in-queue*  
• Op                = enum describing what the user / UI wants  
• Event             = `{ id, msg }` popped from the *out-queue*  
• Codex             = thin façade around those two queues  
• Session           = per-conversation state holder  
• AgentTask         = Tokio task that drives one “turn”  
• turn              = sequence of model calls and tool executions
──────────────────────────────────────────────────────────────────
1.  Entry point – the CLI / SDK
──────────────────────────────────────────────────────────────────
CLI (`codex chat`, `codex repl`), WebSocket bridge, or SDK code calls:

    let (codex, _) = Codex::spawn(config, ctrl_c).await?;
    codex.submit(Op::ConfigureSession{…}).await?;
    codex.submit(Op::UserInput{ items }).await?;

`spawn` [codex.rs 15-64] creates two bounded channels  
`tx_sub`  (Submissions in) and `rx_event` (Events out) and
kicks off `submission_loop` in a background Tokio task.

──────────────────────────────────────────────────────────────────
2.  The central dispatcher – `submission_loop`
──────────────────────────────────────────────────────────────────
Located in the same file [codex.rs 423-620].

• Repeatedly selects on  
  – `rx_sub.recv()`               ⟶ got a Submission  
  – `ctrl_c.notified()`           ⟶ abort current turn

• Holds at most one `Session` (`sess: Option<Arc<Session>>`).

Important `Op` arms
a. `ConfigureSession`  
   – Creates / replaces a `Session` (state, model client, MCP-registry,  
     rollout recorder, writable roots, …).  
   – ACKs with `SessionConfigured` Event and possible MCP errors.

b. `UserInput { items }`  
   – Try `sess.inject_input(items)`; if the user typed while the model
     is still thinking, the message is queued.  
   – If there is no running turn, spawn:

        AgentTask::spawn(Arc::clone(sess), sub.id, items)

c. `Interrupt`, `ExecApproval`, `PatchApproval`, …  
   – Forwarded to helper methods on `Session`.

──────────────────────────────────────────────────────────────────
3.  Spawning and managing a single turn
──────────────────────────────────────────────────────────────────
`AgentTask::spawn` is a Tokio `run_task` [codex.rs 546-611].

Initial work:
• Push `TaskStarted` Event.  
• Wrap the user `InputItem`s into `pending_response_input`.

Main loop:

    loop {
        build `turn_input`  // includes any model follow-ups or queued user
        match run_turn(sess, sub_id, turn_input).await {
            Ok(turn_output) => {
                if follow-up FunctionCallOutputs are required
                    pending_response_input = follow-ups   // loop again
                else
                    break;   // turn done
            }
            Err(e) => emit ErrorEvent; return;
        }
    }
    push TaskComplete Event;  sess.remove_task(...)

Thus the task keeps cycling until the model has nothing more to ask or
do regarding the original user message.

──────────────────────────────────────────────────────────────────
4.  Doing one model round – `run_turn`
──────────────────────────────────────────────────────────────────
[codex.rs 619-666]

• Build `Prompt` (chat history, tool catalogue,…).  
• `ModelClient::stream(prompt)` returns an async stream of
  `ResponseEvent`s. Buffer all items first to avoid tool-timeout race.

For every `ResponseEvent::OutputItemDone(item)` call
`handle_response_item`.

Key cases inside `handle_response_item` [codex.rs 810-898]:
a. Message / Reasoning → converted to `AgentMessageEvent` /
   `AgentReasoningEvent` and forwarded as UI Events.

b. `FunctionCall`  
   – If name is `container.exec`/`shell` → parse JSON args, enforce
     sandbox policy, maybe request user approval, then run
     `process_exec_tool_call`; result becomes `ResponseInputItem::FunctionCallOutput`.
   – If name looks like `server.tool` → delegate to external MCP server
     via `handle_mcp_tool_call`.
   – Else → immediate failure output so the model can resample.

c. `LocalShellCall` (legacy path) → same sandbox / approval machinery.

`ResponseInputItem`s collected from (b) are the “follow-ups” sent back
to the model on the next iteration of `run_task`.

At stream end a `ResponseEvent::Completed{ response_id }` stores the
server-side id for context-window reuse.

──────────────────────────────────────────────────────────────────
5.  Tool execution & safety
──────────────────────────────────────────────────────────────────
`process_exec_tool_call` (in `exec/mod.rs`) actually forks the sandbox,
feeds stdin/stdout, enforces timeout, etc.

Safety gating happens via:

• `assess_command_safety` and `assess_patch_safety`  
   – combine user-chosen `AskForApproval` policy with static whitelist
     (`writable_roots`, network access, etc.).  
   – may emit `ExecApprovalRequestEvent` or
     `ApplyPatchApprovalRequestEvent`.

User decisions come back as `Op::ExecApproval` /
`Op::PatchApproval` which `submission_loop` forwards to
`Session::notify_approval`.

──────────────────────────────────────────────────────────────────
6.  Event stream outwards
──────────────────────────────────────────────────────────────────
Every noteworthy thing—assistant tokens, tool progress, begin/end of
exec, errors, retries—is wrapped in an `Event` and sent through
`tx_event`.  
The frontend simply awaits `codex.next_event()` and renders or acts on
each message until it observes `TaskComplete`, `Error`, or the user
interrupts.

──────────────────────────────────────────────────────────────────
7.  Completion criteria
──────────────────────────────────────────────────────────────────
The turn (i.e. the user query) is finished when:
• `run_turn` produced zero follow-up `ResponseInputItem`s
  (the assistant neither requested a function call nor handed back a
  tool result that itself triggers more calls); **or**
• An unrecoverable error bubbled out and an `ErrorEvent` was emitted; **or**
• The user interrupted (`Op::Interrupt`), which aborts the running
  `AgentTask`.

In all three cases `run_task` exits its loop, emits `TaskComplete` or
`Error`, and clears `Session.state.current_task`, allowing the next
user-initiated `Op::UserInput` to spawn a fresh `AgentTask`.

──────────────────────────────────────────────────────────────────
Summary timeline
──────────────────────────────────────────────────────────────────
USER → `Op::UserInput` → tx_sub  
submission_loop → AgentTask::spawn  
AgentTask `run_task`  
      ↳ gathers inputs  
      ↳ `run_turn` (LLM stream)  
           ↳ `handle_response_item` … tool calls … approvals  
      ↻ loop while follow-ups  
      → emit `TaskComplete`  
      → frontend drains `Codex::next_event` until it sees same id.

That is how Codex ingests a user query, orchestrates model reasoning,
sandboxed code / patch execution, user approvals, and finally signals
completion back to the caller.

#### Follow-ups

In Codex vocabulary a “follow-up” is any piece of information that the model itself asks to receive back on the very next turn.  
Concretely it is represented by `ResponseInputItem` values that Codex produces **after** handling an LLM output item such as a tool call, shell command, patch application, etc.

────────────────────────────────────────────────────────────
1.  Where follow-ups live
────────────────────────────────────────────────────────────
```
760:768:reflib/coding/codex/codex-rs/core/src/codex.rs
let mut pending_response_input: Vec<ResponseInputItem> =
        vec![ResponseInputItem::from(input)];   // ← start with the user request
```
`pending_response_input` is the queue of follow-ups that still need to be
fed into the model.

────────────────────────────────────────────────────────────
2.  How they are produced
────────────────────────────────────────────────────────────
Inside `run_turn` every streamed `ResponseEvent::OutputItemDone(item)` is
passed to `handle_response_item`:

```
1005:1013:codex.rs
match item {
    ResponseItem::FunctionCall { … }  => Some( …handle_function_call… )
    ResponseItem::LocalShellCall { … }=> Some( …handle_container_exec… )
    …                                    …
}
```

`handle_function_call` / `handle_container_exec_with_params` execute the
requested action and return a **FunctionCallOutput** (another
`ResponseInputItem`).  
If the model did not ask for anything actionable the function returns
`None`, meaning no follow-up is required.

────────────────────────────────────────────────────────────
3.  Feeding them back
────────────────────────────────────────────────────────────
After the turn finishes, `run_task` separates the raw output (`items`)
from the generated follow-ups (`responses`):

```
818:833:codex.rs
let (items, responses): (Vec<_>, Vec<_>) = turn_output.into_iter().map(|p| (p.item, p.response)).unzip();
let responses = responses.into_iter().flatten().collect::<Vec<ResponseInputItem>>();
…
if responses.is_empty() {                // ← no follow-ups → turn done
    break;
}
pending_response_input = responses;       // ← save for next loop
```

At the **top** of the loop those saved follow-ups are converted back to
`ResponseItem`s and become the first entries of the next prompt:

```
770:777:codex.rs
let mut net_new_turn_input = pending_response_input
        .drain(..)
        .map(ResponseItem::from)
        .collect::<Vec<_>>();
```

Hence each follow-up automatically becomes part of the system’s next
message to the model, allowing the LLM to see the tool result it asked
for and continue reasoning.

────────────────────────────────────────────────────────────
4.  When the follow-up queue finally empties
────────────────────────────────────────────────────────────
The loop terminates when `responses.is_empty()` — i.e. the LLM produced
no new tool calls, patch requests, or other actions that need a
ResponseInputItem reply.  At that moment Codex emits
`TaskComplete`, finishes the `AgentTask`, and is ready for the next user
message.

────────────────────────────────────────────────────────────
5.  Relation to user interjections
────────────────────────────────────────────────────────────
While a turn is running, the user may type another message.  
Those arrive via `Session::inject_input` and are stored in
`state.pending_input`.  At the beginning of each iteration they are
merged with `net_new_turn_input`:

```
778:781:codex.rs
let pending_input = sess.get_pending_input().into_iter().map(ResponseItem::from);
net_new_turn_input.extend(pending_input);
```

So *follow-ups* (model-requested inputs) and *user interjections* both
enter the next prompt, but they originate from different queues.

────────────────────────────────────────────────────────────
Summary
────────────────────────────────────────────────────────────
• Follow-ups are `ResponseInputItem`s generated after executing the
  model’s requested actions.  
• They are queued in `pending_response_input`, turned back into
  `ResponseItem`s, and prepended to the next prompt sent to the model.  
• The agent keeps looping until that queue is empty, signalling that the
  model has everything it needs and the user’s query is complete.
