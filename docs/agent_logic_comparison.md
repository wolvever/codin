# Agent Logic Comparison: BaseAgent + CodePlanner vs CodeAgent vs Codex.rs

## Overview

This document compares the iterative execution loop logic between three implementations:

1. **BaseAgent + CodePlanner** (new architecture)
2. **CodeAgent** (existing Python implementation) 
3. **Codex.rs** (Rust reference implementation)

## Core Loop Logic Comparison

### 1. Loop Structure

#### BaseAgent + CodePlanner
```python
# In _execute_planning_loop()
while state.iteration < (state.config.turn_budget or 100):
    # Check control signals (pause/cancel)
    # Check budget constraints
    
    # Get steps from planner
    async for step in self.planner.next(state):
        # Execute each step (Message, ToolCall, Think, Event, Finish)
        if step.step_type == StepType.FINISH:
            task_finished_by_step = True
            break
    
    if task_finished_by_step:
        break
    
    state.iteration += 1
```

#### CodeAgent  
```python
# In run()
while iteration < max_iterations:
    # Check continuation decision
    continue_decision = self._should_continue_execution(...)
    if continue_decision != ContinueDecision.CONTINUE:
        break
    
    # Run single turn
    response_messages, tool_results = await self._run_turn(...)
    
    # Check task completion
    if self._is_task_complete(response_messages, tool_results, run_context):
        break
    
    # Continue if there were tool calls, else complete
    if tool_results:
        continue
    else:
        break
```

#### Codex.rs
```rust
// In run_task()
loop {
    // Run turn to get response items
    let turn_output = run_turn(sess, sub_id, turn_input).await?;
    
    let mut responses = Vec::<ResponseInputItem>::new();
    for processed_item in turn_output {
        // Handle response item (execute tools if needed)
        if let Some(response) = response {
            responses.push(response);
        }
    }
    
    if responses.is_empty() {
        break; // No more function calls, task complete
    }
    
    input_for_next_turn = responses;
}
```

### 2. Task Completion Logic

#### BaseAgent + CodePlanner
- **Primary**: Planner yields `FinishStep` 
- **Secondary**: Budget constraints exceeded
- **Fallback**: Max iterations reached

**Key**: Planner controls completion via `should_continue` flag in structured response

#### CodeAgent
- **Primary**: `should_continue=False` in LLM structured response
- **Secondary**: No tool calls + completion indicators in text
- **Fallback**: Max iterations or budget constraints

**Key**: Combination of explicit flag + heuristic text analysis

#### Codex.rs
- **Primary**: No function calls in response (empty `responses` list)
- **Secondary**: Error conditions
- **Fallback**: External interruption

**Key**: Function call presence drives continuation

### 3. State Management

#### BaseAgent + CodePlanner
```python
class State:
    session_id: str
    history: list[Message]
    last_tool_results: list[Any]
    task_list: dict[str, list[str]]  # {"completed": [], "pending": []}
    metrics: Metrics
    # ... other fields
```
- **Memory**: External Memory service
- **Tools**: ToolRegistry with tool objects
- **History**: Message list in State + Memory service

#### CodeAgent
```python
# Direct instance variables
self._conversation_history: list[Message] = []
self._approved_commands: set[tuple[str, ...]] = set()
# ... other tracking variables

# Context passed between turns
run_context = {
    "task_id": task_id,
    "task_list": {"completed": [], "pending": []},
    "should_continue": True
}
```
- **Memory**: MemMemoryService instance
- **Tools**: ToolRegistry with ToolExecutor
- **History**: Direct instance variable management

#### Codex.rs
```rust
struct State {
    approved_commands: HashSet<Vec<String>>,
    current_task: Option<AgentTask>,
    previous_response_id: Option<String>,
    pending_approvals: HashMap<String, oneshot::Sender<ReviewDecision>>,
    zdr_transcript: Option<ConversationHistory>,
}
```
- **Memory**: ConversationHistory for ZDR transcript
- **Tools**: MCP connection manager + built-in tools
- **History**: Conversation transcript management

### 4. Tool Execution Flow

#### BaseAgent + CodePlanner
1. Planner yields `ToolCallStep` 
2. Agent executes tool via `_execute_tool_call()`
3. Results stored in `state.last_tool_results`
4. Tool results fed back to planner in next iteration
5. Planner can access results via `state.last_tool_results`

#### CodeAgent
1. LLM response parsed for tool calls
2. Tools executed in sequence within same turn
3. Results formatted and added to conversation history
4. Next turn includes tool results in conversation context
5. Loop continues until no tool calls or completion signal

#### Codex.rs
1. Model returns function calls in response stream
2. Each function call processed and executed
3. Results collected as `ResponseInputItem`
4. Results become input for next turn
5. Loop continues until empty response list

### 5. Error Handling

#### BaseAgent + CodePlanner
- **Planning errors**: Planner yields error message + FinishStep
- **Tool errors**: Captured in tool result metadata
- **Budget errors**: Automatic FinishStep with budget reason
- **Control errors**: Handled via mailbox system

#### CodeAgent  
- **LLM errors**: Fallback responses, retry logic with backoff
- **Tool errors**: Error messages in tool results
- **Task errors**: Error events emitted
- **Timeout errors**: Per-tool and per-LLM timeouts

#### Codex.rs
- **Stream errors**: Retry with exponential backoff
- **Tool errors**: Captured in function call outputs  
- **Approval errors**: User approval rejection handling
- **Sandbox errors**: Escalation and retry without sandbox

## Key Differences Summary

| Aspect | BaseAgent + CodePlanner | CodeAgent | Codex.rs |
|--------|------------------------|-----------|----------|
| **Loop Driver** | Planner step generation | Tool call presence | Function call presence |
| **Completion Logic** | FinishStep from planner | should_continue flag + heuristics | Empty function call list |
| **State Management** | Immutable State object | Mutable instance variables | Mutex-protected state |
| **Tool Integration** | Step-based execution | Turn-based execution | Stream-based execution |
| **Memory Model** | External Memory service | Internal conversation history | ZDR transcript |
| **Error Recovery** | Step-level error handling | Turn-level error handling | Stream-level retry |

## Compatibility Assessment

The **BaseAgent + CodePlanner** architecture achieves similar functionality to both **CodeAgent** and **Codex.rs** with these key alignments:

### ✅ Compatible Aspects
- **Tool execution**: All three execute tools and feed results back
- **Iterative planning**: All use multi-turn LLM interactions
- **Budget constraints**: All support execution limits
- **Error handling**: All have error recovery mechanisms

### ⚠️ Structural Differences  
- **Completion detection**: Different mechanisms but same end result
- **State representation**: Different data structures but same information
- **Tool result handling**: Different timing but same data flow

### 🔧 Missing Features
- **Approval system**: CodePlanner doesn't implement user approval (yet)
- **Streaming**: BaseAgent supports streaming but CodePlanner uses it limitedly
- **Background tasks**: Less sophisticated than Codex.rs background handling

## Conclusion

The BaseAgent + CodePlanner architecture successfully implements the core iterative agent pattern while providing a cleaner separation of concerns. The main logic differences are in implementation details rather than fundamental approaches, making the new architecture a viable replacement for the existing patterns. 