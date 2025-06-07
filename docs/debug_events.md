# Debug Events in Codin

This document describes the debug event functionality in the Codin agent system.

## Overview

The debug event system allows external components (like CLI interfaces) to receive detailed debug information about the agent's operation without cluttering the agent's core logic with presentation code.

## How It Works

### Event-Based Debug Information

Instead of directly printing debug information within the agent code, the agent now emits structured debug events that can be consumed by event listeners. This separation of concerns allows:

- **Agent Code**: Focuses purely on logic and decision-making
- **Event Renderers**: Handle presentation and formatting of debug information
- **Flexibility**: Different interfaces can display debug information in their own style

### Debug Event Structure

When debug mode is enabled (`debug=True`), the agent emits `debug_llm_response` events with the following structure:

```python
{
    "turn_count": int,                    # Current turn number
    "raw_content_length": int,            # Length of raw LLM response
    "thinking": str | None,               # Agent's thinking process
    "message": str | None,                # Agent's message to user
    "should_continue": bool,              # Whether agent should continue
    "task_list": {
        "completed_count": int,           # Number of completed tasks
        "pending_count": int,             # Number of pending tasks
        "completed": list[str],           # List of completed tasks
        "pending": list[str]              # List of pending tasks
    },
    "tool_calls": [                       # Information about tool calls
        {
            "name": str,                  # Tool name
            "arguments_keys": list[str],  # Keys of tool arguments
            "call_id": str               # Unique call identifier
        }
    ]
}
```

## Implementation

### Agent Side

The agent emits debug events in the `_run_turn` method:

```python
if self.debug:
    debug_info = {
        "turn_count": self._turn_count,
        "raw_content_length": len(content),
        "thinking": parsed_response['thinking'],
        "message": parsed_response['message'],
        "should_continue": parsed_response['should_continue'],
        "task_list": {
            "completed_count": len(parsed_response['task_list']['completed']),
            "pending_count": len(parsed_response['task_list']['pending']),
            "completed": parsed_response['task_list']['completed'],
            "pending": parsed_response['task_list']['pending']
        },
        "tool_calls": [...]
    }
    
    await self._emit_event("debug_llm_response", debug_info)
```

### CLI Side

Both the REPL and quiet mode CLI interfaces register event callbacks to handle debug events:

#### REPL Mode

The REPL uses `click` for colorized output:

```python
elif event.event_type == "debug_llm_response":
    debug_info = event.data
    click.echo(click.style(f"🤖 LLM Response (Turn {debug_info['turn_count']}):", bold=True, fg="cyan"))
    # ... detailed formatting
```

#### Quiet Mode

Quiet mode uses simple print statements:

```python
async def debug_event_callback(event):
    if event.event_type == "debug_llm_response":
        debug_info = event.data
        print(f"🤖 LLM Response (Turn {debug_info['turn_count']}):")
        # ... formatting
```

## Benefits

1. **Separation of Concerns**: Agent logic is separate from presentation
2. **Consistency**: Debug information format is standardized
3. **Flexibility**: Different interfaces can format debug info differently
4. **Testability**: Debug events can be easily tested
5. **Performance**: No overhead when debug mode is disabled

## Usage

### Enabling Debug Mode

```python
# For programmatic use
agent = CodeAgent(debug=True, ...)

# For CLI use
codin --debug "your prompt here"
```

### Handling Debug Events

```python
async def my_debug_handler(event):
    if event.event_type == "debug_llm_response":
        # Custom handling of debug information
        debug_data = event.data
        # ... process debug_data as needed

agent.add_event_callback(my_debug_handler)
```

## Task Events

Two additional events provide high-level progress information:

- `task_start` - emitted when a task begins.
- `task_end` - emitted when the task finishes.

Both events include the following fields:

```python
{
    "session_id": str,   # Conversation/session identifier
    "iteration": int,    # Current iteration when the event was emitted
    "elapsed_time": float  # Seconds since the task started
}
```

Event listeners registered via `add_event_callback` receive these events the same
way as `debug_llm_response`.

## Run Events

When using the simplified planning loop, the agent emits a `run_end` event after
the planning loop completes and all cleanup steps finish. This allows external
systems to know when an agent run has fully finished.

The event data includes:

```python
{
    "agent_id": str,
    "session_id": str,
    "elapsed_time": float  # Total runtime in seconds
}
```

## Migration

The previous direct print-based debug system has been completely replaced with this event-based approach. No breaking changes were made to the public API - the `debug` parameter still works the same way, but the implementation is now cleaner and more flexible. 
