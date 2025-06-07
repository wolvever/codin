# ConcurrentRunner

`ConcurrentRunner` manages multiple `AgentRunner` instances and starts or stops them concurrently. It is useful when a dispatcher or host needs to orchestrate several long-running agents.

## Example

```python
from codin.agent import AgentRunner
from codin.agent.concurrent_runner import ConcurrentRunner

# Assume `agent_a` and `agent_b` are Agent instances
runner_a = AgentRunner(agent_a)
runner_b = AgentRunner(agent_b)

group = ConcurrentRunner()
group.add_runner(runner_a)
group.add_runner(runner_b)

await group.start_all()
# ... agents are now processing messages ...
await group.stop_all()
```
