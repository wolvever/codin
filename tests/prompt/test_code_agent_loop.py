import json
import pytest

from tests.prompt.test_engine import MockLLM
import codin.prompt.run as prompt_run_module
from codin.prompt.run import PromptEngine

@pytest.mark.asyncio
async def test_code_agent_loop_prompt_with_mock_llm(run_async):
    llm = MockLLM()
    llm.response = json.dumps({
        "thinking": "done",
        "message": "completed",
        "tool_calls": [{"name": "tool", "arguments": {"n": i}} for i in range(10)],
        "task_list": {"completed": [], "pending": []},
        "should_continue": False
    })

    prompt_run_module._engine = PromptEngine(llm, endpoint="fs://./prompt_templates")

    variables = {
        "agent_name": "Agent",
        "task_id": "t1",
        "turn_count": 1,
        "has_tools": False,
        "tools": [],
        "has_history": False,
        "history_text": "",
        "user_input": "hello",
        "tool_results": False,
        "tool_results_text": "",
        "task_list": {"completed": [], "pending": []},
        "rules": None,
    }

    response = await prompt_run_module.prompt_run("code_agent_loop", variables=variables)
    assert response.content == llm.response
    data = json.loads(response.content)
    assert data["message"] == "completed"
    assert len(data["tool_calls"]) == 10
    assert data["should_continue"] is False
