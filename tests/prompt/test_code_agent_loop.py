import json
import pytest

from tests.prompt.test_engine import MockLLM
import codin.prompt.run as prompt_run_module
from codin.prompt.run import PromptEngine

@pytest.mark.asyncio
async def test_code_agent_loop_prompt_with_mock_llm(run_async):
    """Run the prompt through multiple stages with a MockLLM."""

    llm = MockLLM()
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

    # ------------------------------------------------------------------
    # Stage 1: Send initial query and start the task
    # ------------------------------------------------------------------
    llm.response = json.dumps({
        "thinking": "start",
        "message": "starting",
        "tool_calls": [{"name": "tool", "arguments": {"n": i}} for i in range(10)],
        "task_list": {"completed": [], "pending": []},
        "should_continue": True,
    })

    response = await prompt_run_module.prompt_run("code_agent_loop", variables=variables)
    stage_data = json.loads(response.content)
    assert stage_data["should_continue"] is True
    assert len(stage_data["tool_calls"]) == 10

    # Prepare dummy results for next turn
    tool_calls = stage_data["tool_calls"]

    # ------------------------------------------------------------------
    # Stage 2: Loop three iterations, feeding previous tool results back in
    # ------------------------------------------------------------------
    for loop in range(3):
        variables["turn_count"] += 1
        variables["tool_results"] = True
        variables["tool_results_text"] = json.dumps(
            [{"name": call["name"], "result": f"r{loop}-{idx}"} for idx, call in enumerate(tool_calls)]
        )

        llm.response = json.dumps({
            "thinking": f"loop-{loop}",
            "message": f"iteration-{loop}",
            "tool_calls": [{"name": "tool", "arguments": {"n": i}} for i in range(10)],
            "task_list": {"completed": [], "pending": []},
            "should_continue": True,
        })

        response = await prompt_run_module.prompt_run("code_agent_loop", variables=variables)
        stage_data = json.loads(response.content)
        assert stage_data["should_continue"] is True
        assert len(stage_data["tool_calls"]) == 10
        tool_calls = stage_data["tool_calls"]

    # ------------------------------------------------------------------
    # Stage 3: No more tool calls, task should finish
    # ------------------------------------------------------------------
    variables["turn_count"] += 1
    variables["tool_results_text"] = json.dumps(
        [{"name": call["name"], "result": "final"} for call in tool_calls]
    )

    llm.response = json.dumps({
        "thinking": "done",
        "message": "completed",
        "tool_calls": [],
        "task_list": {"completed": [], "pending": []},
        "should_continue": False,
    })

    response = await prompt_run_module.prompt_run("code_agent_loop", variables=variables)
    final_data = json.loads(response.content)

    assert final_data["message"] == "completed"
    assert len(final_data["tool_calls"]) == 0
    assert final_data["should_continue"] is False
