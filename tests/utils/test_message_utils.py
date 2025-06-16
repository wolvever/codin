import pytest
from codin.agent.types import Message, TextPart, Role
from codin.agent.types import ToolCallResult
from codin.utils.message import (
    extract_text_from_message,
    format_history_for_prompt,
    format_tool_results_for_conversation,
)


def test_extract_text_from_message():
    msg = Message(messageId="1", parts=[TextPart(text="hello"), TextPart(text="world")], role=Role.user)
    assert extract_text_from_message(msg) == "hello\nworld"


def test_format_history_for_prompt():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert format_history_for_prompt(history) == "User: Hi\n\nAssistant: Hello"


def test_format_tool_results_for_conversation():
    results = [
        ToolCallResult(call_id="1", success=True, output="done"),
        ToolCallResult(call_id="2", success=False, error="fail"),
    ]
    expected = "\n".join(
        [
            "**Tool Call 1** ✅ Success",
            "Output: done",
            "",
            "**Tool Call 2** ❌ Failed",
            "Error: fail",
            "",
        ]
    ).strip()
    assert format_tool_results_for_conversation(results).strip() == expected
