"""Test enhanced Step system with A2A compatibility."""

import pytest
from datetime import datetime
from codin.agent.types import (
    Step,
    StepType,
    MessageStep,
    EventStep,
    ToolCallStep,
    ThinkStep,
    FinishStep,
    Message,
    Task,
    ToolCall,
    ToolCallResult,
    ToolUsePart,
    EventType,
    RunEvent,
    TaskStatusUpdateEvent,
)
from codin.agent.types import TaskState, TaskStatus


class TestEnhancedSteps:
    """Test enhanced Step system with mixed content support."""

    def test_step_content_detection(self):
        """Test step content detection methods."""
        # Create a step with message content
        message = Message(messageId="test-msg", role="user", parts=[])
        step = Step(step_id="test-1", step_type=StepType.MESSAGE, message=message)

        assert step.has_message_content()
        assert not step.has_event_content()
        assert not step.has_tool_content()
        assert step.get_content_types() == ["message"]

    def test_step_mixed_content(self):
        """Test step with mixed content types."""
        message = Message(messageId="test-msg", role="agent", parts=[])
        tool_call = ToolCall(call_id="call-1", name="test_tool", arguments={})

        step = Step(
            step_id="test-2",
            step_type=StepType.MESSAGE,
            message=message,
            tool_call=tool_call,
            thinking="Some internal reasoning",
        )

        assert step.has_message_content()
        assert step.has_tool_content()
        assert not step.has_event_content()
        content_types = step.get_content_types()
        assert "message" in content_types
        assert "tool" in content_types
        assert "thinking" in content_types

    def test_message_step_tool_call_parts(self):
        """Test MessageStep with tool call parts."""
        # Create tool call and result
        tool_call = ToolCall(call_id="call-123", name="search_web", arguments={"query": "latest AI news"})
        tool_result = ToolCallResult(call_id="call-123", success=True, output="Found 10 articles about AI")

        # Create message with tool call parts
        message = Message(messageId="msg-with-tools", role="agent", parts=[])
        message.add_text_part("I'll search for that information.")
        message.add_tool_call_part(tool_call)
        message.add_tool_result_part(tool_result, tool_call.name)
        message.add_text_part("Here are the results I found.")

        # Create message step
        step = MessageStep(step_id="msg-step-1", message=message)

        assert step.has_message_content()
        assert len(step.message.parts) == 4

        # Check part types
        assert step.message.parts[0].kind == "text"
        assert step.message.parts[1].kind == "tool-use"
        assert step.message.parts[1].type == "call"
        assert step.message.parts[2].kind == "tool-use"
        assert step.message.parts[2].type == "result"
        assert step.message.parts[3].kind == "text"

    def test_tool_call_step_enhanced(self):
        """Test enhanced ToolCallStep with result handling."""
        tool_call = ToolCall(
            call_id="call-456", name="write_file", arguments={"path": "test.txt", "content": "Hello world"}
        )

        step = ToolCallStep(step_id="tool-step-1", tool_call=tool_call)

        assert step.has_tool_content()
        assert step.tool_call_result is None
        assert step.success is True  # Default

        # Add result
        result = ToolCallResult(call_id="call-456", success=True, output="File written successfully")
        step.add_result(result)

        assert step.tool_call_result is not None
        assert step.success is True

        # Test conversion to message parts
        call_part, result_part = step.to_message_parts()
        assert isinstance(call_part, ToolUsePart)
        assert isinstance(result_part, ToolUsePart)
        assert call_part.kind == "tool-use"
        assert call_part.type == "call"
        assert result_part.kind == "tool-use"
        assert result_part.type == "result"

    def test_event_step_a2a_vs_internal(self):
        """Test EventStep with A2A vs internal events."""
        # A2A event
        a2a_event = TaskStatusUpdateEvent(
            contextId="ctx-1", taskId="task-1", status=TaskStatus(state=TaskState.working), final=False
        )

        a2a_step = EventStep(step_id="event-step-1", event=a2a_event, event_type=EventType.TASK_STATUS_UPDATE)

        assert a2a_step.has_event_content()
        assert a2a_step.is_a2a_event()
        assert not a2a_step.is_internal_event()

        # Internal event
        internal_event = RunEvent(event_type="custom_event", data={"custom": "data"})

        internal_step = EventStep(step_id="event-step-2", event=internal_event, event_type=EventType.THINK)

        assert internal_step.has_event_content()
        assert not internal_step.is_a2a_event()
        assert internal_step.is_internal_event()

    def test_finish_step_completion_event(self):
        """Test FinishStep with completion event creation."""
        step = FinishStep(step_id="finish-1", reason="Task completed successfully")

        assert step.has_message_content()
        assert step.message is not None
        assert len(step.message.parts) == 1
        assert step.message.parts[0].text == "Task completed successfully"

        # Create completion event
        completion_event = step.create_completion_event(task_id="task-123", context_id="ctx-456")

        assert isinstance(completion_event, TaskStatusUpdateEvent)
        assert completion_event.taskId == "task-123"
        assert completion_event.contextId == "ctx-456"
        assert completion_event.final is True
        assert completion_event.status.state == TaskState.completed

    async def test_message_step_streaming(self):
        """Test MessageStep streaming functionality."""
        message = Message(messageId="streaming-msg", role="agent", parts=[])
        message.add_text_part(
            "This is a longer text that will be streamed in chunks to demonstrate " + "the streaming functionality."
        )

        step = MessageStep(step_id="stream-step-1", message=message, is_streaming=True)

        chunks = []
        async for chunk in step.stream_content():
            chunks.append(chunk)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Recombined chunks should equal original text
        full_text = "".join(chunks)
        assert full_text == message.parts[0].text

    def test_tool_use_part_validation(self):
        """Test ToolUsePart validation for both calls and results."""
        # Test tool call
        tool_call_part = ToolUsePart(
            type="call", id="call-789", name="calculate", input={"x": 5, "y": 3}, metadata={"priority": "high"}
        )

        assert tool_call_part.kind == "tool-use"
        assert tool_call_part.type == "call"
        assert tool_call_part.id == "call-789"
        assert tool_call_part.name == "calculate"
        assert tool_call_part.input == {"x": 5, "y": 3}
        assert tool_call_part.metadata["priority"] == "high"

        # Test tool result
        result_part = ToolUsePart(
            type="result", id="call-789", name="calculate", output="8", metadata={"execution_time": 0.05}
        )

        assert result_part.kind == "tool-use"
        assert result_part.type == "result"
        assert result_part.id == "call-789"
        assert result_part.name == "calculate"
        assert result_part.output == "8"
        assert result_part.metadata["execution_time"] == 0.05

    def test_comprehensive_message_with_all_parts(self):
        """Test message with all types of parts including tool calls."""
        message = Message(messageId="comprehensive-msg", role="agent", parts=[])

        # Add various part types
        message.add_text_part("Let me help you with that task.")

        message.add_data_part({"task_info": {"id": "123", "type": "analysis"}})

        tool_call = ToolCall(call_id="call-999", name="analyze_data", arguments={"dataset": "user_data.csv"})
        message.add_tool_call_part(tool_call)

        tool_result = ToolCallResult(
            call_id="call-999", success=True, output="Analysis complete: 1000 records processed"
        )
        message.add_tool_result_part(tool_result, "analyze_data")

        message.add_text_part("Analysis is complete! Here are the results.")

        # Verify all parts
        assert len(message.parts) == 5
        assert message.parts[0].kind == "text"
        assert message.parts[1].kind == "data"
        assert message.parts[2].kind == "tool-use"
        assert message.parts[2].type == "call"
        assert message.parts[3].kind == "tool-use"
        assert message.parts[3].type == "result"
        assert message.parts[4].kind == "text"

        # Create step with this comprehensive message
        step = MessageStep(step_id="comprehensive-step", message=message)

        content_types = step.get_content_types()
        assert "message" in content_types
        assert step.has_message_content()
