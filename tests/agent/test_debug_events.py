"""Test debug event functionality in CodeAgent."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.codin.agent.code_agent import CodeAgent, AgentEvent
from src.codin.sandbox import LocalSandbox
from src.codin.tool.registry import ToolRegistry
from a2a.types import Role, TextPart
from src.codin.agent.types import Message, AgentRunInput


class TestDebugEvents:
    """Test debug event functionality."""

    @pytest.fixture
    async def agent_with_debug(self):
        """Create a CodeAgent with debug enabled."""
        # Mock the sandbox and LLM
        sandbox = MagicMock()
        sandbox.up = AsyncMock()
        sandbox.down = AsyncMock()
        
        agent = CodeAgent(
            name="Test Agent",
            description="Test agent for debug events",
            llm_model="test-model",
            sandbox=sandbox,
            debug=True  # Enable debug mode
        )
        
        # Mock the LLM to return a structured response
        mock_response = MagicMock()
        mock_response.content = (
            '{"thinking": "Test thinking", "message": "Test message", "should_continue": false, '
            '"task_list": {"completed": [], "pending": []}, "tool_calls": []}'
        )
        
        # Mock the model's run method
        agent._model = MagicMock()
        agent._model.run = AsyncMock(return_value=mock_response)
        
        yield agent
        
        await agent.cleanup()

    @pytest.fixture
    async def agent_without_debug(self):
        """Create a CodeAgent with debug disabled."""
        # Mock the sandbox and LLM
        sandbox = MagicMock()
        sandbox.up = AsyncMock()
        sandbox.down = AsyncMock()
        
        agent = CodeAgent(
            name="Test Agent",
            description="Test agent without debug",
            llm_model="test-model",
            sandbox=sandbox,
            debug=False  # Disable debug mode
        )
        
        # Mock the LLM to return a structured response
        mock_response = MagicMock()
        mock_response.content = (
            '{"thinking": "Test thinking", "message": "Test message", "should_continue": false, '
            '"task_list": {"completed": [], "pending": []}, "tool_calls": []}'
        )
        
        # Mock the model's run method
        agent._model = MagicMock()
        agent._model.run = AsyncMock(return_value=mock_response)
        
        yield agent
        
        await agent.cleanup()

    @pytest.mark.asyncio
    async def test_debug_event_emitted_when_debug_enabled(self, agent_with_debug):
        """Test that debug events are emitted when debug mode is enabled."""
        # Track events
        received_events = []
        
        async def event_callback(event: AgentEvent):
            received_events.append(event)
        
        agent_with_debug.add_event_callback(event_callback)
        
        # Create a test message
        user_message = Message(
            messageId="test-message",
            role=Role.user,
            parts=[TextPart(text="Test prompt")]
        )
        
        # Run the agent
        agent_input = AgentRunInput(message=user_message)
        await agent_with_debug.run(agent_input)
        
        # Check that debug events were emitted
        debug_events = [e for e in received_events if e.event_type == "debug_llm_response"]
        assert len(debug_events) > 0, (
            "Debug events should be emitted when debug mode is enabled"
        )
        
        # Verify the debug event structure
        debug_event = debug_events[0]
        assert "turn_count" in debug_event.data
        assert "raw_content_length" in debug_event.data
        assert "thinking" in debug_event.data
        assert "message" in debug_event.data
        assert "should_continue" in debug_event.data
        assert "task_list" in debug_event.data
        assert "tool_calls" in debug_event.data

    @pytest.mark.asyncio
    async def test_debug_event_not_emitted_when_debug_disabled(self, agent_without_debug):
        """Test that debug events are not emitted when debug mode is disabled."""
        # Track events
        received_events = []
        
        async def event_callback(event: AgentEvent):
            received_events.append(event)
        
        agent_without_debug.add_event_callback(event_callback)
        
        # Create a test message
        user_message = Message(
            messageId="test-message",
            role=Role.user,
            parts=[TextPart(text="Test prompt")]
        )
        
        # Run the agent
        agent_input = AgentRunInput(message=user_message)
        await agent_without_debug.run(agent_input)
        
        # Check that no debug events were emitted
        debug_events = [e for e in received_events if e.event_type == "debug_llm_response"]
        assert len(debug_events) == 0, (
            "Debug events should not be emitted when debug mode is disabled"
        )

    @pytest.mark.asyncio
    async def test_debug_event_data_structure(self, agent_with_debug):
        """Test that debug event data has the correct structure."""
        # Track events
        received_events = []
        
        async def event_callback(event: AgentEvent):
            received_events.append(event)
        
        agent_with_debug.add_event_callback(event_callback)
        
        # Mock the response parsing to return specific data
        mock_tool_call_1 = MagicMock()
        mock_tool_call_1.name = "test_tool"
        mock_tool_call_1.call_id = "call_1"
        mock_tool_call_1.arguments = {"arg1": "value1"}
        
        mock_tool_call_2 = MagicMock()
        mock_tool_call_2.name = "another_tool"
        mock_tool_call_2.call_id = "call_2"
        mock_tool_call_2.arguments = {"arg2": "value2", "arg3": "value3"}
        
        mock_parsed_response = {
            "thinking": "Test thinking content",
            "message": "Test message content", 
            "should_continue": False,  # Set to False to prevent infinite loop
            "task_list": {
                "completed": ["Task 1", "Task 2"],
                "pending": ["Task 3"]
            },
            "tool_calls": [mock_tool_call_1, mock_tool_call_2]
        }
        
        # Mock the parsing method
        agent_with_debug._parse_structured_response = MagicMock(return_value=mock_parsed_response)
        
        # Create a test message
        user_message = Message(
            messageId="test-message",
            role=Role.user,
            parts=[TextPart(text="Test prompt")]
        )
        
        # Run the agent with timeout to prevent hanging
        agent_input = AgentRunInput(message=user_message)
        try:
            await asyncio.wait_for(agent_with_debug.run(agent_input), timeout=10.0)
        except asyncio.TimeoutError:
            pytest.fail("Agent run timed out - this suggests the test is hanging")
        
        # Get the debug event
        debug_events = [e for e in received_events if e.event_type == "debug_llm_response"]
        assert len(debug_events) > 0
        
        debug_data = debug_events[0].data
        
        # Verify the structure
        assert debug_data["thinking"] == "Test thinking content"
        assert debug_data["message"] == "Test message content"
        assert debug_data["should_continue"] == False  # Updated to match new mock
        assert debug_data["task_list"]["completed_count"] == 2
        assert debug_data["task_list"]["pending_count"] == 1
        assert debug_data["task_list"]["completed"] == ["Task 1", "Task 2"]
        assert debug_data["task_list"]["pending"] == ["Task 3"]
        assert len(debug_data["tool_calls"]) == 2
        assert debug_data["tool_calls"][0]["name"] == "test_tool"
        assert debug_data["tool_calls"][0]["arguments_keys"] == ["arg1"]
        assert debug_data["tool_calls"][1]["name"] == "another_tool"
        assert debug_data["tool_calls"][1]["arguments_keys"] == ["arg2", "arg3"] 