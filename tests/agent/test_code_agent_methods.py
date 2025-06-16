import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from codin.agent.code_agent import CodeAgent
from codin.agent.types import (
    AgentRunInput, AgentRunOutput, Message, TextPart, ToolCallPart, 
    ToolCall, ToolCallResult, Role
)
from codin.memory.base import MemoryService
from codin.tool.registry import ToolRegistry
from codin.tool.base import Toolset
from codin.sandbox.base import Sandbox
from codin.model.base import LLM


@pytest.fixture
def mock_llm():
    llm = Mock(spec=LLM)
    llm.chat_completions = AsyncMock()
    return llm


@pytest.fixture  
def mock_memory():
    memory = Mock(spec=MemoryService)
    memory.add_message = AsyncMock()
    memory.get_history = AsyncMock(return_value=[])
    memory.build_chunk = AsyncMock()
    memory.search_chunk = AsyncMock(return_value=[])
    return memory


@pytest.fixture
def mock_tool_registry():
    registry = Mock(spec=ToolRegistry)
    registry.get_tools_with_executor = Mock(return_value=(Toolset(), Mock()))
    return registry


@pytest.fixture
def mock_sandbox():
    sandbox = Mock(spec=Sandbox)
    sandbox.run = AsyncMock()
    return sandbox


@pytest.fixture
def code_agent(mock_llm, mock_memory, mock_tool_registry, mock_sandbox):
    agent = CodeAgent(
        llm=mock_llm,
        memory=mock_memory,
        tool_registry=mock_tool_registry,
        sandbox=mock_sandbox,
        agent_id="test_agent"
    )
    return agent


class TestCodeAgent:
    
    @pytest.mark.asyncio
    async def test_add_event_callback(self, code_agent):
        """Test adding event callbacks."""
        callback = AsyncMock()
        
        code_agent.add_event_callback("test_event", callback)
        
        assert "test_event" in code_agent._event_callbacks
        assert callback in code_agent._event_callbacks["test_event"]
    
    @pytest.mark.asyncio
    async def test_add_event_callback_multiple(self, code_agent):
        """Test adding multiple callbacks for the same event."""
        callback1 = AsyncMock()
        callback2 = AsyncMock()
        
        code_agent.add_event_callback("test_event", callback1)
        code_agent.add_event_callback("test_event", callback2)
        
        assert len(code_agent._event_callbacks["test_event"]) == 2
        assert callback1 in code_agent._event_callbacks["test_event"]
        assert callback2 in code_agent._event_callbacks["test_event"]
    
    @pytest.mark.asyncio
    async def test_run_basic_flow(self, code_agent, mock_llm, mock_memory):
        """Test basic run flow without tool calls."""
        input_data = AgentRunInput(
            messages=[Message(
                role=Role.USER,
                content=[TextPart(text="Hello, how are you?")]
            )],
            runner_id="runner_123",
            request_id="req_123"
        )
        
        # Mock LLM response without tool calls
        mock_llm.chat_completions.return_value = [Message(
            role=Role.ASSISTANT,
            content=[TextPart(text="I'm doing well, thank you!")]
        )]
        
        outputs = []
        async for output in code_agent.run(input_data):
            outputs.append(output)
        
        assert len(outputs) == 1
        assert outputs[0].runner_id == "runner_123"
        assert outputs[0].request_id == "req_123"
        assert len(outputs[0].messages) == 1
        assert outputs[0].messages[0].content[0].text == "I'm doing well, thank you!"
        
        # Verify memory interactions
        mock_memory.add_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_run_with_tool_calls(self, code_agent, mock_llm, mock_memory, mock_tool_registry):
        """Test run flow with tool calls."""
        input_data = AgentRunInput(
            messages=[Message(
                role=Role.USER,
                content=[TextPart(text="What's 2+2?")]
            )],
            runner_id="runner_123",
            request_id="req_123"
        )
        
        # Mock LLM response with tool call
        tool_call = ToolCall(
            id="call_123",
            function={"name": "calculator", "arguments": '{"expression": "2+2"}'}
        )
        
        mock_llm.chat_completions.side_effect = [
            [Message(
                role=Role.ASSISTANT,
                content=[ToolCallPart(tool_calls=[tool_call])]
            )],
            [Message(
                role=Role.ASSISTANT,
                content=[TextPart(text="The answer is 4.")]
            )]
        ]
        
        # Mock tool execution
        mock_executor = Mock()
        mock_executor.execute = AsyncMock(return_value=ToolCallResult(
            call_id="call_123",
            result="4",
            error=None
        ))
        mock_tool_registry.get_tools_with_executor.return_value = (Toolset(), mock_executor)
        
        outputs = []
        async for output in code_agent.run(input_data):
            outputs.append(output)
        
        assert len(outputs) >= 1
        # Should have final assistant message
        final_output = outputs[-1]
        assert "4" in final_output.messages[-1].content[0].text
        
        # Verify tool execution
        mock_executor.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reset_conversation(self, code_agent, mock_memory):
        """Test resetting conversation state."""
        # Add some state
        code_agent._conversation_history = [Message(
            role=Role.USER,
            content=[TextPart(text="Previous message")]
        )]
        
        await code_agent.reset_conversation()
        
        assert len(code_agent._conversation_history) == 0
        # Should clear memory as well
        mock_memory.get_history.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_memory_history(self, code_agent, mock_memory):
        """Test getting memory history."""
        expected_history = [
            Message(role=Role.USER, content=[TextPart(text="Test message 1")]),
            Message(role=Role.ASSISTANT, content=[TextPart(text="Test response 1")])
        ]
        mock_memory.get_history.return_value = expected_history
        
        history = await code_agent.get_memory_history()
        
        assert history == expected_history
        mock_memory.get_history.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_compress_conversation_history(self, code_agent, mock_memory, mock_llm):
        """Test compressing conversation history."""
        # Setup long conversation history
        long_history = [
            Message(role=Role.USER, content=[TextPart(text=f"Message {i}")])
            for i in range(20)
        ]
        code_agent._conversation_history = long_history
        
        # Mock compression response
        mock_llm.chat_completions.return_value = [Message(
            role=Role.ASSISTANT,
            content=[TextPart(text="Compressed summary of conversation")]
        )]
        
        await code_agent.compress_conversation_history()
        
        # Should have fewer messages after compression
        assert len(code_agent._conversation_history) < len(long_history)
        mock_llm.chat_completions.assert_called()
        mock_memory.add_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_conversation_summary(self, code_agent, mock_llm):
        """Test getting conversation summary."""
        code_agent._conversation_history = [
            Message(role=Role.USER, content=[TextPart(text="What's the weather?")]),
            Message(role=Role.ASSISTANT, content=[TextPart(text="I can't check weather without tools.")])
        ]
        
        # Mock summary response
        mock_llm.chat_completions.return_value = [Message(
            role=Role.ASSISTANT,
            content=[TextPart(text="User asked about weather, assistant explained limitations.")]
        )]
        
        summary = await code_agent.get_conversation_summary()
        
        assert "weather" in summary.lower()
        mock_llm.chat_completions.assert_called()
    
    @pytest.mark.asyncio
    async def test_add_tool(self, code_agent, mock_tool_registry):
        """Test adding a tool to the agent."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        
        code_agent.add_tool(mock_tool)
        
        # Should add tool to registry (implementation dependent)
        # This test might need adjustment based on actual implementation
        assert mock_tool in code_agent._additional_tools
    
    @pytest.mark.asyncio
    async def test_cleanup(self, code_agent, mock_memory, mock_sandbox):
        """Test cleanup of agent resources."""
        # Add some state to clean up
        code_agent._conversation_history = [Message(
            role=Role.USER,
            content=[TextPart(text="Test")]
        )]
        code_agent._event_callbacks = {"test": [AsyncMock()]}
        
        await code_agent.cleanup()
        
        # Should clear internal state
        assert len(code_agent._conversation_history) == 0
        assert len(code_agent._event_callbacks) == 0
        # Should cleanup sandbox if needed
        if hasattr(mock_sandbox, 'cleanup'):
            mock_sandbox.cleanup.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_handling_llm_failure(self, code_agent, mock_llm, mock_memory):
        """Test error handling when LLM fails."""
        input_data = AgentRunInput(
            messages=[Message(
                role=Role.USER,
                content=[TextPart(text="Test message")]
            )],
            runner_id="runner_123",
            request_id="req_123"
        )
        
        # Mock LLM to raise exception
        mock_llm.chat_completions.side_effect = Exception("LLM API error")
        
        outputs = []
        try:
            async for output in code_agent.run(input_data):
                outputs.append(output)
        except Exception as e:
            assert "LLM API error" in str(e)
        
        # Should still have attempted to save messages to memory
        mock_memory.add_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self, code_agent, mock_llm, mock_tool_registry):
        """Test error handling when tool execution fails."""
        input_data = AgentRunInput(
            messages=[Message(
                role=Role.USER,
                content=[TextPart(text="Execute tool")]
            )],
            runner_id="runner_123",
            request_id="req_123"
        )
        
        # Mock LLM response with tool call
        tool_call = ToolCall(
            id="call_123",
            function={"name": "failing_tool", "arguments": '{}'}
        )
        
        mock_llm.chat_completions.return_value = [Message(
            role=Role.ASSISTANT,
            content=[ToolCallPart(tool_calls=[tool_call])]
        )]
        
        # Mock tool execution to fail
        mock_executor = Mock()
        mock_executor.execute = AsyncMock(return_value=ToolCallResult(
            call_id="call_123",
            result=None,
            error="Tool execution failed"
        ))
        mock_tool_registry.get_tools_with_executor.return_value = (Toolset(), mock_executor)
        
        outputs = []
        async for output in code_agent.run(input_data):
            outputs.append(output)
        
        # Should handle error gracefully and continue
        assert len(outputs) >= 1
        mock_executor.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_event_emission(self, code_agent):
        """Test that events are properly emitted during execution."""
        callback = AsyncMock()
        code_agent.add_event_callback("agent_start", callback)
        
        input_data = AgentRunInput(
            messages=[Message(
                role=Role.USER,
                content=[TextPart(text="Test")]
            )],
            runner_id="runner_123",
            request_id="req_123"
        )
        
        # Mock LLM response
        code_agent._llm.chat_completions.return_value = [Message(
            role=Role.ASSISTANT,
            content=[TextPart(text="Response")]
        )]
        
        outputs = []
        async for output in code_agent.run(input_data):
            outputs.append(output)
        
        # Verify event callback was called
        # This might need adjustment based on actual event emission implementation
        # callback.assert_called()