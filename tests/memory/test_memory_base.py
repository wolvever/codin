from unittest.mock import AsyncMock, Mock

import pytest

from codin.agent.types import Message, Role, TextPart
from codin.memory.base import MemMemoryService, MemoryChunk, MemoryService


@pytest.fixture
def sample_messages():
    return [
        Message(role=Role.USER, content=[TextPart(text="Hello, how are you?")]),
        Message(role=Role.ASSISTANT, content=[TextPart(text="I'm doing well, thank you!")]),
        Message(role=Role.USER, content=[TextPart(text="What's the weather like?")]),
        Message(role=Role.ASSISTANT, content=[TextPart(text="I don't have access to weather data.")])
    ]


class TestMemoryChunk:
    
    def test_memory_chunk_creation(self):
        """Test creating a memory chunk."""
        chunk = MemoryChunk(
            id="chunk_1",
            content={"summary": "Weather discussion"},
            metadata={"type": "conversation", "topic": "weather"}
        )
        
        assert chunk.id == "chunk_1"
        assert chunk.content["summary"] == "Weather discussion"
        assert chunk.metadata["topic"] == "weather"
    
    def test_get_content_dict(self):
        """Test getting content as dictionary."""
        content = {"summary": "Test summary", "details": ["item1", "item2"]}
        chunk = MemoryChunk(id="test", content=content, metadata={})
        
        result = chunk.get_content_dict()
        
        assert result == content
        assert isinstance(result, dict)
    
    def test_get_content_string(self):
        """Test getting content as string."""
        content = {"summary": "Test summary", "key_points": ["point1", "point2"]}
        chunk = MemoryChunk(id="test", content=content, metadata={})
        
        result = chunk.get_content_string()
        
        assert isinstance(result, str)
        assert "Test summary" in result
        assert "point1" in result
        assert "point2" in result
    
    def test_to_message(self):
        """Test converting chunk to message."""
        content = {"summary": "Previous conversation about weather"}
        metadata = {"timestamp": "2023-01-01T12:00:00Z"}
        chunk = MemoryChunk(id="test", content=content, metadata=metadata)
        
        message = chunk.to_message()
        
        assert isinstance(message, Message)
        assert message.role == Role.SYSTEM
        assert "Previous conversation about weather" in message.content[0].text
    
    def test_to_message_with_custom_role(self):
        """Test converting chunk to message with custom role."""
        content = {"summary": "User context"}
        chunk = MemoryChunk(id="test", content=content, metadata={})
        
        message = chunk.to_message(role=Role.USER)
        
        assert message.role == Role.USER


class TestMemMemoryService:
    
    @pytest.fixture
    def memory_service(self):
        return MemMemoryService()
    
    @pytest.mark.asyncio
    async def test_add_message(self, memory_service, sample_messages):
        """Test adding messages to memory."""
        for message in sample_messages:
            await memory_service.add_message(message)
        
        history = await memory_service.get_history()
        assert len(history) == len(sample_messages)
        assert history == sample_messages
    
    @pytest.mark.asyncio
    async def test_get_history(self, memory_service, sample_messages):
        """Test getting conversation history."""
        # Add messages
        for message in sample_messages:
            await memory_service.add_message(message)
        
        history = await memory_service.get_history()
        
        assert len(history) == 4
        assert history[0].content[0].text == "Hello, how are you?"
        assert history[-1].content[0].text == "I don't have access to weather data."
    
    @pytest.mark.asyncio
    async def test_get_history_with_limit(self, memory_service, sample_messages):
        """Test getting history with limit."""
        for message in sample_messages:
            await memory_service.add_message(message)
        
        history = await memory_service.get_history(limit=2)
        
        assert len(history) == 2
        # Should return most recent messages
        assert history[0].content[0].text == "What's the weather like?"
        assert history[1].content[0].text == "I don't have access to weather data."
    
    @pytest.mark.asyncio
    async def test_set_chunk_builder(self, memory_service):
        """Test setting chunk builder."""
        mock_builder = Mock()
        
        await memory_service.set_chunk_builder(mock_builder)
        
        assert memory_service._chunk_builder == mock_builder
    
    @pytest.mark.asyncio
    async def test_build_chunk_without_builder(self, memory_service, sample_messages):
        """Test building chunk without chunk builder set."""
        for message in sample_messages:
            await memory_service.add_message(message)
        
        chunk = await memory_service.build_chunk("weather_conversation")
        
        # Should create a basic chunk even without builder
        assert chunk is not None
        assert chunk.id == "weather_conversation"
        assert "summary" in chunk.content
    
    @pytest.mark.asyncio
    async def test_build_chunk_with_builder(self, memory_service, sample_messages):
        """Test building chunk with custom chunk builder."""
        mock_builder = AsyncMock()
        expected_chunk = MemoryChunk(
            id="custom_chunk",
            content={"custom_summary": "Custom built chunk"},
            metadata={"builder": "custom"}
        )
        mock_builder.build_chunk.return_value = expected_chunk
        
        await memory_service.set_chunk_builder(mock_builder)
        for message in sample_messages:
            await memory_service.add_message(message)
        
        chunk = await memory_service.build_chunk("test_chunk")
        
        assert chunk == expected_chunk
        mock_builder.build_chunk.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_chunk_basic(self, memory_service):
        """Test basic chunk searching."""
        # Create some chunks
        chunk1 = MemoryChunk(
            id="chunk1",
            content={"summary": "Discussion about weather and climate"},
            metadata={"topic": "weather"}
        )
        chunk2 = MemoryChunk(
            id="chunk2", 
            content={"summary": "Conversation about programming languages"},
            metadata={"topic": "programming"}
        )
        
        memory_service._chunks = [chunk1, chunk2]
        
        results = await memory_service.search_chunk("weather")
        
        assert len(results) == 1
        assert results[0].id == "chunk1"
    
    @pytest.mark.asyncio
    async def test_search_chunk_no_results(self, memory_service):
        """Test chunk searching with no results."""
        chunk = MemoryChunk(
            id="chunk1",
            content={"summary": "Programming discussion"},
            metadata={}
        )
        memory_service._chunks = [chunk]
        
        results = await memory_service.search_chunk("weather")
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_chunk_multiple_results(self, memory_service):
        """Test chunk searching with multiple results."""
        chunks = [
            MemoryChunk(
                id="chunk1",
                content={"summary": "Weather forecast discussion"},
                metadata={}
            ),
            MemoryChunk(
                id="chunk2",
                content={"summary": "Climate and weather patterns"},
                metadata={}
            ),
            MemoryChunk(
                id="chunk3",
                content={"summary": "Programming tutorial"},
                metadata={}
            )
        ]
        memory_service._chunks = chunks
        
        results = await memory_service.search_chunk("weather")
        
        assert len(results) == 2
        assert {r.id for r in results} == {"chunk1", "chunk2"}
    
    @pytest.mark.asyncio
    async def test_clear_history(self, memory_service, sample_messages):
        """Test clearing conversation history."""
        for message in sample_messages:
            await memory_service.add_message(message)
        
        assert len(await memory_service.get_history()) == 4
        
        await memory_service.clear_history()
        
        assert len(await memory_service.get_history()) == 0
    
    @pytest.mark.asyncio
    async def test_memory_persistence_across_operations(self, memory_service):
        """Test that memory persists across different operations."""
        # Add some messages
        message1 = Message(role=Role.USER, content=[TextPart(text="First message")])
        message2 = Message(role=Role.ASSISTANT, content=[TextPart(text="First response")])
        
        await memory_service.add_message(message1)
        await memory_service.add_message(message2)
        
        # Build a chunk
        await memory_service.build_chunk("test_chunk")
        
        # Add more messages
        message3 = Message(role=Role.USER, content=[TextPart(text="Second message")])
        await memory_service.add_message(message3)
        
        # Verify all data is still there
        history = await memory_service.get_history()
        assert len(history) == 3
        
        search_results = await memory_service.search_chunk("summary")
        assert len(search_results) >= 1


class TestMemoryServiceInterface:
    
    def test_memory_service_is_abstract(self):
        """Test that MemoryService is an abstract class."""
        with pytest.raises(TypeError):
            MemoryService()
    
    @pytest.mark.asyncio
    async def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteMemory(MemoryService):
            pass
        
        with pytest.raises(TypeError):
            IncompleteMemory()
    
    @pytest.mark.asyncio
    async def test_valid_subclass_implementation(self):
        """Test a valid MemoryService subclass implementation."""
        class ValidMemory(MemoryService):
            def __init__(self):
                self._messages = []
            
            async def add_message(self, message):
                self._messages.append(message)
            
            async def get_history(self, limit=None):
                if limit:
                    return self._messages[-limit:]
                return self._messages.copy()
            
            async def set_chunk_builder(self, builder):
                self._builder = builder
            
            async def build_chunk(self, chunk_id):
                return MemoryChunk(
                    id=chunk_id,
                    content={"summary": "test"},
                    metadata={}
                )
            
            async def search_chunk(self, query):
                return []
        
        memory = ValidMemory()
        assert isinstance(memory, MemoryService)
        
        # Test basic functionality
        message = Message(role=Role.USER, content=[TextPart(text="Test")])
        await memory.add_message(message)
        
        history = await memory.get_history()
        assert len(history) == 1
        assert history[0] == message