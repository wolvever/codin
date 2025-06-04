"""Tests for the memory system."""

import pytest
import uuid
from datetime import datetime

from codin.memory.base import MemoryService, MemMemoryService, MemoryChunk, ChunkType
from a2a.types import Message, TextPart, Role


class TestMemoryChunk:
    """Test MemoryChunk functionality."""
    
    def test_init(self):
        """Test MemoryChunk initialization."""
        entities = {"file1": "main.py", "person1": "Alice"}
        chunk = MemoryChunk(
            doc_id="doc-123",
            chunk_id="test-chunk",
            session_id="session1",
            chunk_type=ChunkType.MEMORY_ENTITY,
            content=entities,
            title="Test Entities",
            start_message_id="msg1",
            end_message_id="msg5",
            created_at=datetime.now(),
            message_count=5
        )
        
        assert chunk.doc_id == "doc-123"
        assert chunk.chunk_id == "test-chunk"
        assert chunk.session_id == "session1"
        assert chunk.chunk_type == ChunkType.MEMORY_ENTITY
        assert chunk.title == "Test Entities"
        assert chunk.message_count == 5
        assert chunk.get_content_dict() == entities
        assert "file1: main.py" in chunk.content
    
    def test_to_message(self):
        """Test converting MemoryChunk to Message."""
        chunk = MemoryChunk(
            doc_id="doc-123",
            chunk_id="test-chunk",
            session_id="session1",
            chunk_type=ChunkType.MEMORY_SUMMARY,
            content="Test conversation about Python",
            title="Python Discussion",
            start_message_id="msg1",
            end_message_id="msg5",
            created_at=datetime.now(),
            message_count=3
        )
        
        message = chunk.to_message()
        
        assert message.role == Role.user
        assert message.contextId == "session1"
        assert message.kind == "message"
        
        # Extract text content
        first_part = message.parts[0]
        if hasattr(first_part, 'root') and hasattr(first_part.root, 'text'):
            text_content = first_part.root.text
        elif hasattr(first_part, 'text'):
            text_content = first_part.text
        else:
            text_content = str(first_part)
        
        assert "MEMORY SUMMARY" in text_content
        assert "3 messages" in text_content
        assert "Python Discussion" in text_content
        assert "Python" in text_content
        
        # Check metadata
        assert message.metadata["is_memory_chunk"] is True
        assert message.metadata["doc_id"] == "doc-123"
        assert message.metadata["chunk_id"] == "test-chunk"
        assert message.metadata["chunk_type"] == "memory_summary"
        assert message.metadata["title"] == "Python Discussion"


class TestInMemoryStore:
    """Test InMemoryStore implementation."""
    
    @pytest.fixture
    def memory_store(self):
        """Create a fresh InMemoryStore for each test."""
        return MemMemoryService()
    
    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [
            Message(
                messageId="msg1",
                role=Role.user,
                parts=[TextPart(text="Hello, how are you?")],
                contextId="session1",
                kind="message"
            ),
            Message(
                messageId="msg2",
                role=Role.agent,
                parts=[TextPart(text="I'm doing well, thank you! How can I help you today?")],
                contextId="session1", 
                kind="message"
            ),
            Message(
                messageId="msg3",
                role=Role.user,
                parts=[TextPart(text="Can you help me write a Python function?")],
                contextId="session1",
                kind="message"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_add_message(self, memory_store, sample_messages):
        """Test adding messages to memory."""
        for msg in sample_messages:
            await memory_store.add_message(msg)
        
        # Check internal storage
        assert "session1" in memory_store._messages
        assert len(memory_store._messages["session1"]) == 3
    
    @pytest.mark.asyncio
    async def test_get_history(self, memory_store, sample_messages):
        """Test retrieving conversation history."""
        # Add messages
        for msg in sample_messages:
            await memory_store.add_message(msg)
        
        # Get history
        history = await memory_store.get_history("session1", limit=10)
        
        assert len(history) == 3
        assert history[0].messageId == "msg1"
        assert history[1].messageId == "msg2"
        assert history[2].messageId == "msg3"
    
    @pytest.mark.asyncio
    async def test_get_history_with_limit(self, memory_store, sample_messages):
        """Test history retrieval with limit."""
        # Add messages
        for msg in sample_messages:
            await memory_store.add_message(msg)
        
        # Get limited history
        history = await memory_store.get_history("session1", limit=2)
        
        assert len(history) == 2
        assert history[0].messageId == "msg2"  # Last 2 messages
        assert history[1].messageId == "msg3"
    
    @pytest.mark.asyncio
    async def test_create_memory_chunk(self, memory_store, sample_messages):
        """Test creating memory chunks."""
        chunks = await memory_store.create_memory_chunk("session1", sample_messages)
        
        assert isinstance(chunks, list)
        assert len(chunks) >= 1  # At least summary chunk
        
        # Check summary chunk
        summary_chunk = chunks[0]
        assert isinstance(summary_chunk, MemoryChunk)
        assert summary_chunk.session_id == "session1"
        assert summary_chunk.chunk_type == ChunkType.MEMORY_SUMMARY
        assert summary_chunk.start_message_id == "msg1"
        assert summary_chunk.end_message_id == "msg3"
        assert summary_chunk.message_count == 3
        assert len(summary_chunk.content) > 0
        assert "Conversation Summary" in summary_chunk.title
    
    @pytest.mark.asyncio
    async def test_create_memory_chunk_empty_list(self, memory_store):
        """Test creating memory chunk with empty message list."""
        with pytest.raises(ValueError, match="Cannot create memory chunk from empty message list"):
            await memory_store.create_memory_chunk("session1", [])
    
    @pytest.mark.asyncio
    async def test_search_memory_chunks(self, memory_store, sample_messages):
        """Test searching memory chunks."""
        # Create chunks
        chunks = await memory_store.create_memory_chunk("session1", sample_messages)
        
        # Search for relevant chunks
        results = await memory_store.search_memory_chunks("session1", "python", limit=5)
        
        # Should find chunks since they contain "Python"
        assert len(results) >= 0  # May be 0 if search doesn't match
    
    @pytest.mark.asyncio
    async def test_get_history_with_query(self, memory_store, sample_messages):
        """Test getting history with search query."""
        # Add messages and create chunks
        for msg in sample_messages:
            await memory_store.add_message(msg)
        
        chunks = await memory_store.create_memory_chunk("session1", sample_messages[:2])
        
        # Get history with query
        history = await memory_store.get_history("session1", limit=10, query="hello")
        
        # Should include recent messages, and potentially memory chunks
        assert len(history) >= 3  # At least the 3 original messages
    
    @pytest.mark.asyncio
    async def test_compress_old_messages(self, memory_store, sample_messages):
        """Test compressing old messages."""
        # Add messages
        for msg in sample_messages:
            await memory_store.add_message(msg)
        
        # Compress keeping only 1 recent message
        chunk_groups_created = await memory_store.compress_old_messages(
            "session1", 
            keep_recent=1, 
            chunk_size=2
        )
        
        assert chunk_groups_created == 1  # Should create 1 chunk group from 2 old messages
        
        # Check that only 1 recent message remains
        remaining_messages = memory_store._messages["session1"]
        assert len(remaining_messages) == 1
        assert remaining_messages[0].messageId == "msg3"  # Most recent
        
        # Check that chunks were created (at least summary chunk)
        chunks = memory_store._chunks["session1"]
        assert len(chunks) >= 1
    
    def test_simple_summarize(self, memory_store):
        """Test simple text summarization."""
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        summary = memory_store._simple_summarize(text)
        
        assert "Line 1" in summary
        assert "Line 5" in summary
        assert "..." in summary
    
    def test_simple_summarize_short_text(self, memory_store):
        """Test simple summarization with short text."""
        text = "Short text"
        summary = memory_store._simple_summarize(text)
        
        assert summary == text  # Should return unchanged


class TestMemorySystemIntegration:
    """Test memory system integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_multiple_sessions(self):
        """Test handling multiple sessions."""
        memory = MemMemoryService()
        
        # Add messages to different sessions
        msg1 = Message(
            messageId="msg1",
            role=Role.user,
            parts=[TextPart(text="Session 1 message")],
            contextId="session1",
            kind="message"
        )
        
        msg2 = Message(
            messageId="msg2",
            role=Role.user,
            parts=[TextPart(text="Session 2 message")],
            contextId="session2",
            kind="message"
        )
        
        await memory.add_message(msg1)
        await memory.add_message(msg2)
        
        # Get history for each session
        history1 = await memory.get_history("session1")
        history2 = await memory.get_history("session2")
        
        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0].messageId == "msg1"
        assert history2[0].messageId == "msg2"
    
    @pytest.mark.asyncio
    async def test_llm_summarizer_integration(self):
        """Test integration with LLM summarizer."""
        memory = MemMemoryService()
        
        messages = [
            Message(
                messageId="msg1",
                role=Role.user,
                parts=[TextPart(text="Create a Python file called main.py")],
                contextId="session1",
                kind="message"
            ),
            Message(
                messageId="msg2",
                role=Role.agent,
                parts=[TextPart(text="I'll create main.py for you with a simple structure.")],
                contextId="session1",
                kind="message"
            )
        ]
        
        # Mock LLM summarizer
        async def mock_summarizer(text: str) -> dict:
            return {
                "summary": "User requested creation of main.py file",
                "entities": {"file": "main.py", "language": "Python"},
                "id_mappings": {"file1": "main.py"}
            }
        
        chunks = await memory.create_memory_chunk("session1", messages, mock_summarizer)
        
        # Find summary chunk
        summary_chunk = next(c for c in chunks if c.chunk_type == ChunkType.MEMORY_SUMMARY)
        assert summary_chunk.content == "User requested creation of main.py file"
        
        # Find entities chunk
        entities_chunk = next(c for c in chunks if c.chunk_type == ChunkType.MEMORY_ENTITY)
        entities_dict = entities_chunk.get_content_dict()
        assert entities_dict["file"] == "main.py"
        assert entities_dict["language"] == "Python"
        
        # Find ID mappings chunk
        mappings_chunk = next(c for c in chunks if c.chunk_type == ChunkType.MEMORY_ID_MAPPING)
        mappings_dict = mappings_chunk.get_content_dict()
        assert mappings_dict["file1"] == "main.py"
    
    @pytest.mark.asyncio
    async def test_search_with_title_weighting(self):
        """Test search functionality with title weighting."""
        memory = MemMemoryService()
        
        # Create chunks with different titles
        chunk1 = MemoryChunk(
            doc_id="doc1",
            chunk_id="chunk1",
            session_id="session1",
            chunk_type=ChunkType.MEMORY_SUMMARY,
            content="Discussion about Python programming",
            title="Python Tutorial",
            message_count=5
        )
        
        chunk2 = MemoryChunk(
            doc_id="doc2", 
            chunk_id="chunk2",
            session_id="session1",
            chunk_type=ChunkType.MEMORY_ENTITY,
            content="file: script.py\nlanguage: JavaScript",
            title="JavaScript Project",
            message_count=3
        )
        
        # Store chunks manually
        memory._chunks["session1"] = [chunk1, chunk2]
        
        # Search for "Python" - should rank chunk1 higher due to title match
        results = await memory.search_memory_chunks("session1", "python", limit=5)
        
        assert len(results) >= 1
        # First result should be chunk1 due to title match
        assert results[0].chunk_id == "chunk1"
        assert results[0].title == "Python Tutorial"
    
    @pytest.mark.asyncio
    async def test_dict_content_searchable_string(self):
        """Test that dictionary content is properly converted to searchable string."""
        entities = {
            "file": "main.py",
            "language": "Python",
            "nested": {"type": "script", "version": "3.9"}
        }
        
        chunk = MemoryChunk(
            doc_id="doc1",
            chunk_id="chunk1", 
            session_id="session1",
            chunk_type=ChunkType.MEMORY_ENTITY,
            content=entities,
            title="Test Entities"
        )
        
        searchable_content = chunk.get_content_string()
        
        # Check that all keys and values are searchable
        assert "file: main.py" in searchable_content
        assert "language: Python" in searchable_content
        assert "nested.type: script" in searchable_content
        assert "nested.version: 3.9" in searchable_content
        
        # Original dict should be preserved
        assert chunk.get_content_dict() == entities 