import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from codin.session.base import SessionContext, SessionManager
from codin.session.persistence import FileSessionPersistence, SessionPersistence


class TestSessionContext:
    
    def test_session_context_creation(self):
        """Test creating session context."""
        context = SessionContext(
            session_id="session_123",
            user_id="user_456",
            created_at=datetime.now(),
            metadata={"project": "test_project"}
        )
        
        assert context.session_id == "session_123"
        assert context.user_id == "user_456"
        assert context.metadata["project"] == "test_project"
        assert isinstance(context.created_at, datetime)
    
    def test_session_context_defaults(self):
        """Test session context with defaults."""
        context = SessionContext(session_id="test_session")
        
        assert context.session_id == "test_session"
        assert context.user_id is None or context.user_id == ""
        assert isinstance(context.metadata, dict)
        assert len(context.metadata) == 0
        assert isinstance(context.created_at, datetime)
    
    def test_session_context_serialization(self):
        """Test serializing session context to dict."""
        context = SessionContext(
            session_id="test_session",
            user_id="test_user",
            metadata={"key": "value"}
        )
        
        data = context.to_dict()
        
        assert data["session_id"] == "test_session"
        assert data["user_id"] == "test_user"
        assert data["metadata"]["key"] == "value"
        assert "created_at" in data
    
    def test_session_context_deserialization(self):
        """Test deserializing session context from dict."""
        data = {
            "session_id": "test_session",
            "user_id": "test_user",
            "created_at": "2023-01-01T12:00:00Z",
            "metadata": {"project": "test"}
        }
        
        context = SessionContext.from_dict(data)
        
        assert context.session_id == "test_session"
        assert context.user_id == "test_user"
        assert context.metadata["project"] == "test"
    
    def test_session_context_update_metadata(self):
        """Test updating session context metadata."""
        context = SessionContext(
            session_id="test_session",
            metadata={"original": "value"}
        )
        
        context.update_metadata({"new": "data", "original": "updated"})
        
        assert context.metadata["new"] == "data"
        assert context.metadata["original"] == "updated"
    
    def test_session_context_get_age(self):
        """Test getting session age."""
        import time
        old_time = datetime.now()
        time.sleep(0.01)  # Small delay
        
        context = SessionContext(
            session_id="test_session",
            created_at=old_time
        )
        
        age = context.get_age()
        assert age.total_seconds() > 0


class TestFileSessionPersistence:
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def persistence(self, temp_dir):
        return FileSessionPersistence(storage_path=temp_dir)
    
    @pytest.mark.asyncio
    async def test_save_session_context(self, persistence, temp_dir):
        """Test saving session context to file."""
        context = SessionContext(
            session_id="test_session",
            user_id="test_user",
            metadata={"project": "test_project"}
        )
        
        await persistence.save_context(context)
        
        # Check file was created
        session_file = temp_dir / "test_session" / "context.json"
        assert session_file.exists()
    
    @pytest.mark.asyncio
    async def test_load_session_context(self, persistence, temp_dir):
        """Test loading session context from file."""
        # Save first
        original_context = SessionContext(
            session_id="load_test",
            user_id="test_user",
            metadata={"test": "data"}
        )
        await persistence.save_context(original_context)
        
        # Load back
        loaded_context = await persistence.load_context("load_test")
        
        assert loaded_context is not None
        assert loaded_context.session_id == "load_test"
        assert loaded_context.user_id == "test_user"
        assert loaded_context.metadata["test"] == "data"
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_session(self, persistence):
        """Test loading non-existent session context."""
        context = await persistence.load_context("nonexistent_session")
        
        assert context is None
    
    @pytest.mark.asyncio
    async def test_save_session_data(self, persistence):
        """Test saving arbitrary session data."""
        session_id = "data_test"
        test_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "state": {"current_task": "testing"}
        }
        
        await persistence.save_data(session_id, "conversation", test_data)
        
        # Verify data was saved
        loaded_data = await persistence.load_data(session_id, "conversation")
        assert loaded_data == test_data
    
    @pytest.mark.asyncio
    async def test_load_session_data(self, persistence):
        """Test loading session data."""
        session_id = "load_data_test"
        data_key = "test_data"
        test_data = {"key": "value", "number": 42}
        
        await persistence.save_data(session_id, data_key, test_data)
        loaded_data = await persistence.load_data(session_id, data_key)
        
        assert loaded_data == test_data
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_data(self, persistence):
        """Test loading non-existent session data."""
        data = await persistence.load_data("nonexistent", "data_key")
        
        assert data is None
    
    @pytest.mark.asyncio
    async def test_delete_session(self, persistence, temp_dir):
        """Test deleting a session."""
        session_id = "delete_test"
        
        # Create session with context and data
        context = SessionContext(session_id=session_id)
        await persistence.save_context(context)
        await persistence.save_data(session_id, "test_data", {"key": "value"})
        
        # Verify files exist
        session_dir = temp_dir / session_id
        assert session_dir.exists()
        
        # Delete session
        await persistence.delete_session(session_id)
        
        # Verify files are removed
        assert not session_dir.exists()
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, persistence):
        """Test listing all sessions."""
        # Create multiple sessions
        session_ids = ["session_1", "session_2", "session_3"]
        
        for session_id in session_ids:
            context = SessionContext(session_id=session_id)
            await persistence.save_context(context)
        
        # List sessions
        sessions = await persistence.list_sessions()
        
        assert len(sessions) >= 3
        session_ids_found = {session.session_id for session in sessions}
        for session_id in session_ids:
            assert session_id in session_ids_found
    
    @pytest.mark.asyncio
    async def test_session_exists(self, persistence):
        """Test checking if session exists."""
        session_id = "exists_test"
        
        # Should not exist initially
        assert await persistence.session_exists(session_id) is False
        
        # Create session
        context = SessionContext(session_id=session_id)
        await persistence.save_context(context)
        
        # Should exist now
        assert await persistence.session_exists(session_id) is True
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, persistence):
        """Test concurrent persistence operations."""
        session_id = "concurrent_test"
        
        # Concurrent saves
        tasks = []
        for i in range(10):
            task = persistence.save_data(session_id, f"key_{i}", {"value": i})
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Verify all data was saved
        for i in range(10):
            data = await persistence.load_data(session_id, f"key_{i}")
            assert data["value"] == i


class TestSessionManager:
    
    @pytest.fixture
    def mock_persistence(self):
        persistence = Mock(spec=SessionPersistence)
        persistence.save_context = AsyncMock()
        persistence.load_context = AsyncMock()
        persistence.save_data = AsyncMock()
        persistence.load_data = AsyncMock()
        persistence.delete_session = AsyncMock()
        persistence.list_sessions = AsyncMock(return_value=[])
        persistence.session_exists = AsyncMock(return_value=False)
        return persistence
    
    @pytest.fixture
    def session_manager(self, mock_persistence):
        return SessionManager(persistence=mock_persistence)
    
    @pytest.mark.asyncio
    async def test_create_session(self, session_manager, mock_persistence):
        """Test creating a new session."""
        session_id = "new_session"
        user_id = "test_user"
        metadata = {"project": "test"}
        
        context = await session_manager.create_session(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata
        )
        
        assert context.session_id == session_id
        assert context.user_id == user_id
        assert context.metadata["project"] == "test"
        mock_persistence.save_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_session_existing(self, session_manager, mock_persistence):
        """Test getting an existing session."""
        session_id = "existing_session"
        expected_context = SessionContext(session_id=session_id, user_id="test_user")
        
        mock_persistence.load_context.return_value = expected_context
        
        context = await session_manager.get_session(session_id)
        
        assert context == expected_context
        mock_persistence.load_context.assert_called_once_with(session_id)
    
    @pytest.mark.asyncio
    async def test_get_session_nonexistent(self, session_manager, mock_persistence):
        """Test getting a non-existent session."""
        mock_persistence.load_context.return_value = None
        
        context = await session_manager.get_session("nonexistent")
        
        assert context is None
    
    @pytest.mark.asyncio
    async def test_update_session_metadata(self, session_manager, mock_persistence):
        """Test updating session metadata."""
        session_id = "update_test"
        original_context = SessionContext(
            session_id=session_id,
            metadata={"original": "value"}
        )
        
        mock_persistence.load_context.return_value = original_context
        
        updated_context = await session_manager.update_session_metadata(
            session_id,
            {"new": "data", "original": "updated"}
        )
        
        assert updated_context.metadata["new"] == "data"
        assert updated_context.metadata["original"] == "updated"
        mock_persistence.save_context.assert_called()
    
    @pytest.mark.asyncio
    async def test_save_session_data(self, session_manager, mock_persistence):
        """Test saving session data."""
        session_id = "save_data_test"
        data_key = "test_data"
        data = {"key": "value"}
        
        await session_manager.save_session_data(session_id, data_key, data)
        
        mock_persistence.save_data.assert_called_once_with(session_id, data_key, data)
    
    @pytest.mark.asyncio
    async def test_load_session_data(self, session_manager, mock_persistence):
        """Test loading session data."""
        session_id = "load_data_test"
        data_key = "test_data"
        expected_data = {"key": "value"}
        
        mock_persistence.load_data.return_value = expected_data
        
        data = await session_manager.load_session_data(session_id, data_key)
        
        assert data == expected_data
        mock_persistence.load_data.assert_called_once_with(session_id, data_key)
    
    @pytest.mark.asyncio
    async def test_delete_session(self, session_manager, mock_persistence):
        """Test deleting a session."""
        session_id = "delete_test"
        
        await session_manager.delete_session(session_id)
        
        mock_persistence.delete_session.assert_called_once_with(session_id)
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, session_manager, mock_persistence):
        """Test listing all sessions."""
        expected_sessions = [
            SessionContext(session_id="session_1"),
            SessionContext(session_id="session_2")
        ]
        mock_persistence.list_sessions.return_value = expected_sessions
        
        sessions = await session_manager.list_sessions()
        
        assert sessions == expected_sessions
        mock_persistence.list_sessions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_session_exists(self, session_manager, mock_persistence):
        """Test checking if session exists."""
        session_id = "exists_test"
        mock_persistence.session_exists.return_value = True
        
        exists = await session_manager.session_exists(session_id)
        
        assert exists is True
        mock_persistence.session_exists.assert_called_once_with(session_id)
    
    @pytest.mark.asyncio
    async def test_get_or_create_session_existing(self, session_manager, mock_persistence):
        """Test get_or_create with existing session."""
        session_id = "existing"
        existing_context = SessionContext(session_id=session_id)
        
        mock_persistence.load_context.return_value = existing_context
        
        context = await session_manager.get_or_create_session(session_id)
        
        assert context == existing_context
        mock_persistence.save_context.assert_not_called()  # Should not create new
    
    @pytest.mark.asyncio
    async def test_get_or_create_session_new(self, session_manager, mock_persistence):
        """Test get_or_create with new session."""
        session_id = "new_session"
        
        mock_persistence.load_context.return_value = None  # Doesn't exist
        
        context = await session_manager.get_or_create_session(session_id)
        
        assert context.session_id == session_id
        mock_persistence.save_context.assert_called_once()  # Should create new


class TestSessionIntegration:
    
    @pytest.mark.asyncio
    async def test_full_session_lifecycle(self):
        """Test complete session lifecycle with file persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = FileSessionPersistence(storage_path=Path(tmpdir))
            manager = SessionManager(persistence=persistence)
            
            # Create session
            session_id = "lifecycle_test"
            context = await manager.create_session(
                session_id=session_id,
                user_id="test_user",
                metadata={"project": "integration_test"}
            )
            
            # Save some data
            conversation_data = {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"}
                ]
            }
            await manager.save_session_data(session_id, "conversation", conversation_data)
            
            # Update metadata
            updated_context = await manager.update_session_metadata(
                session_id,
                {"status": "active", "last_activity": "now"}
            )
            
            # Verify persistence
            loaded_context = await manager.get_session(session_id)
            assert loaded_context.session_id == session_id
            assert loaded_context.metadata["project"] == "integration_test"
            assert loaded_context.metadata["status"] == "active"
            
            loaded_data = await manager.load_session_data(session_id, "conversation")
            assert loaded_data == conversation_data
            
            # List sessions
            sessions = await manager.list_sessions()
            assert len(sessions) >= 1
            assert any(s.session_id == session_id for s in sessions)
            
            # Delete session
            await manager.delete_session(session_id)
            
            # Verify deletion
            assert await manager.session_exists(session_id) is False