import pytest
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock # For HttpPersistor test

from codin.session import SessionManager, Session
from codin.session.persistence import LocalFilePersistor, HttpPersistor, SessionPersistor
from httpx import AsyncClient, MockTransport, Response, Request # For HttpPersistor test


@pytest.mark.asyncio
async def test_session_manager_no_endpoint_in_memory(mocker): # Added mocker for potential future use
    manager = SessionManager()
    session_id = "s_mem_1"

    async with manager.session(session_id) as s1:
        assert s1.session_id == session_id
        s1.context["data"] = "test_data_in_memory"
        assert manager.get_session(session_id) is s1 # Should be in memory within context

    # s1 is removed from _sessions cache on context exit by default
    assert manager.get_session(session_id) is None

    # Create a new session with the same ID, it should be fresh
    async with manager.session(session_id) as s2:
        assert s2.session_id == session_id
        assert "data" not in s2.context # Should be a new session, as nothing was persisted

    await manager.cleanup()


@pytest.mark.asyncio
async def test_session_manager_with_localfile_endpoint(tmp_path: Path, mocker):
    file_endpoint = str(tmp_path / "file_sessions_sm")
    manager = SessionManager(endpoint=file_endpoint)
    session_id = "s_local_sm"
    original_session_data = {"file_data": "persisted_sm", "ops": 1}

    async with manager.session(session_id) as s1:
        assert s1.session_id == session_id
        s1.context.update(original_session_data)

    # Session s1 should have been saved to file on successful context exit
    # and removed from in-memory cache.
    assert manager.get_session(session_id) is None

    # Now, when we request it again, it should be loaded from the file
    async with manager.session(session_id) as s2:
        assert s2.session_id == session_id
        assert s2.context.get("file_data") == "persisted_sm"
        assert s2.context.get("ops") == 1

    assert (Path(file_endpoint) / f"{session_id}.json").exists()
    await manager.cleanup()


@pytest.mark.asyncio
async def test_session_manager_with_http_endpoint_loads_and_saves(mocker):
    session_id = "s_http_sm"

    # Mock HttpPersistor instance that SessionManager will create
    mock_persistor_instance = AsyncMock(spec=HttpPersistor)
    mock_persistor_instance.load_session.return_value = Session(session_id=session_id, context={"loaded": True}, created_at=datetime.now(timezone.utc))

    # Patch the HttpPersistor class within the base module where SessionManager looks for it
    with patch('codin.session.base.HttpPersistor', return_value=mock_persistor_instance) as mock_http_persistor_class:
        manager = SessionManager(endpoint="http://dummy-endpoint.com/api")

        # Check if the manager's persistor is indeed our mocked instance
        assert manager._persistor is mock_persistor_instance

        # Test 1: Load existing session, modify, and save
        async with manager.session(session_id) as s1:
            assert s1.context.get("loaded") is True, "Session should have been loaded with initial context."
            s1.context["modified"] = True # Modify the session

        mock_persistor_instance.load_session.assert_called_once_with(session_id)
        mock_persistor_instance.save_session.assert_called_once()
        saved_arg_s1 = mock_persistor_instance.save_session.call_args[0][0]
        assert isinstance(saved_arg_s1, Session)
        assert saved_arg_s1.session_id == session_id
        assert saved_arg_s1.context.get("modified") is True
        assert saved_arg_s1.context.get("loaded") is True # Original loaded data should also be there

        # Reset mocks for the next part of the test
        mock_persistor_instance.load_session.reset_mock()
        mock_persistor_instance.save_session.reset_mock()

        # Simulate session not found on next load attempt for a new ID
        mock_persistor_instance.load_session.return_value = None
        new_session_id = "s_http_sm_new"

        # Test 2: Create new session, modify, and save
        async with manager.session(new_session_id) as s2:
            assert not s2.context.get("loaded"), "New session should not have 'loaded' context."
            s2.context["new_data"] = "fresh"

        mock_persistor_instance.load_session.assert_called_once_with(new_session_id)
        mock_persistor_instance.save_session.assert_called_once()
        saved_arg_s2 = mock_persistor_instance.save_session.call_args[0][0]
        assert isinstance(saved_arg_s2, Session)
        assert saved_arg_s2.session_id == new_session_id
        assert saved_arg_s2.context.get("new_data") == "fresh"

        # Test cleanup calls persistor.close if available
        # Add a close method to the AsyncMock if it's not already part of the spec
        # or if spec doesn't perfectly define it as async.
        if not hasattr(mock_persistor_instance, 'close') or not isinstance(mock_persistor_instance.close, AsyncMock):
            mock_persistor_instance.close = AsyncMock()

        await manager.cleanup()
        mock_persistor_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_session_manager_cleanup_saves_active_sessions(tmp_path: Path, mocker):
    file_endpoint = str(tmp_path / "cleanup_sm_sessions")
    manager = SessionManager(endpoint=file_endpoint)
    session_id1 = "s_cleanup_active1"
    session_id2 = "s_cleanup_active2"

    # Create sessions but don't exit their contexts, so they remain in _sessions
    s1 = await manager.get_or_create_session(session_id1)
    s1.context["data1"] = "active1_data"

    s2 = await manager.get_or_create_session(session_id2)
    s2.context["data2"] = "active2_data"

    assert session_id1 in manager._sessions
    assert session_id2 in manager._sessions

    assert isinstance(manager._persistor, LocalFilePersistor)
    # Spy on the save_session method of the actual LocalFilePersistor instance
    spy_save = mocker.spy(manager._persistor, 'save_session')

    await manager.cleanup()

    assert spy_save.call_count == 2

    # Verify files were written
    s1_file = Path(file_endpoint) / f"{session_id1}.json"
    assert s1_file.exists()
    s1_data_on_disk = json.loads(s1_file.read_text())
    assert s1_data_on_disk["context"]["data1"] == "active1_data"

    s2_file = Path(file_endpoint) / f"{session_id2}.json"
    assert s2_file.exists()
    s2_data_on_disk = json.loads(s2_file.read_text())
    assert s2_data_on_disk["context"]["data2"] == "active2_data"

    # Check that in-memory cache is cleared after cleanup
    assert not manager._sessions


@pytest.mark.asyncio
async def test_session_manager_persistor_init_schemes(tmp_path: Path):
    # Test file scheme
    file_endpoint_abs = tmp_path / "file_abs"
    manager_file_abs = SessionManager(endpoint=str(file_endpoint_abs))
    assert isinstance(manager_file_abs._persistor, LocalFilePersistor)
    assert manager_file_abs._persistor.base_path == str(file_endpoint_abs)

    file_endpoint_uri = f"file://{file_endpoint_abs.as_posix()}"
    manager_file_uri = SessionManager(endpoint=file_endpoint_uri)
    assert isinstance(manager_file_uri._persistor, LocalFilePersistor)
    # Path conversion from URI can be tricky, check it's what LocalFilePersistor expects
    # For now, ensuring it's an instance of LocalFilePersistor is key.
    # The actual path might differ slightly based on OS and parsing (e.g. leading slashes)
    # but LocalFilePersistor should handle it.

    # Test HTTP scheme
    http_endpoint = "http://example.com/sessions"
    manager_http = SessionManager(endpoint=http_endpoint)
    assert isinstance(manager_http._persistor, HttpPersistor)
    assert manager_http._persistor.base_url == http_endpoint

    https_endpoint = "https://secure.example.com/sessions"
    manager_https = SessionManager(endpoint=https_endpoint)
    assert isinstance(manager_https._persistor, HttpPersistor)
    assert manager_https._persistor.base_url == https_endpoint

    # Test no persistor if endpoint is None or unrecognized scheme
    manager_none = SessionManager(endpoint=None)
    assert manager_none._persistor is None

    manager_unknown_scheme = SessionManager(endpoint="unknownscheme://whatever")
    assert manager_unknown_scheme._persistor is None

    # Test Windows file URI if possible (hard to make OS-specific tests robustly here)
    if os.name == 'nt':
        win_path_uri = "file:///C:/sessions_test" # Typical representation
        # On Windows, urlparse("file:///C:/foo").path gives "/C:/foo"
        # urlparse("file:C:/foo").path gives "C:/foo"
        # LocalFilePersistor's __init__ should handle these.
        manager_win_file = SessionManager(endpoint=win_path_uri)
        assert isinstance(manager_win_file._persistor, LocalFilePersistor)
        # Expecting "C:\sessions_test" after path normalization by persistor.
        # This assertion is tricky without actually running on Windows.
        # For now, instance check is the main goal.

    await manager_file_abs.cleanup() # Cleanup to close any open persistor clients (Http)
    await manager_file_uri.cleanup()
    await manager_http.cleanup()
    await manager_https.cleanup()
    await manager_none.cleanup()
    await manager_unknown_scheme.cleanup()
    if os.name == 'nt' and 'manager_win_file' in locals():
        await manager_win_file.cleanup()

@pytest.mark.asyncio
async def test_session_manager_explicit_close_session(tmp_path: Path):
    file_endpoint = str(tmp_path / "explicit_close_test")
    manager = SessionManager(endpoint=file_endpoint)
    session_id = "s_explicit_close"

    # Create a session and put some data in it using the context manager
    async with manager.session(session_id) as s:
        s.context["data"] = "initial_data"

    # At this point, session is saved (if persistor is present) and removed from memory cache.
    assert manager.get_session(session_id) is None
    session_file = Path(file_endpoint) / f"{session_id}.json"
    assert session_file.exists() # Verify it was saved by context manager

    # Load it again (or create if not found by persistor)
    s_reloaded = await manager.get_or_create_session(session_id)
    assert s_reloaded.context.get("data") == "initial_data"
    assert session_id in manager._sessions # Now it's in memory

    # Explicitly close it
    await manager.close_session(session_id)
    assert manager.get_session(session_id) is None # Should be removed from memory

    # The file should still exist as close_session doesn't delete from persistence
    assert session_file.exists()

    await manager.cleanup()

@pytest.mark.asyncio
async def test_session_manager_cleanup_inactive_sessions(tmp_path: Path, mocker):
    file_endpoint = str(tmp_path / "inactive_cleanup_test")
    manager = SessionManager(endpoint=file_endpoint)

    s_active_id = "s_active"
    s_inactive_id = "s_inactive"

    # Create an active session (last_activity will be recent)
    s_active = await manager.get_or_create_session(s_active_id)
    s_active.context["status"] = "active"

    # Create an inactive session by manually setting its last_activity metric far in the past
    s_inactive = await manager.get_or_create_session(s_inactive_id)
    s_inactive.context["status"] = "inactive"
    # Simulate past activity: current time - (max_age_seconds + some buffer)
    # Default max_age_seconds is 3600.
    s_inactive.metrics["last_activity"] = datetime.now(timezone.utc).timestamp() - (3600 + 60)

    assert s_active_id in manager._sessions
    assert s_inactive_id in manager._sessions

    assert isinstance(manager._persistor, LocalFilePersistor)
    spy_save = mocker.spy(manager._persistor, 'save_session')

    cleaned_count = await manager.cleanup_inactive_sessions(max_age_seconds=3600)
    assert cleaned_count == 1

    # Inactive session should have been saved and removed from memory
    spy_save.assert_called_once_with(s_inactive) # Check it was s_inactive that got saved
    assert s_inactive_id not in manager._sessions
    inactive_file = Path(file_endpoint) / f"{s_inactive_id}.json"
    assert inactive_file.exists()

    # Active session should still be in memory and not saved by this specific call
    assert s_active_id in manager._sessions
    active_file = Path(file_endpoint) / f"{s_active_id}.json"
    assert not active_file.exists() # Not saved by cleanup_inactive_sessions

    await manager.cleanup() # Full cleanup will save the active one
    assert active_file.exists()
