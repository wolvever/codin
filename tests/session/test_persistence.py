import pytest
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import asyncio
import aiofiles # Added import

# Assuming Session and LocalFilePersistor are correctly exported from codin.session
from codin.session import Session
from codin.session.persistence import LocalFilePersistor, HttpPersistor

from httpx import Request, Response, AsyncClient, MockTransport # Added for HttpPersistor tests


# --- Tests for LocalFilePersistor ---
@pytest.mark.asyncio
async def test_localfilepersistor_base_path_creation(tmp_path: Path):
    base_dir = tmp_path / "sessions_create_dir"
    assert not base_dir.exists()
    persistor = LocalFilePersistor(base_path=str(base_dir))
    assert base_dir.exists()
    assert base_dir.is_dir()

@pytest.mark.asyncio
async def test_localfilepersistor_save_and_load_session(tmp_path: Path):
    base_dir = tmp_path / "sessions_save_load"
    persistor = LocalFilePersistor(base_path=str(base_dir))

    original_created_at = datetime.now(timezone.utc)
    test_session = Session(
        session_id="test_s1",
        created_at=original_created_at,
        metrics={"calls": 10, "last_activity": original_created_at.timestamp()},
        context={"user": "test_user", "mode": "test"}
    )
    await persistor.save_session(test_session)
    file_path = base_dir / "test_s1.json"
    assert file_path.exists()
    loaded_session = await persistor.load_session("test_s1")
    assert loaded_session is not None
    assert loaded_session.session_id == test_session.session_id
    assert loaded_session.created_at == test_session.created_at
    assert loaded_session.created_at.tzinfo is not None
    assert loaded_session.context == test_session.context
    assert loaded_session.metrics == test_session.metrics

@pytest.mark.asyncio
async def test_localfilepersistor_load_non_existent_session(tmp_path: Path):
    base_dir = tmp_path / "sessions_load_non"
    persistor = LocalFilePersistor(base_path=str(base_dir))
    loaded_session = await persistor.load_session("non_existent_s1")
    assert loaded_session is None

@pytest.mark.asyncio
async def test_localfilepersistor_delete_session(tmp_path: Path):
    base_dir = tmp_path / "sessions_delete"
    persistor = LocalFilePersistor(base_path=str(base_dir))
    session_id_to_delete = "test_s2"
    created_at_for_delete_test = datetime.now(timezone.utc)
    session_to_delete = Session(
        session_id=session_id_to_delete,
        created_at=created_at_for_delete_test
    )
    await persistor.save_session(session_to_delete)
    file_path = base_dir / f"{session_id_to_delete}.json"
    assert file_path.exists()
    await persistor.delete_session(session_id_to_delete)
    assert not file_path.exists()

@pytest.mark.asyncio
async def test_localfilepersistor_delete_non_existent_session(tmp_path: Path):
    base_dir = tmp_path / "sessions_delete_non"
    persistor = LocalFilePersistor(base_path=str(base_dir))
    await persistor.delete_session("non_existent_s2")
    assert base_dir.exists()
    assert len(list(base_dir.iterdir())) == 0

@pytest.mark.asyncio
async def test_localfilepersistor_load_corrupted_session_file(tmp_path: Path):
    base_dir = tmp_path / "sessions_corrupted"
    persistor = LocalFilePersistor(base_path=str(base_dir))
    session_id = "corrupt_s1"
    file_path = base_dir / f"{session_id}.json"
    os.makedirs(base_dir, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("{'invalid_json': True,")
    loaded_session = await persistor.load_session(session_id)
    assert loaded_session is None

@pytest.mark.asyncio
async def test_localfilepersistor_concurrent_saves(tmp_path: Path):
    base_dir = tmp_path / "sessions_concurrent_save"
    persistor = LocalFilePersistor(base_path=str(base_dir))
    session_ids = [f"s_concurrent_{i}" for i in range(5)]
    sessions_to_save = [
        Session(session_id=sid, created_at=datetime.now(timezone.utc), context={"index": i})
        for i, sid in enumerate(session_ids)
    ]
    await asyncio.gather(*(persistor.save_session(s) for s in sessions_to_save))
    for s in sessions_to_save:
        file_path = base_dir / f"{s.session_id}.json"
        assert file_path.exists()
        loaded_s = await persistor.load_session(s.session_id)
        assert loaded_s is not None
        assert loaded_s.session_id == s.session_id
        assert loaded_s.context == s.context

@pytest.mark.asyncio
async def test_localfilepersistor_concurrent_loads(tmp_path: Path):
    base_dir = tmp_path / "sessions_concurrent_load"
    persistor = LocalFilePersistor(base_path=str(base_dir))
    session_ids = [f"s_concurrent_load_{i}" for i in range(5)]
    sessions_to_create = [
        Session(session_id=sid, created_at=datetime.now(timezone.utc), context={"load_index": i})
        for i, sid in enumerate(session_ids)
    ]
    for s in sessions_to_create:
        file_path = base_dir / f"{s.session_id}.json"
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(s.model_dump_json(indent=2))
        assert file_path.exists()
    loaded_sessions = await asyncio.gather(*(persistor.load_session(sid) for sid in session_ids))
    assert len(loaded_sessions) == len(session_ids)
    for i, loaded_s in enumerate(loaded_sessions):
        assert loaded_s is not None
        assert loaded_s.session_id == session_ids[i]
        assert loaded_s.context == {"load_index": i}

@pytest.mark.asyncio
async def test_localfilepersistor_datetime_precision_and_timezone(tmp_path: Path):
    base_dir = tmp_path / "sessions_datetime"
    persistor = LocalFilePersistor(base_path=str(base_dir))
    original_dt = datetime(2023, 10, 26, 12, 30, 45, 123456, tzinfo=timezone.utc)
    test_session = Session(
        session_id="dt_test_s1",
        created_at=original_dt,
        metrics={"last_activity": original_dt.timestamp() - 3600},
        context={"description": "Testing datetime handling"}
    )
    await persistor.save_session(test_session)
    loaded_session = await persistor.load_session("dt_test_s1")
    assert loaded_session is not None
    assert loaded_session.created_at == original_dt
    assert loaded_session.created_at.tzinfo == timezone.utc
    assert loaded_session.metrics["last_activity"] == test_session.metrics["last_activity"]
    offset = timedelta(hours=-5)
    tz_minus_5 = timezone(offset)
    original_dt_tz = datetime(2023, 11, 1, 8, 0, 0, 0, tzinfo=tz_minus_5)
    test_session_tz = Session(session_id="dt_test_s2", created_at=original_dt_tz)
    await persistor.save_session(test_session_tz)
    loaded_session_tz = await persistor.load_session("dt_test_s2")
    assert loaded_session_tz is not None
    assert loaded_session_tz.created_at == original_dt_tz
    assert loaded_session_tz.created_at.tzinfo is not None
    assert loaded_session_tz.created_at.tzinfo.utcoffset(None) == offset
    naive_dt = datetime(2023, 1, 1, 10, 0, 0)
    test_session_naive = Session(session_id="dt_test_s3", created_at=naive_dt)
    await persistor.save_session(test_session_naive)
    file_path_naive = base_dir / "dt_test_s3.json"
    async with aiofiles.open(file_path_naive, 'r', encoding='utf-8') as f:
        raw_content = await f.read()
    loaded_session_naive = await persistor.load_session("dt_test_s3")
    assert loaded_session_naive is not None
    if "Z" in raw_content or "+" in raw_content.split("T")[1]:
         assert loaded_session_naive.created_at.tzinfo is not None
    else:
        assert loaded_session_naive.created_at.tzinfo is None
        assert loaded_session_naive.created_at == naive_dt
    default_session = Session(session_id="dt_test_s4")
    await persistor.save_session(default_session)
    loaded_default_session = await persistor.load_session("dt_test_s4")
    assert loaded_default_session is not None
    assert (default_session.created_at.tzinfo is None and loaded_default_session.created_at.tzinfo is None) or \
           (default_session.created_at.tzinfo is not None and loaded_default_session.created_at.tzinfo is not None)
    assert abs(loaded_default_session.created_at.timestamp() - default_session.created_at.timestamp()) < 1


# --- Tests for HttpPersistor ---

# Define a global base_url for HttpPersistor tests
BASE_HTTP_URL = "http://testserver.com/api/sessions"

# Mock HTTP handler dictionary to store expected responses for paths
# This allows each test to set up its specific mock responses.
mock_responses = {}

def mock_http_handler(request: Request):
    path = request.url.path
    method = request.method

    # Check if there's a specific handler for this path and method
    if path in mock_responses and method in mock_responses[path]:
        handler_or_response = mock_responses[path][method]
        if callable(handler_or_response):
            return handler_or_response(request)
        return handler_or_response # Should be an httpx.Response

    # Default response if no specific handler is found
    print(f"Unhandled request: {method} {path}")
    return Response(404, json={"message": "Not Found by MockTransport"})


@pytest.fixture
def http_transport(request): # `request` fixture to allow per-test setup if needed
    # Clear mock_responses before each test that uses this fixture
    mock_responses.clear()
    return MockTransport(mock_http_handler)

@pytest.mark.asyncio
async def test_httppersistor_save_session(http_transport: MockTransport):
    session_id = "s_http_save"
    save_url_path = f"/api/sessions/{session_id}" # Path part of the URL

    def custom_save_handler(request: Request):
        assert request.method == "PUT"
        content = json.loads(request.content)
        assert content["session_id"] == session_id
        return Response(200, json={"message": "Session saved"})

    mock_responses[save_url_path] = {"PUT": custom_save_handler}

    async with AsyncClient(transport=http_transport, base_url=BASE_HTTP_URL.rsplit('/api/sessions', 1)[0]) as client:
        persistor = HttpPersistor(base_url=BASE_HTTP_URL, client=client)

        test_session = Session(session_id=session_id, created_at=datetime.now(timezone.utc))
        await persistor.save_session(test_session)
        # Assertion is within custom_save_handler
        await persistor.close() # Close client if persistor owns it, or manage externally

@pytest.mark.asyncio
async def test_httppersistor_load_session(http_transport: MockTransport):
    session_id = "s_http_load"
    load_url_path = f"/api/sessions/{session_id}"
    created_at_iso = datetime.now(timezone.utc).isoformat() # Pydantic will parse this

    sample_session_data = {"session_id": session_id, "created_at": created_at_iso, "context": {}, "metrics": {}}
    mock_responses[load_url_path] = {"GET": Response(200, json=sample_session_data)}

    async with AsyncClient(transport=http_transport, base_url=BASE_HTTP_URL.rsplit('/api/sessions', 1)[0]) as client:
        persistor = HttpPersistor(base_url=BASE_HTTP_URL, client=client)
        loaded_session = await persistor.load_session(session_id)

        assert loaded_session is not None
        assert loaded_session.session_id == session_id
        assert loaded_session.created_at.isoformat().startswith(created_at_iso.split('.')[0]) # Compare without microseconds for safety
        await persistor.close()

@pytest.mark.asyncio
async def test_httppersistor_load_session_not_found(http_transport: MockTransport):
    session_id = "s_http_notfound"
    url_path = f"/api/sessions/{session_id}"
    mock_responses[url_path] = {"GET": Response(404)}

    async with AsyncClient(transport=http_transport, base_url=BASE_HTTP_URL.rsplit('/api/sessions', 1)[0]) as client:
        persistor = HttpPersistor(base_url=BASE_HTTP_URL, client=client)
        loaded_session = await persistor.load_session(session_id)
        assert loaded_session is None
        await persistor.close()

@pytest.mark.asyncio
async def test_httppersistor_delete_session(http_transport: MockTransport):
    session_id = "s_http_delete"
    url_path = f"/api/sessions/{session_id}"

    delete_called = False
    def custom_delete_handler(request: Request):
        nonlocal delete_called
        assert request.method == "DELETE"
        delete_called = True
        return Response(204) # No content, success

    mock_responses[url_path] = {"DELETE": custom_delete_handler}

    async with AsyncClient(transport=http_transport, base_url=BASE_HTTP_URL.rsplit('/api/sessions', 1)[0]) as client:
        persistor = HttpPersistor(base_url=BASE_HTTP_URL, client=client)
        await persistor.delete_session(session_id)
        assert delete_called
        await persistor.close()

@pytest.mark.asyncio
async def test_httppersistor_delete_session_not_found(http_transport: MockTransport):
    session_id = "s_http_delete_notfound"
    url_path = f"/api/sessions/{session_id}"
    mock_responses[url_path] = {"DELETE": Response(404)}

    async with AsyncClient(transport=http_transport, base_url=BASE_HTTP_URL.rsplit('/api/sessions', 1)[0]) as client:
        persistor = HttpPersistor(base_url=BASE_HTTP_URL, client=client)
        await persistor.delete_session(session_id) # Should not raise error
        await persistor.close()

@pytest.mark.asyncio
async def test_httppersistor_save_session_error(http_transport: MockTransport):
    session_id = "s_http_error"
    url_path = f"/api/sessions/{session_id}"
    mock_responses[url_path] = {"PUT": Response(500, json={"message": "Internal Server Error"})}

    async with AsyncClient(transport=http_transport, base_url=BASE_HTTP_URL.rsplit('/api/sessions', 1)[0]) as client:
        persistor = HttpPersistor(base_url=BASE_HTTP_URL, client=client)
        # HttpPersistor prints error and does not re-raise by default.
        # If it were to re-raise, we'd use pytest.raises here.
        await persistor.save_session(Session(session_id=session_id, created_at=datetime.now(timezone.utc)))
        # Add assertion for logged error if logging is captured, or check stdout if needed.
        await persistor.close()

@pytest.mark.asyncio
async def test_httppersistor_load_session_http_error(http_transport: MockTransport):
    session_id = "s_http_load_error"
    url_path = f"/api/sessions/{session_id}"
    mock_responses[url_path] = {"GET": Response(503, json={"message": "Service Unavailable"})}

    async with AsyncClient(transport=http_transport, base_url=BASE_HTTP_URL.rsplit('/api/sessions', 1)[0]) as client:
        persistor = HttpPersistor(base_url=BASE_HTTP_URL, client=client)
        loaded_session = await persistor.load_session(session_id)
        assert loaded_session is None # Should return None on HTTP error other than 404
        await persistor.close()
