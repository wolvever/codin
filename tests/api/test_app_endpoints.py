import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from codin.actor.dispatcher import Dispatcher, DispatchResult
from codin.api.app import SubmitRequest, SubmitResponse, _sse_stream, create_app


@pytest.fixture
def mock_dispatcher():
    dispatcher = Mock(spec=Dispatcher)
    dispatcher.submit = AsyncMock(return_value="runner_123")
    dispatcher.get_status = AsyncMock(return_value=DispatchResult(
        runner_id="runner_123",
        request_id="req_123",
        task_id="task_123",
        status="running",
        agents=["agent_1"],
        metadata={"test": "data"}
    ))
    dispatcher.get_stream_queue = AsyncMock(return_value=asyncio.Queue())
    return dispatcher


@pytest.fixture
def app(mock_dispatcher):
    return create_app(dispatcher=mock_dispatcher)


@pytest.fixture  
def client(app):
    return TestClient(app)


class TestAPIEndpoints:
    
    def test_create_app_with_default_dispatcher(self):
        """Test app creation with default dispatcher."""
        app = create_app()
        assert app is not None
        assert hasattr(app, 'routes')
    
    def test_create_app_with_custom_dispatcher(self, mock_dispatcher):
        """Test app creation with custom dispatcher."""
        app = create_app(dispatcher=mock_dispatcher)
        assert app is not None
    
    @pytest.mark.asyncio
    async def test_submit_endpoint_success(self, app, mock_dispatcher):
        """Test successful submission to /v1/submit endpoint."""
        with TestClient(app) as client:
            request_data = {
                "a2a_request": {
                    "runner_id": "runner_123",
                    "request_id": "req_123", 
                    "agent_id": "test_agent",
                    "input": {"messages": []}
                }
            }
            
            response = client.post("/v1/submit", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["runner_id"] == "runner_123"
            
            # Verify dispatcher was called with replay_factory added
            mock_dispatcher.submit.assert_called_once()
            call_args = mock_dispatcher.submit.call_args[0][0]
            assert "replay_factory" in call_args
            assert callable(call_args["replay_factory"])
    
    @pytest.mark.asyncio
    async def test_submit_endpoint_dispatcher_error(self, app, mock_dispatcher):
        """Test submit endpoint when dispatcher raises an error."""
        mock_dispatcher.submit.side_effect = Exception("Dispatcher error")
        
        with TestClient(app) as client:
            request_data = {
                "a2a_request": {
                    "runner_id": "runner_123",
                    "request_id": "req_123"
                }
            }
            
            response = client.post("/v1/submit", json=request_data)
            
            assert response.status_code == 500
    
    def test_submit_endpoint_invalid_request(self, app):
        """Test submit endpoint with invalid request data."""
        with TestClient(app) as client:
            # Missing required field
            request_data = {"invalid": "data"}
            
            response = client.post("/v1/submit", json=request_data)
            
            assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_status_endpoint_success(self, app, mock_dispatcher):
        """Test successful status retrieval from /v1/status/{runner_id}."""
        with TestClient(app) as client:
            runner_id = "runner_123"
            
            response = client.get(f"/v1/status/{runner_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["runner_id"] == runner_id
            assert data["status"] == "running"
            assert data["task_id"] == "task_123"
            assert data["agents"] == ["agent_1"]
            
            mock_dispatcher.get_status.assert_called_once_with(runner_id)
    
    @pytest.mark.asyncio 
    async def test_status_endpoint_not_found(self, app, mock_dispatcher):
        """Test status endpoint when runner is not found."""
        mock_dispatcher.get_status.return_value = None
        
        with TestClient(app) as client:
            runner_id = "nonexistent_runner"
            
            response = client.get(f"/v1/status/{runner_id}")
            
            assert response.status_code == 404
            data = response.json()
            assert data["detail"] == "runner not found"
    
    @pytest.mark.asyncio
    async def test_status_endpoint_dispatcher_error(self, app, mock_dispatcher):
        """Test status endpoint when dispatcher raises an error."""
        mock_dispatcher.get_status.side_effect = Exception("Status error")
        
        with TestClient(app) as client:
            runner_id = "runner_123"
            
            response = client.get(f"/v1/status/{runner_id}")
            
            assert response.status_code == 500
    
    @pytest.mark.asyncio
    async def test_stream_endpoint_success(self, app, mock_dispatcher):
        """Test successful streaming from /v1/stream/{runner_id}."""
        # Create a queue with test data
        test_queue = asyncio.Queue()
        await test_queue.put({"type": "message", "data": "test"})
        await test_queue.put({"type": "complete", "data": "done"})
        await test_queue.put(None)  # End marker
        
        mock_dispatcher.get_stream_queue.return_value = test_queue
        
        with TestClient(app) as client:
            runner_id = "runner_123"
            
            response = client.get(f"/v1/stream/{runner_id}")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
            
            # Read the streaming content
            content = response.content.decode()
            assert "data: " in content
            assert "test" in content
            assert "done" in content
    
    @pytest.mark.asyncio
    async def test_stream_endpoint_dispatcher_error(self, app, mock_dispatcher):
        """Test stream endpoint when dispatcher raises an error."""
        mock_dispatcher.get_stream_queue.side_effect = Exception("Stream error")
        
        with TestClient(app) as client:
            runner_id = "runner_123"
            
            response = client.get(f"/v1/stream/{runner_id}")
            
            assert response.status_code == 500


class TestSSEStream:
    
    @pytest.mark.asyncio
    async def test_sse_stream_with_data(self):
        """Test SSE stream formatting with data."""
        queue = asyncio.Queue()
        await queue.put({"message": "hello"})
        await queue.put({"status": "complete"})
        await queue.put(None)  # End marker
        
        chunks = []
        async for chunk in _sse_stream(queue):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0] == 'data: {"message": "hello"}\n\n'
        assert chunks[1] == 'data: {"status": "complete"}\n\n'
    
    @pytest.mark.asyncio
    async def test_sse_stream_empty_queue(self):
        """Test SSE stream with immediately terminated queue."""
        queue = asyncio.Queue()
        await queue.put(None)  # Immediate end marker
        
        chunks = []
        async for chunk in _sse_stream(queue):
            chunks.append(chunk)
        
        assert len(chunks) == 0
    
    @pytest.mark.asyncio
    async def test_sse_stream_json_serialization(self):
        """Test SSE stream handles complex JSON serialization."""
        queue = asyncio.Queue()
        complex_data = {
            "nested": {"data": [1, 2, 3]},
            "timestamp": "2023-01-01T00:00:00Z",
            "boolean": True,
            "null_value": None
        }
        await queue.put(complex_data)
        await queue.put(None)
        
        chunks = []
        async for chunk in _sse_stream(queue):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        # Verify JSON is properly formatted
        data_line = chunks[0]
        assert data_line.startswith("data: ")
        assert data_line.endswith("\n\n")
        
        # Extract and parse JSON
        json_str = data_line[6:-2]  # Remove "data: " and "\n\n"
        parsed = json.loads(json_str)
        assert parsed == complex_data


class TestRequestResponseModels:
    
    def test_submit_request_validation(self):
        """Test SubmitRequest model validation."""
        valid_data = {"a2a_request": {"key": "value"}}
        request = SubmitRequest(**valid_data)
        assert request.a2a_request == {"key": "value"}
    
    def test_submit_request_invalid_data(self):
        """Test SubmitRequest with invalid data."""
        with pytest.raises(Exception):  # Pydantic validation error
            SubmitRequest(invalid_field="value")
    
    def test_submit_response_creation(self):
        """Test SubmitResponse model creation."""
        response = SubmitResponse(runner_id="test_runner")
        assert response.runner_id == "test_runner"
    
    def test_submit_response_json_serialization(self):
        """Test SubmitResponse JSON serialization."""
        response = SubmitResponse(runner_id="test_runner")
        json_data = response.model_dump()
        assert json_data == {"runner_id": "test_runner"}


class TestIntegration:
    
    @pytest.mark.asyncio
    async def test_full_submit_to_stream_flow(self, mock_dispatcher):
        """Test full flow from submit to stream."""
        app = create_app(dispatcher=mock_dispatcher)
        
        # Setup stream queue
        stream_queue = asyncio.Queue()
        await stream_queue.put({"type": "start", "runner_id": "runner_123"})
        await stream_queue.put({"type": "result", "data": "completed"})
        await stream_queue.put(None)
        
        mock_dispatcher.get_stream_queue.return_value = stream_queue
        
        with TestClient(app) as client:
            # Submit request
            submit_response = client.post("/v1/submit", json={
                "a2a_request": {"test": "data"}
            })
            assert submit_response.status_code == 200
            runner_id = submit_response.json()["runner_id"]
            
            # Check status
            status_response = client.get(f"/v1/status/{runner_id}")
            assert status_response.status_code == 200
            
            # Stream results
            stream_response = client.get(f"/v1/stream/{runner_id}")
            assert stream_response.status_code == 200
            content = stream_response.content.decode()
            assert "start" in content
            assert "completed" in content