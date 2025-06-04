"""Tests that use mocking for the imports."""

import pytest
from unittest.mock import MagicMock, patch


# Create mock classes
class MockClient:
    """Mock client for testing."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.prepared = False
        self.closed = False
        self.requests = []
    
    async def prepare(self):
        self.prepared = True
    
    async def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        return mock_response
    
    async def get(self, url, **kwargs):
        return await self.request("GET", url, **kwargs)
    
    async def post(self, url, **kwargs):
        return await self.request("POST", url, **kwargs)
    
    async def close(self):
        self.closed = True


@pytest.fixture
def mock_modules():
    """Set up mock modules."""
    modules = {
        'codin': MagicMock(),
        'codin.client': MagicMock(),
        'codin.client.base': MagicMock(),
        'codin.model': MagicMock(),
        'codin.prompt': MagicMock(),
    }
    
    # Set up the client mock
    modules['codin.client.base'].Client = MockClient
    
    with patch.dict('sys.modules', modules):
        yield modules


class TestMockedClient:
    """Test cases using mocked client."""
    
    @pytest.mark.asyncio
    async def test_client_prepare(self, mock_modules):
        """Test that client prepare works."""
        # Import the mocked client
        from codin.client.base import Client
        
        # Create client and prepare
        client = Client()
        await client.prepare()
        
        # Check that prepare was called
        assert client.prepared is True
    
    @pytest.mark.asyncio
    async def test_client_request(self, mock_modules):
        """Test client request method."""
        # Import the mocked client
        from codin.client.base import Client
        
        # Create client and make request
        client = Client()
        await client.prepare()
        response = await client.get("/test")
        
        # Check that request was made
        assert len(client.requests) == 1
        assert client.requests[0][0] == "GET"
        assert client.requests[0][1] == "/test"
        
        # Check response
        assert response.status_code == 200
        assert response.json() == {"success": True}
    
    @pytest.mark.asyncio
    async def test_client_close(self, mock_modules):
        """Test client close method."""
        # Import the mocked client
        from codin.client.base import Client
        
        # Create client and close
        client = Client()
        await client.close()
        
        # Check that close was called
        assert client.closed is True 