import pytest
from unittest.mock import patch, MagicMock

from codin.tool.fetch_tool import FetchTool, FetchInput

@pytest.mark.asyncio
async def test_fetch_tool_success():
    fetch_tool = FetchTool()
    args = FetchInput(url='http://example.com')

    # Mock the requests.get call
    mock_response = MagicMock()
    mock_response.text = 'Example Domain content'
    mock_response.raise_for_status = MagicMock()

    with patch('requests.get', return_value=mock_response) as mock_get:
        result = await fetch_tool.run(args, tool_context=MagicMock())
        mock_get.assert_called_once_with('http://example.com')
        assert result == 'Example Domain content'

@pytest.mark.asyncio
async def test_fetch_tool_failure():
    fetch_tool = FetchTool()
    args = FetchInput(url='http://invalid-url-that-does-not-exist.com')

    # Mock requests.get to raise an exception
    with patch('requests.get', side_effect=requests.exceptions.RequestException('Test error')) as mock_get:
        result = await fetch_tool.run(args, tool_context=MagicMock())
        mock_get.assert_called_once_with('http://invalid-url-that-does-not-exist.com')
        assert 'Error: Could not fetch URL. Test error' in result
