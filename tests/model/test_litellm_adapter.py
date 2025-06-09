import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os
import asyncio

# Conditional import for LiteLLM
try:
    import litellm
    _HAS_LITELLM = True
except ImportError:
    _HAS_LITELLM = False

from src.codin.model.litellm_adapter import LiteLLMAdapter
from src.codin.model.config import ModelConfig

# Skip all tests in this file if litellm is not installed
pytestmark = pytest.mark.skipif(not _HAS_LITELLM, reason="litellm not installed")

@pytest.fixture
def litellm_env_vars(monkeypatch):
    """Set up environment variables for LiteLLM tests."""
    monkeypatch.delenv("LITELLM_API_KEY", raising=False) # Generic LiteLLM key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False) # Example provider key
    monkeypatch.delenv("LITELLM_CACHE", raising=False)
    # Set a specific provider key for testing if config passes it
    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "env-provider-key-for-litellm")

class TestLiteLLMAdapter:

    def test_init_with_model_and_config(self):
        config = ModelConfig(
            api_key="cfg-litellm-key", # Could be a generic key or specific one
            timeout=120.0,
            max_retries=3
        )
        adapter = LiteLLMAdapter(model="gpt-3.5-turbo", config=config)
        assert adapter.model == "gpt-3.5-turbo"
        assert adapter.config.api_key == "cfg-litellm-key"
        assert adapter.config.timeout == 120.0
        assert adapter.config.max_retries == 3

    def test_init_model_only(self):
        adapter = LiteLLMAdapter(model="claude-instant-1.2")
        assert adapter.model == "claude-instant-1.2"
        assert adapter.config is not None # Default ModelConfig
        assert adapter.config.api_key is None

    # test_prepare_is_noop will be removed as prepare() method itself will be removed.

    @pytest.mark.asyncio
    @patch('litellm.completion') # Patch the synchronous litellm.completion
    async def test_generate_non_streaming(self, mock_litellm_completion, litellm_env_vars):
        """Test non-streaming generate call."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "LiteLLM test response"
        mock_litellm_completion.return_value = mock_response

        config = ModelConfig(api_key="test_key_direct", timeout=90, max_retries=2)
        adapter = LiteLLMAdapter(model="gpt-3.5-turbo", config=config)
        # adapter.prepare() was a no-op and will be removed.

        response = await adapter.generate("Hello LiteLLM")
        assert response == "LiteLLM test response"

        mock_litellm_completion.assert_called_once()
        call_kwargs = mock_litellm_completion.call_args.kwargs
        assert call_kwargs['model'] == "gpt-3.5-turbo"
        assert call_kwargs['messages'] == [{"role": "user", "content": "Hello LiteLLM"}]
        assert call_kwargs['stream'] is False
        assert call_kwargs['api_key'] == "test_key_direct" # From config
        assert call_kwargs['request_timeout'] == 90 # From config
        assert call_kwargs['num_retries'] == 2 # From config

    @pytest.mark.asyncio
    @patch('litellm.completion') # Patch the synchronous litellm.completion
    async def test_generate_streaming(self, mock_litellm_completion, litellm_env_vars):
        """Test streaming generate call."""

        # Mocking the iterator behavior for streaming chunks
        mock_stream_chunk_1 = MagicMock()
        mock_stream_chunk_1.choices = [MagicMock()]
        mock_stream_chunk_1.choices[0].delta.content = "Hello "

        mock_stream_chunk_2 = MagicMock()
        mock_stream_chunk_2.choices = [MagicMock()]
        mock_stream_chunk_2.choices[0].delta.content = "LiteLLM!"

        # This mock will be an iterator
        mock_streaming_response_iterator = iter([mock_stream_chunk_1, mock_stream_chunk_2])
        mock_litellm_completion.return_value = mock_streaming_response_iterator

        adapter = LiteLLMAdapter(model="gpt-3.5-turbo", config=ModelConfig(api_key="stream_key"))
        # adapter.prepare() was a no-op and will be removed.

        stream = await adapter.generate("Hello Stream", stream=True)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert "".join(chunks) == "Hello LiteLLM!"
        mock_litellm_completion.assert_called_once()
        call_kwargs = mock_litellm_completion.call_args.kwargs
        assert call_kwargs['model'] == "gpt-3.5-turbo"
        assert call_kwargs['stream'] is True
        assert call_kwargs['api_key'] == "stream_key"

    @pytest.mark.asyncio
    @patch('litellm.completion')
    async def test_generate_with_tools_non_streaming(self, mock_litellm_completion, litellm_env_vars):
        mock_tool_response = MagicMock()
        mock_tool_message = MagicMock()
        mock_tool_message.content = "Tool use response"

        # Mocking LiteLLM's ToolCall object structure
        tool_call_obj = MagicMock()
        tool_call_obj.id = "call_abc"
        tool_call_obj.type = "function"
        tool_call_obj.function.name = "test_function"
        tool_call_obj.function.arguments = '{"arg": "val"}'

        mock_tool_message.tool_calls = [tool_call_obj]
        mock_tool_response.choices = [MagicMock(message=mock_tool_message)]
        mock_litellm_completion.return_value = mock_tool_response

        adapter = LiteLLMAdapter(model="gpt-4-tools", config=ModelConfig(api_key="tool_key"))
        # adapter.prepare() was a no-op and will be removed.

        tools = [{"type": "function", "function": {"name": "test_function", "description": "A test func"}}]
        response = await adapter.generate_with_tools("Call a tool", tools=tools)

        assert response['content'] == "Tool use response"
        assert len(response['tool_calls']) == 1
        assert response['tool_calls'][0]['id'] == "call_abc"
        assert response['tool_calls'][0]['function']['name'] == "test_function"

        mock_litellm_completion.assert_called_once()
        call_kwargs = mock_litellm_completion.call_args.kwargs
        assert call_kwargs['tools'] == tools
        assert call_kwargs['api_key'] == "tool_key"

    def test_litellm_cache_env_var(self, monkeypatch):
        """Test LITELLM_CACHE environment variable enables litellm.cache."""
        # Reset litellm.cache to default (or False) before test if possible
        # For this test, we assume litellm.cache is False by default or can be set
        if _HAS_LITELLM:
            litellm.cache = False # Ensure it's false initially
            monkeypatch.setenv("LITELLM_CACHE", "true")
            # Re-evaluate __init__ by creating an instance after setting env var
            _ = LiteLLMAdapter(model="test-cache-model")
            assert litellm.cache is True
            litellm.cache = False # Reset for other tests
        else:
            pytest.skip("litellm not installed")

    def test_import_error_if_litellm_not_installed(self):
        """Test that ImportError is raised if litellm is not installed."""
        global _HAS_LITELLM
        original_has_litellm = _HAS_LITELLM

        # Simulate litellm not being installed
        _HAS_LITELLM = False
        # Also need to ensure the import litellm line in the module itself fails
        # This is tricky as it's already imported. Patching _HAS_LITELLM in the module.
        with patch('src.codin.model.litellm_adapter._HAS_LITELLM', False):
            with pytest.raises(ImportError, match="LiteLLM is not installed"):
                LiteLLMAdapter(model="test")

        _HAS_LITELLM = original_has_litellm # Restore
        # Note: This test is a bit fragile as it depends on manipulating the import flag.
        # A more robust way might involve a separate test environment where litellm is truly not installed.
