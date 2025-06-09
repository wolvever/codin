import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os

import json # For JSONDecodeError
import httpx # For mock_http_response spec
from src.codin.model.anthropic_llm import AnthropicLLM
from src.codin.model.config import ModelConfig
from src.codin.client import Client # For spec if needed
from src.codin.model.http_utils import ModelResponseParsingError, ContentExtractionError, StreamProcessingError # For new tests

@pytest.fixture
def mock_anthropic_client():
    client_instance = MagicMock(spec=Client)
    client_instance.prepare = AsyncMock()
    client_instance.post = AsyncMock()
    client_instance.close = AsyncMock()
    return client_instance

@pytest.fixture(autouse=True)
def anthropic_env_vars(monkeypatch):
    """Set up environment variables for Anthropic tests."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_VERSION", raising=False)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic-key")
    # No default ANTHROPIC_API_BASE, will use class default if not in config
    # No default ANTHROPIC_API_VERSION, will use class default

class TestAnthropicLLM:

    # These __init__ tests remain synchronous as __init__ itself is not async
    # The async nature comes when the client setup (previously in prepare) is done.
    # Now, the entire object instantiation `await AnthropicLLM(...)` is async.
    # So, these tests are for the synchronous part of __init__ (attribute setting).
    def test_init_with_model_name_only(self):
        # This test is less relevant if __init__ becomes async and does all setup.
        # It would need to be async and await the constructor.
        # For now, assuming these are testing pre-async-init logic.
        # If __init__ becomes async, these tests need to be async and await.
        # Let's assume for now these are testing the state before client setup.
        config = ModelConfig()
        llm = AnthropicLLM(model="claude-2.1", config=config) # This line would change if __init__ is async
        assert llm.model == "claude-2.1"
        assert llm.config.model_name is None
        assert llm.config.api_key is None

    def test_init_with_config(self):
        config = ModelConfig(
            model_name="claude-instant-1",
            api_key="cfg-anthropic-key",
            base_url="https://api.anthropic.cfg/v1",
            api_version="2023-07-01",
            timeout=90.0
        )
        llm = AnthropicLLM(config=config, model=config.model_name) # Pass model if relying on config.model_name
        assert llm.model == "claude-instant-1"
        assert llm.config.api_key == "cfg-anthropic-key"
        assert llm.config.base_url == "https://api.anthropic.cfg/v1"
        assert llm.config.api_version == "2023-07-01"
        assert llm.config.timeout == 90.0

    def test_init_model_arg_overrides_config_model_name(self):
        config = ModelConfig(model_name="claude-opus")
        llm = AnthropicLLM(model="claude-sonnet", config=config)
        assert llm.model == "claude-sonnet"
        assert llm.config.model_name == "claude-opus"

    @pytest.mark.asyncio
    async def test_init_client_with_env_vars(self, mock_anthropic_client, anthropic_env_vars): # Renamed
        """Test async __init__ using environment variables."""
        with patch("src.codin.model.anthropic_llm.Client", return_value=mock_anthropic_client) as mock_client_cls:
            llm = await AnthropicLLM(model="claude-2.0", config=None) # Await instantiation

            mock_client_cls.assert_called_once()
            client_config_arg = mock_client_cls.call_args[0][0]

            assert client_config_arg.default_headers["x-api-key"] == "env-anthropic-key"
            assert client_config_arg.base_url == AnthropicLLM.DEFAULT_BASE_URL
            assert client_config_arg.default_headers["anthropic-version"] == AnthropicLLM.DEFAULT_API_VERSION
            mock_anthropic_client.prepare.assert_called_once() # Client.prepare() is called in __init__

    @pytest.mark.asyncio
    async def test_init_client_with_full_config(self, mock_anthropic_client, monkeypatch): # Renamed
        """Test async __init__ using a full ModelConfig, no env vars."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        config = ModelConfig(
            api_key="cfg-anthropic-key-full",
            base_url="https://custom.anthropic.com/api",
            api_version="2023-08-08",
            timeout=120.0,
            max_retries=3
        )
        with patch("src.codin.model.anthropic_llm.Client", return_value=mock_anthropic_client) as mock_client_cls:
            llm = await AnthropicLLM(model="claude-configured", config=config) # Await

            mock_client_cls.assert_called_once()
            client_config_arg = mock_client_cls.call_args[0][0]

            assert client_config_arg.default_headers["x-api-key"] == "cfg-anthropic-key-full"
            assert client_config_arg.base_url == "https://custom.anthropic.com/api"
            assert client_config_arg.default_headers["anthropic-version"] == "2023-08-08"
            assert client_config_arg.timeout == 120.0
            assert client_config_arg.max_retries == 3
            mock_anthropic_client.prepare.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_config_overrides_env(self, mock_anthropic_client, anthropic_env_vars): # Renamed
        """Test ModelConfig overrides environment variables during async __init__."""
        config = ModelConfig(api_key="cfg-override-key", api_version="v-cfg")

        with patch("src.codin.model.anthropic_llm.Client", return_value=mock_anthropic_client) as mock_client_cls:
            llm = await AnthropicLLM(model="claude-override", config=config) # Await

            client_config_arg = mock_client_cls.call_args[0][0]
            assert client_config_arg.default_headers["x-api-key"] == "cfg-override-key"
            assert client_config_arg.base_url == AnthropicLLM.DEFAULT_BASE_URL
            assert client_config_arg.default_headers["anthropic-version"] == "v-cfg"

    @pytest.mark.asyncio
    async def test_init_partial_config_with_env(self, mock_anthropic_client, anthropic_env_vars, monkeypatch): # Renamed
        """Test partial ModelConfig merges with environment variables during async __init__."""
        monkeypatch.setenv("ANTHROPIC_API_BASE", "https://env.anthropic.com")
        monkeypatch.setenv("ANTHROPIC_API_VERSION", "v-env")

        config = ModelConfig(api_key="partial-cfg-key")

        with patch("src.codin.model.anthropic_llm.Client", return_value=mock_anthropic_client) as mock_client_cls:
            llm = await AnthropicLLM(model="claude-partial-cfg", config=config) # Await

            client_config_arg = mock_client_cls.call_args[0][0]
            assert client_config_arg.default_headers["x-api-key"] == "partial-cfg-key"
            assert client_config_arg.base_url == "https://env.anthropic.com"
            assert client_config_arg.default_headers["anthropic-version"] == "v-env"

    @pytest.mark.asyncio
    async def test_init_missing_api_key_error(self, monkeypatch): # Renamed
        """Test ValueError during async __init__ if API key is not in config or env."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        config = ModelConfig(api_key=None)

        with pytest.raises(ValueError, match="API key not found"):
            await AnthropicLLM(model="claude-no-key", config=config) # Await

    @pytest.mark.asyncio
    async def test_generate_simple(self, mock_anthropic_client, anthropic_env_vars):
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {"content": [{"type": "text", "text": "Test response"}]}

        with patch("src.codin.model.anthropic_llm.make_post_request", AsyncMock(return_value=mock_http_response)) as mock_post:
            # Instantiation now includes client setup
            llm = await AnthropicLLM(model="claude-test") # Await

            response = await llm.generate("Hello")
            assert response == "Test response"
            mock_post.assert_called_once()
            call_args = mock_post.call_args[0]
            assert call_args[2]['model'] == "claude-test"
            assert call_args[2]['messages'] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_complete_response_json_decode_error(self, mock_anthropic_client, anthropic_env_vars):
        """Test ModelResponseParsingError on JSON decode error in _complete_response."""
        with patch("src.codin.model.anthropic_llm.Client", return_value=mock_anthropic_client):
            llm = await AnthropicLLM(model="claude-test")

        mock_http_response = AsyncMock(spec=httpx.Response)
        mock_http_response.json = MagicMock(side_effect=json.JSONDecodeError("err", "doc", 0))
        mock_http_response.text = "invalid json string"

        with patch("src.codin.model.anthropic_llm.make_post_request", AsyncMock(return_value=mock_http_response)):
            with pytest.raises((ModelResponseParsingError, StreamProcessingError)): # AnthropicLLM's _complete_response raises StreamProcessingError
                await llm._complete_response({"some": "payload"})

    @pytest.mark.asyncio
    async def test_complete_response_content_extraction_none(self, mock_anthropic_client, anthropic_env_vars):
        """Test ContentExtractionError when extractor returns None in _complete_response."""
        with patch("src.codin.model.anthropic_llm.Client", return_value=mock_anthropic_client):
            llm = await AnthropicLLM(model="claude-test")

        mock_http_response = AsyncMock(spec=httpx.Response)
        # Valid JSON, but structure will make extractor return None (e.g., 'content' is empty or not list)
        mock_http_response.json = MagicMock(return_value={"content": []})

        with patch("src.codin.model.anthropic_llm.make_post_request", AsyncMock(return_value=mock_http_response)):
            # The extractor for Anthropic joins parts. If content is [], it returns "",
            # extract_content_from_json by default does not raise error for empty string.
            # To test ContentExtractionError for "Extractor returned None", the helper itself must return None.
            # Anthropic's _extract_content_from_response returns "" for no text, not None.
            # So, we test the case where 'content' key is missing, which makes the helper return None.
            mock_http_response.json = MagicMock(return_value={"other_key": "no_content_blocks"})
            with pytest.raises(ContentExtractionError, match="Extractor returned None"):
                await llm._complete_response({"some": "payload"})

    @pytest.mark.asyncio
    async def test_stream_response_delta_extraction_error(self, mock_anthropic_client, anthropic_env_vars):
        """Test StreamProcessingError if delta_extractor fails in _stream_response."""
        with patch("src.codin.model.anthropic_llm.Client", return_value=mock_anthropic_client):
            llm = await AnthropicLLM(model="claude-test")

        mock_http_stream_response = AsyncMock(spec=httpx.Response)
        async def faulty_aiter_lines():
            # This data_chunk, when passed to _extract_delta_from_stream_chunk, might cause an error if not handled
            # For example, if 'delta' was expected but missing, and not handled by .get()
            yield "data: {\"type\": \"content_block_delta\", \"delta\": \"not_a_dict_as_expected\"}"
        mock_http_stream_response.aiter_lines = faulty_aiter_lines
        mock_http_stream_response.aclose = AsyncMock()

        with patch("src.codin.model.anthropic_llm.make_post_request", AsyncMock(return_value=mock_http_stream_response)):
            # Patch the specific helper method to simulate an internal error during its execution
            with patch.object(llm, '_extract_delta_from_stream_chunk', side_effect=TypeError("Simulated type error")):
                with pytest.raises(StreamProcessingError, match="Delta extraction failed for chunk"):
                    stream = await llm._stream_response({"some": "payload"})
                    async for _ in stream: pass
