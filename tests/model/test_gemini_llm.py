import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os

import json # For JSONDecodeError
import httpx # For mock_http_response spec
from src.codin.model.gemini_llm import GeminiLLM
from src.codin.model.config import ModelConfig
from src.codin.client import Client # For spec if needed
from src.codin.model.http_utils import ModelResponseParsingError, ContentExtractionError, StreamProcessingError # For new tests

@pytest.fixture
def mock_gemini_client():
    client_instance = MagicMock(spec=Client)
    client_instance.prepare = AsyncMock()
    client_instance.post = AsyncMock()
    client_instance.close = AsyncMock()
    return client_instance

@pytest.fixture(autouse=True)
def gemini_env_vars(monkeypatch):
    """Set up environment variables for Gemini tests."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("GOOGLE_API_BASE", raising=False)

    monkeypatch.setenv("GOOGLE_API_KEY", "env-gemini-key")
    # No GOOGLE_API_BASE set, will use class default unless config overrides

class TestGeminiLLM:

    # Synchronous tests for attribute setting part of __init__
    def test_init_config_attributes(self):
        config = ModelConfig(
            model_name="gemini-1.0-pro",
            api_key="cfg-gemini-key",
            base_url="https://custom.googleapis.com/gemini",
            timeout=70.0
        )
        # Note: Actual client setup is async, so we don't await here for these sync checks.
        # This test assumes that __init__ can set these before any awaitable calls.
        # If all setup including model name resolution is after first await, this test needs to be async.
        llm_sync_part = GeminiLLM(config=config, model=config.model_name)
        assert llm_sync_part.model == "gemini-1.0-pro"
        assert llm_sync_part.config.api_key == "cfg-gemini-key"
        assert llm_sync_part.config.base_url == "https://custom.googleapis.com/gemini"
        assert llm_sync_part.config.timeout == 70.0

    def test_init_model_arg_overrides_config(self):
        config = ModelConfig(model_name="gemini-flash")
        llm_sync_part = GeminiLLM(model="gemini-pro-direct", config=config)
        assert llm_sync_part.model == "gemini-pro-direct"
        assert llm_sync_part.config.model_name == "gemini-flash"

    @pytest.mark.asyncio
    async def test_init_client_with_env_vars(self, mock_gemini_client, gemini_env_vars): # Renamed
        """Test async __init__ using environment variables."""
        with patch("src.codin.model.gemini_llm.Client", return_value=mock_gemini_client) as mock_client_cls:
            llm = await GeminiLLM(model="gemini-1.5-pro-env", config=None) # Await

            assert llm._resolved_api_key == "env-gemini-key"
            mock_client_cls.assert_called_once()
            client_config_arg = mock_client_cls.call_args[0][0]

            assert client_config_arg.base_url == GeminiLLM.DEFAULT_BASE_URL
            assert client_config_arg.default_headers == {'Content-Type': 'application/json'}
            mock_gemini_client.prepare.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_client_with_full_config(self, mock_gemini_client, monkeypatch): # Renamed
        """Test async __init__ using a full ModelConfig, no env vars."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        config = ModelConfig(
            api_key="cfg-gemini-key-full",
            base_url="https://custom.gemini.ai/api",
            timeout=110.0,
            max_retries=4
        )
        with patch("src.codin.model.gemini_llm.Client", return_value=mock_gemini_client) as mock_client_cls:
            llm = await GeminiLLM(model="gemini-configured", config=config) # Await

            assert llm._resolved_api_key == "cfg-gemini-key-full"
            mock_client_cls.assert_called_once()
            client_config_arg = mock_client_cls.call_args[0][0]

            assert client_config_arg.base_url == "https://custom.gemini.ai/api"
            assert client_config_arg.timeout == 110.0
            assert client_config_arg.max_retries == 4
            mock_gemini_client.prepare.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_config_overrides_env(self, mock_gemini_client, gemini_env_vars): # Renamed
        """Test ModelConfig overrides environment variables during async __init__."""
        config = ModelConfig(api_key="cfg-override-key", base_url="https://cfg.gemini.com")

        with patch("src.codin.model.gemini_llm.Client", return_value=mock_gemini_client) as mock_client_cls:
            llm = await GeminiLLM(model="gemini-override", config=config) # Await

            assert llm._resolved_api_key == "cfg-override-key"
            client_config_arg = mock_client_cls.call_args[0][0]
            assert client_config_arg.base_url == "https://cfg.gemini.com"

    @pytest.mark.asyncio
    async def test_init_partial_config_with_env(self, mock_gemini_client, gemini_env_vars, monkeypatch): # Renamed
        """Test partial ModelConfig merges with environment variables during async __init__."""
        monkeypatch.setenv("GOOGLE_API_BASE", "https://env.gemini.com/v1beta")

        config = ModelConfig(api_key="partial-cfg-key")

        with patch("src.codin.model.gemini_llm.Client", return_value=mock_gemini_client) as mock_client_cls:
            llm = await GeminiLLM(model="gemini-partial-cfg", config=config) # Await

            assert llm._resolved_api_key == "partial-cfg-key"
            client_config_arg = mock_client_cls.call_args[0][0]
            assert client_config_arg.base_url == "https://env.gemini.com/v1beta"

    @pytest.mark.asyncio
    async def test_init_missing_api_key_error(self, monkeypatch): # Renamed
        """Test ValueError during async __init__ if API key is not in config or env."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        config = ModelConfig(api_key=None)
        with pytest.raises(ValueError, match="API key not found"):
            await GeminiLLM(model="gemini-no-key", config=config) # Await

    @pytest.mark.asyncio
    async def test_generate_simple(self, mock_gemini_client, gemini_env_vars):
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Test Gemini response"}]}}]
        }

        with patch("src.codin.model.gemini_llm.make_post_request", AsyncMock(return_value=mock_http_response)) as mock_post:
            llm = await GeminiLLM(model="gemini-test") # Await

            response = await llm.generate("Hello Gemini")
            assert response == "Test Gemini response"
            mock_post.assert_called_once()

            call_args = mock_post.call_args[0]
            assert llm._resolved_api_key in call_args[1]
            assert call_args[2]['contents'][0]['parts'][0]['text'] == "Hello Gemini"
            assert call_args[2]['contents'][0]['role'] == "user"

    @pytest.mark.asyncio
    async def test_complete_response_json_decode_error(self, mock_gemini_client, gemini_env_vars):
        """Test ModelResponseParsingError on JSON decode error in _complete_response."""
        with patch("src.codin.model.gemini_llm.Client", return_value=mock_gemini_client):
            llm = await GeminiLLM(model="gemini-test")

        mock_http_response = AsyncMock(spec=httpx.Response)
        mock_http_response.json = MagicMock(side_effect=json.JSONDecodeError("err", "doc", 0))
        mock_http_response.text = "invalid json string"

        with patch("src.codin.model.gemini_llm.make_post_request", AsyncMock(return_value=mock_http_response)):
            with pytest.raises((ModelResponseParsingError, StreamProcessingError)): # GeminiLLM's _complete_response raises StreamProcessingError for this
                # Need to pass endpoint and payload to _complete_response
                # The endpoint includes the API key, which is resolved in llm instance
                endpoint = f'/v1beta/models/{llm.model}:generateContent?key={llm._resolved_api_key}'
                await llm._complete_response(endpoint, {"some": "payload"})

    @pytest.mark.asyncio
    async def test_complete_response_content_extraction_none(self, mock_gemini_client, gemini_env_vars):
        """Test ContentExtractionError when extractor returns None in _complete_response."""
        with patch("src.codin.model.gemini_llm.Client", return_value=mock_gemini_client):
            llm = await GeminiLLM(model="gemini-test")

        mock_http_response = AsyncMock(spec=httpx.Response)
        # Valid JSON, but structure will make extractor return None (e.g., 'candidates' is empty)
        mock_http_response.json = MagicMock(return_value={"candidates": []})

        with patch("src.codin.model.gemini_llm.make_post_request", AsyncMock(return_value=mock_http_response)):
            # Gemini's _extract_content_from_response returns None if candidates is empty,
            # which then causes extract_content_from_json to raise ContentExtractionError.
            endpoint = f'/v1beta/models/{llm.model}:generateContent?key={llm._resolved_api_key}'
            with pytest.raises(ContentExtractionError, match="Extractor returned None"):
                await llm._complete_response(endpoint, {"some": "payload"})

    @pytest.mark.asyncio
    async def test_complete_response_parsing_error_missing_parts(self, mock_gemini_client, gemini_env_vars):
        """Test ModelResponseParsingError for missing 'parts' in _complete_response."""
        with patch("src.codin.model.gemini_llm.Client", return_value=mock_gemini_client):
            llm = await GeminiLLM(model="gemini-test")

        mock_http_response = AsyncMock(spec=httpx.Response)
        # Valid JSON, but structure will make extractor cause ModelResponseParsingError (e.g. parts is missing)
        mock_http_response.json = MagicMock(return_value={"candidates": [{"content": {"no_parts_here": True}}]})

        with patch("src.codin.model.gemini_llm.make_post_request", AsyncMock(return_value=mock_http_response)):
            endpoint = f'/v1beta/models/{llm.model}:generateContent?key={llm._resolved_api_key}'
            # The error "Extractor returned None" is raised because the helper returns None when parts are missing
            with pytest.raises(ContentExtractionError, match="Extractor returned None"):
                 await llm._complete_response(endpoint, {"some": "payload"})


    @pytest.mark.asyncio
    async def test_stream_response_delta_extraction_error(self, mock_gemini_client, gemini_env_vars):
        """Test StreamProcessingError if delta_extractor fails in _stream_response."""
        with patch("src.codin.model.gemini_llm.Client", return_value=mock_gemini_client):
            llm = await GeminiLLM(model="gemini-test")

        mock_http_stream_response = AsyncMock(spec=httpx.Response)
        async def faulty_aiter_lines():
            yield "data: {\"candidates\": [{\"content\": {\"parts\": [{\"text_typo\": \"some text\"}]}}]}" # text_typo will cause TypeError in extractor
        mock_http_stream_response.aiter_lines = faulty_aiter_lines
        mock_http_stream_response.aclose = AsyncMock()

        with patch("src.codin.model.gemini_llm.make_post_request", AsyncMock(return_value=mock_http_stream_response)):
            # Patch the specific helper method to simulate an internal error
            with patch.object(llm, '_extract_delta_from_stream_chunk', side_effect=TypeError("Simulated type error")):
                with pytest.raises(StreamProcessingError, match="Delta extraction failed for chunk"):
                    endpoint = f'/v1beta/models/{llm.model}:generateContent?key={llm._resolved_api_key}&alt=sse'
                    stream = await llm._stream_response(endpoint, {"some": "payload"})
                    async for _ in stream: pass
