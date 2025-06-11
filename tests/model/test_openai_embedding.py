import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os

import json # For JSONDecodeError
import httpx # For mock_http_response spec
from src.codin.model.openai_embedding import OpenAIEmbedding
from src.codin.model.config import ModelConfig
from src.codin.client import Client # For spec if needed
from src.codin.model.http_utils import ModelResponseParsingError, ContentExtractionError # For new tests

@pytest.fixture
def mock_embedding_client():
    client_instance = MagicMock(spec=Client)
    client_instance.prepare = AsyncMock()
    client_instance.post = AsyncMock()
    client_instance.close = AsyncMock()
    return client_instance

@pytest.fixture(autouse=True)
def openai_embedding_env_vars(monkeypatch):
    """Set up environment variables for OpenAIEmbedding tests."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False) # Specific for embeddings
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    # LLM_API_KEY might also be checked by ModelConfig, ensure it's not interfering unless intended
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    monkeypatch.setenv("OPENAI_API_KEY", "env-embedding-key")
    # No OPENAI_API_BASE set by default, will use class default unless config overrides

class TestOpenAIEmbedding:

    # __init__ is now async, so original sync tests for __init__ are removed/merged.
    # Tests will now await the constructor and check the resulting state or client calls.

    @pytest.mark.asyncio
    async def test_init_client_with_env_vars(self, mock_embedding_client, openai_embedding_env_vars): # Renamed
        """Test async __init__ using environment variables."""
        with patch("src.codin.model.openai_embedding.Client", return_value=mock_embedding_client) as mock_client_cls:
            emb_model = await OpenAIEmbedding(model="text-embedding-ada-002", config=None) # Await

            assert emb_model.model == "text-embedding-ada-002" # Ensure model name is set

            mock_client_cls.assert_called_once()
            client_config_arg = mock_client_cls.call_args[0][0]

            assert client_config_arg.default_headers["Authorization"] == "Bearer env-embedding-key"
            assert client_config_arg.base_url == OpenAIEmbedding.DEFAULT_BASE_URL
            mock_embedding_client.prepare.assert_called_once() # Client.prepare() called in __init__

    @pytest.mark.asyncio
    async def test_init_client_with_full_config(self, mock_embedding_client, monkeypatch): # Renamed
        """Test async __init__ using a full ModelConfig, no env vars."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config = ModelConfig(
            api_key="cfg-embed-key-full",
            base_url="https://custom.openai.com/embed_api",
            timeout=75.0,
            max_retries=2
        )
        with patch("src.codin.model.openai_embedding.Client", return_value=mock_embedding_client) as mock_client_cls:
            emb_model = await OpenAIEmbedding(model="text-embedding-ada-002", config=config) # Await

            mock_client_cls.assert_called_once()
            client_config_arg = mock_client_cls.call_args[0][0]

            assert client_config_arg.default_headers["Authorization"] == "Bearer cfg-embed-key-full"
            assert client_config_arg.base_url == "https://custom.openai.com/embed_api"
            assert client_config_arg.timeout == 75.0
            assert client_config_arg.max_retries == 2
            mock_embedding_client.prepare.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_config_overrides_env(self, mock_embedding_client, openai_embedding_env_vars): # Renamed
        """Test ModelConfig overrides environment variables during async __init__."""
        config = ModelConfig(api_key="cfg-override-key", base_url="https://cfg.openai.com/embed")

        with patch("src.codin.model.openai_embedding.Client", return_value=mock_embedding_client) as mock_client_cls:
            emb_model = await OpenAIEmbedding(model="text-embedding-ada-002", config=config) # Await

            client_config_arg = mock_client_cls.call_args[0][0]
            assert client_config_arg.default_headers["Authorization"] == "Bearer cfg-override-key"
            assert client_config_arg.base_url == "https://cfg.openai.com/embed"

    @pytest.mark.asyncio
    async def test_init_partial_config_with_env(self, mock_embedding_client, openai_embedding_env_vars, monkeypatch): # Renamed
        """Test partial ModelConfig merges with environment variables during async __init__."""
        monkeypatch.setenv("OPENAI_API_BASE", "https://env.openai.com/embed_base")

        config = ModelConfig(api_key="partial-cfg-key")

        with patch("src.codin.model.openai_embedding.Client", return_value=mock_embedding_client) as mock_client_cls:
            emb_model = await OpenAIEmbedding(model="text-embedding-ada-002", config=config) # Await

            client_config_arg = mock_client_cls.call_args[0][0]
            assert client_config_arg.default_headers["Authorization"] == "Bearer partial-cfg-key"
            assert client_config_arg.base_url == "https://env.openai.com/embed_base"

    @pytest.mark.asyncio
    async def test_init_missing_api_key_error(self, monkeypatch): # Renamed
        """Test ValueError during async __init__ if API key is not in config or env."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        config = ModelConfig(api_key=None)
        with pytest.raises(ValueError, match="API key not found. Set in ModelConfig or OPENAI_API_KEY"):
            await OpenAIEmbedding(model="text-embedding-ada-002", config=config) # Await

    @pytest.mark.asyncio
    async def test_embed_simple(self, mock_embedding_client, openai_embedding_env_vars):
        """Test basic embed() call."""
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]}
            ],
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }
        with patch("src.codin.model.openai_embedding.make_post_request", AsyncMock(return_value=mock_http_response)) as mock_make_post:
            # Instantiation now includes client setup (which uses mock_embedding_client via the class patch if not for make_post_request)
            # For this test, we are directly mocking make_post_request, so the internal _client of emb_model isn't used by the patched method.
            # If we were testing the _client interaction, we'd let make_post_request be real and mock _client.post.
            # This test as written focuses on the embed method's interaction with make_post_request.

            # We need to patch the Client used by __init__ if we want to avoid real client creation logic.
            with patch("src.codin.model.openai_embedding.Client", return_value=mock_embedding_client):
                 emb_model = await OpenAIEmbedding(model="text-embedding-ada-002") # Await

            texts_to_embed = ["hello", "world"]
            embeddings = await emb_model.embed(texts_to_embed)

            assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_make_post.assert_called_once()

            # Check that emb_model._client was passed to make_post_request
            # call_args_list[0][0] is the first positional arg tuple
            # call_args_list[0][0][0] is the first argument, which should be the client instance
            assert mock_make_post.call_args_list[0][0][0] == emb_model._client

            payload = mock_make_post.call_args_list[0][0][2] # Third positional arg is payload
            assert payload['model'] == "text-embedding-ada-002"
            assert payload['input'] == texts_to_embed

    @pytest.mark.asyncio
    async def test_embed_json_decode_error(self, mock_embedding_client, openai_embedding_env_vars):
        """Test ModelResponseParsingError on JSON decode error in embed()."""
        with patch("src.codin.model.openai_embedding.Client", return_value=mock_embedding_client):
            emb_model = await OpenAIEmbedding(model="text-embedding-ada-002")

        mock_http_response = AsyncMock(spec=httpx.Response)
        mock_http_response.json = MagicMock(side_effect=json.JSONDecodeError("decode error", "doc", 0))
        mock_http_response.text = "invalid json"

        with patch("src.codin.model.openai_embedding.make_post_request", AsyncMock(return_value=mock_http_response)):
            # The embed method wraps this in ContentExtractionError currently.
            # Let's refine this to ModelResponseParsingError if that's more appropriate.
            # Current embed() catches generic Exception and wraps in ContentExtractionError after json() call.
            # The http_utils.ModelResponseParsingError is for parsing structure, not json decode itself.
            # The specific error from embed() for json decode is ContentExtractionError.
            with pytest.raises(ContentExtractionError, match="Failed to get embeddings: Failed to decode JSON response"):
                await emb_model.embed(["test text"])

    @pytest.mark.asyncio
    async def test_embed_missing_data_key(self, mock_embedding_client, openai_embedding_env_vars):
        """Test ModelResponseParsingError for missing 'data' key in embed()."""
        with patch("src.codin.model.openai_embedding.Client", return_value=mock_embedding_client):
            emb_model = await OpenAIEmbedding(model="text-embedding-ada-002")

        mock_http_response = AsyncMock(spec=httpx.Response)
        mock_http_response.json = MagicMock(return_value={"no_data_field": True}) # Missing 'data'

        with patch("src.codin.model.openai_embedding.make_post_request", AsyncMock(return_value=mock_http_response)):
            with pytest.raises(ModelResponseParsingError, match="Invalid response format from model text-embedding-ada-002: 'data' field missing or not a list."):
                await emb_model.embed(["test text"])

    @pytest.mark.asyncio
    async def test_embed_data_not_a_list(self, mock_embedding_client, openai_embedding_env_vars):
        """Test ModelResponseParsingError if 'data' field is not a list in embed()."""
        with patch("src.codin.model.openai_embedding.Client", return_value=mock_embedding_client):
            emb_model = await OpenAIEmbedding(model="text-embedding-ada-002")

        mock_http_response = AsyncMock(spec=httpx.Response)
        mock_http_response.json = MagicMock(return_value={"data": "not_a_list"}) # 'data' is not a list

        with patch("src.codin.model.openai_embedding.make_post_request", AsyncMock(return_value=mock_http_response)):
            with pytest.raises(ModelResponseParsingError, match="'data' field missing or not a list"):
                await emb_model.embed(["test text"])

    @pytest.mark.asyncio
    async def test_embed_item_missing_embedding_key(self, mock_embedding_client, openai_embedding_env_vars, caplog):
        """Test warning and partial results if an item in 'data' misses 'embedding' key."""
        with patch("src.codin.model.openai_embedding.Client", return_value=mock_embedding_client):
            emb_model = await OpenAIEmbedding(model="text-embedding-ada-002")

        mock_http_response = AsyncMock(spec=httpx.Response)
        mock_http_response.json = MagicMock(return_value={
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "no_embedding_key": [0.3, 0.4]}, # Item missing 'embedding'
                {"index": 2, "embedding": [0.5, 0.6]}
            ]
        })

        with patch("src.codin.model.openai_embedding.make_post_request", AsyncMock(return_value=mock_http_response)):
            results = await emb_model.embed(["text1", "text2", "text3"])
            assert len(results) == 2 # Should skip the item with missing key
            assert results[0] == [0.1, 0.2]
            assert results[1] == [0.5, 0.6]
            assert "Skipping item in embedding response" in caplog.text
            assert "missing 'embedding' key" in caplog.text
