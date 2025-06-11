import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os

from src.codin.model.vllm_client import VLLMClient
from src.codin.model.config import ModelConfig
from src.codin.client import Client # For spec

@pytest.fixture
def mock_vllm_internal_client(): # Mocks the codin.client.Client
    client_instance = MagicMock(spec=Client)
    client_instance.post = AsyncMock() # Used by OpenAICompatibleBaseLLM's helpers
    client_instance.close = AsyncMock()
    return client_instance

@pytest.fixture(autouse=True)
def vllm_env_vars(monkeypatch):
    """Clear and set up environment variables for VLLMClient tests."""
    var_list = [
        "VLLM_MODEL", "VLLM_BASE_URL", "VLLM_API_KEY",
        "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY",
        "OPENAI_API_KEY" # Used as ultimate fallback by OpenAICompatibleBaseLLM
    ]
    for var in var_list:
        monkeypatch.delenv(var, raising=False)

    # Set some specific VLLM vars for some tests
    monkeypatch.setenv("VLLM_MODEL", "env-vllm-model-path")
    monkeypatch.setenv("VLLM_BASE_URL", "http://env.vllm.server:7000/v1")
    # VLLM_API_KEY is often not set, OpenAICompatibleBaseLLM handles None/empty api_key

class TestVLLMClient:

    @pytest.mark.asyncio
    async def test_init_from_config(self, mock_vllm_internal_client):
        """Test VLLMClient initialization primarily from ModelConfig."""
        with patch("src.codin.model.openai_compatible_llm.Client", return_value=mock_vllm_internal_client) as mock_client_constructor:
            cfg = ModelConfig(
                model_name="config/vllm-model",
                base_url="http://config.vllm.server:8080/v1",
                api_key="cfg_vllm_key_or_empty", # vLLM might not need a key
                provider="vllm_custom_provider_in_cfg" # provider in ModelConfig
            )
            llm = await VLLMClient(config=cfg)

            assert llm.model == "config/vllm-model"
            assert llm.config is cfg # Should use the provided config object

            mock_client_constructor.assert_called_once()
            client_config_arg: ClientConfig = mock_client_constructor.call_args[0][0]

            assert client_config_arg.base_url == "http://config.vllm.server:8080/v1"
            if cfg.api_key: # Only assert Authorization if api_key was non-empty
                assert client_config_arg.default_headers["Authorization"] == f"Bearer {cfg.api_key}"
            else:
                assert "Authorization" not in client_config_arg.default_headers
            # VLLMClient itself doesn't add more specific headers beyond what OpenAICompatibleBaseLLM does

    @pytest.mark.asyncio
    async def test_init_from_env_vars(self, mock_vllm_internal_client, vllm_env_vars):
        """Test VLLMClient initialization using its specific environment variables."""
        with patch("src.codin.model.openai_compatible_llm.Client", return_value=mock_vllm_internal_client) as mock_client_constructor:
            # Pass model via direct arg, other from env
            llm = await VLLMClient(model="direct-model-path")

            assert llm.model == "direct-model-path" # Direct model arg takes precedence
            # Config object is created by default if not passed
            assert llm.config.base_url == "http://env.vllm.server:7000/v1" # From VLLM_BASE_URL
            # VLLM_API_KEY is not set in fixture, so api_key in config should be "" (empty string)
            assert llm.config.api_key == ""

            mock_client_constructor.assert_called_once()
            client_config_arg: ClientConfig = mock_client_constructor.call_args[0][0]
            assert client_config_arg.base_url == "http://env.vllm.server:7000/v1"
            assert "Authorization" not in client_config_arg.default_headers # Due to empty api_key

    @pytest.mark.asyncio
    async def test_init_fallback_to_generic_env_vars(self, mock_vllm_internal_client, monkeypatch):
        """Test VLLMClient falls back to LLM_MODEL, LLM_BASE_URL if VLLM_ ones are not set."""
        monkeypatch.setenv("LLM_MODEL", "generic-llm-model-for-vllm")
        monkeypatch.setenv("LLM_BASE_URL", "http://generic.llm.server:9000/v1")
        monkeypatch.setenv("LLM_API_KEY", "generic_key_for_vllm")
        # Ensure VLLM_ specific env vars are not set
        monkeypatch.delenv("VLLM_MODEL", raising=False)
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        monkeypatch.delenv("VLLM_API_KEY", raising=False)


        with patch("src.codin.model.openai_compatible_llm.Client", return_value=mock_vllm_internal_client) as mock_client_constructor:
            llm = await VLLMClient()

            assert llm.model == "generic-llm-model-for-vllm"
            assert llm.config.api_key == "generic_key_for_vllm"
            assert llm.config.base_url == "http://generic.llm.server:9000/v1"

            client_config_arg: ClientConfig = mock_client_constructor.call_args[0][0]
            assert client_config_arg.default_headers["Authorization"] == "Bearer generic_key_for_vllm"

    @pytest.mark.asyncio
    async def test_init_uses_vllm_default_base_url(self, mock_vllm_internal_client, monkeypatch):
        """Test VLLMClient uses its own DEFAULT_BASE_URL if no other URL is specified."""
        monkeypatch.setenv("VLLM_MODEL", "some-model") # Model must be provided
        # Ensure no base URL env vars are set
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        with patch("src.codin.model.openai_compatible_llm.Client", return_value=mock_vllm_internal_client) as mock_client_constructor:
            llm = await VLLMClient(config=ModelConfig(api_key="")) # Provide empty API key

            assert llm.config.base_url == VLLMClient.DEFAULT_BASE_URL # http://localhost:8000/v1
            client_config_arg: ClientConfig = mock_client_constructor.call_args[0][0]
            assert client_config_arg.base_url == VLLMClient.DEFAULT_BASE_URL


    @pytest.mark.asyncio
    async def test_init_model_name_mandatory(self, monkeypatch):
        """Test VLLMClient __init__ raises ValueError if model name cannot be resolved."""
        # Clear all relevant model name env vars
        monkeypatch.delenv("VLLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        # Also ensure DEFAULT_MODEL of ConcreteVLLM (if used) or VLLMClient.DEFAULT_MODEL is None
        # VLLMClient.DEFAULT_MODEL is currently None, which is what we want to test here.

        with pytest.raises(ValueError, match="Model name/path for VLLM must be specified"):
            await VLLMClient(config=ModelConfig(api_key="")) # Model name not in config either

    @pytest.mark.asyncio
    @patch("src.codin.model.openai_compatible_llm.make_post_request")
    async def test_generate_calls_super(self, mock_make_post, vllm_env_vars, mock_vllm_internal_client):
        """Test VLLMClient.generate calls the base class method correctly."""
        mock_response_json = {"choices": [{"message": {"content": "vLLM response"}}]}
        mock_http_response = AsyncMock(spec=httpx.Response)
        mock_http_response.json = MagicMock(return_value=mock_response_json)
        mock_make_post.return_value = mock_http_response

        # Patch the Client constructor within the scope of OpenAICompatibleBaseLLM
        with patch("src.codin.model.openai_compatible_llm.Client", return_value=mock_vllm_internal_client):
            llm = await VLLMClient(model="test-vllm-model/Llama-2-7b-hf")

        prompt = "Hello vLLM"
        result = await llm.generate(prompt)

        assert result == "vLLM response"
        mock_make_post.assert_called_once()
        call_kwargs = mock_make_post.call_args.kwargs
        assert call_kwargs['json_payload']['model'] == "test-vllm-model/Llama-2-7b-hf"
        assert call_kwargs['json_payload']['messages'] == [{"role": "user", "content": prompt}]

    def test_supported_models_is_generic(self):
        assert VLLMClient.supported_models() == [r'.*']
