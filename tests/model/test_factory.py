import pytest
import os
from unittest.mock import patch, MagicMock

from src.codin.model.factory import LLMFactory, create_llm_from_env
from src.codin.model.config import ModelConfig
from src.codin.model.openai_llm import OpenAILLM
from src.codin.model.anthropic_llm import AnthropicLLM
# Add other LLM types if they are part of PROVIDER_CLASSES and need specific testing

# Mock the actual LLM classes to prevent real API calls and to check constructor args
@pytest.fixture(autouse=True)
def mock_llm_classes():
    with patch.dict(LLMFactory.PROVIDER_CLASSES, {
        'openai': MagicMock(spec=OpenAILLM),
        'anthropic': MagicMock(spec=AnthropicLLM),
        # 'gemini': MagicMock(spec=GeminiLLM), # Assuming GeminiLLM exists
        # 'litellm': MagicMock(spec=LiteLLMAdapter) # Assuming LiteLLMAdapter exists
    }) as mock_providers:
        # Configure the .return_value for each mocked class constructor
        for provider_name, mock_class in mock_providers.items():
            llm_instance_mock = MagicMock()
            # Mock methods expected to be called on the instance, e.g., prepare, generate
            llm_instance_mock.prepare = MagicMock()
            mock_class.return_value = llm_instance_mock
        yield mock_providers

class TestLLMFactory:

    @pytest.fixture(autouse=True)
    def clear_env_vars(self, monkeypatch):
        """Clear relevant environment variables before each test."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_create_llm_with_direct_args(self, mock_llm_classes): # Made async
        """Test creating LLM with direct arguments."""
        factory_instance = await LLMFactory.create_llm( # Added await
            model="gpt-direct",
            provider="openai",
            api_key="direct_key",
            base_url="https://direct.url/v1"
        )

        mock_openai_class = mock_llm_classes['openai']
        mock_openai_class.assert_called_once()

        args, kwargs = mock_openai_class.call_args
        assert isinstance(kwargs['config'], ModelConfig)
        assert kwargs['config'].api_key == "direct_key"
        assert kwargs['config'].base_url == "https://direct.url/v1"
        assert kwargs['config'].provider == "openai"
        assert kwargs['model'] == "gpt-direct"

    @pytest.mark.asyncio
    async def test_create_llm_with_model_config_object(self, mock_llm_classes): # Made async
        """Test creating LLM with a ModelConfig object."""
        model_config = ModelConfig(
            api_key="cfg_key",
            base_url="https://cfg.url/v1",
            provider="openai",
            timeout=90.0
        )
        # Pass model name directly as it's a required arg for create_llm if not in env
        await LLMFactory.create_llm(model="actual_model_name", config=model_config) # Added await

        mock_openai_class = mock_llm_classes['openai']
        mock_openai_class.assert_called_once()

        args, kwargs = mock_openai_class.call_args
        assert isinstance(kwargs['config'], ModelConfig)
        assert kwargs['config'].api_key == "cfg_key"
        assert kwargs['config'].base_url == "https://cfg.url/v1"
        assert kwargs['config'].provider == "openai"
        assert kwargs['config'].timeout == 90.0
        assert kwargs['model'] == "actual_model_name"

    @pytest.mark.asyncio
    async def test_create_llm_with_env_vars(self, monkeypatch, mock_llm_classes): # Made async
        """Test creating LLM using environment variables."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "env_model")
        monkeypatch.setenv("LLM_API_KEY", "env_key")
        monkeypatch.setenv("LLM_BASE_URL", "https://env.url/v1")

        await LLMFactory.create_llm() # Added await

        mock_openai_class = mock_llm_classes['openai']
        mock_openai_class.assert_called_once()

        args, kwargs = mock_openai_class.call_args
        assert isinstance(kwargs['config'], ModelConfig)
        assert kwargs['config'].api_key == "env_key"
        assert kwargs['config'].base_url == "https://env.url/v1"
        assert kwargs['config'].provider == "openai"
        assert kwargs['model'] == "env_model"

    @pytest.mark.asyncio
    async def test_create_llm_priority_direct_args_vs_config(self, mock_llm_classes): # Made async
        """Test direct arguments override ModelConfig."""
        model_config = ModelConfig(api_key="cfg_key", base_url="https://cfg.url/v1", provider="anthropic")
        await LLMFactory.create_llm( # Added await
            model="direct_model",
            provider="openai",
            api_key="direct_key",
            config=model_config
        )

        mock_openai_class = mock_llm_classes['openai']
        mock_openai_class.assert_called_once()

        args, kwargs = mock_openai_class.call_args
        assert kwargs['config'].api_key == "direct_key"
        assert kwargs['config'].provider == "openai"
        assert kwargs['model'] == "direct_model"

    @pytest.mark.asyncio
    async def test_create_llm_priority_config_vs_env(self, monkeypatch, mock_llm_classes): # Made async
        """Test ModelConfig overrides environment variables."""
        monkeypatch.setenv("LLM_API_KEY", "env_key")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")

        model_config = ModelConfig(api_key="cfg_key", provider="openai")
        await LLMFactory.create_llm(model="some_model", config=model_config) # Added await

        mock_openai_class = mock_llm_classes['openai']
        mock_openai_class.assert_called_once()

        args, kwargs = mock_openai_class.call_args
        assert kwargs['config'].api_key == "cfg_key"
        assert kwargs['config'].provider == "openai"

    @pytest.mark.asyncio
    async def test_create_llm_auto_detect_provider(self, mock_llm_classes): # Made async
        """Test auto-detection of provider based on model name."""
        await LLMFactory.create_llm(model="claude-2") # Added await
        mock_llm_classes['anthropic'].assert_called_once()

        mock_llm_classes['anthropic'].reset_mock()

        await LLMFactory.create_llm(model="gpt-4") # Added await
        mock_llm_classes['openai'].assert_called_once()

    @pytest.mark.asyncio
    async def test_create_llm_missing_model_name(self): # Made async
        """Test error when model name is not provided anywhere."""
        with pytest.raises(ValueError, match="Model name must be provided"):
            await LLMFactory.create_llm() # Added await

    @pytest.mark.asyncio
    async def test_create_llm_unsupported_provider(self): # Made async
        """Test error for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider 'bogus_provider'"):
            await LLMFactory.create_llm(model="any_model", provider="bogus_provider") # Added await

    @pytest.mark.asyncio
    async def test_create_llm_from_env_uses_factory(self, monkeypatch): # Made async
        """Test create_llm_from_env correctly calls LLMFactory.create_llm."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "env_model_for_helper")

        # Since create_llm is now async, its mock needs to be an AsyncMock if we want to assert await
        # However, here we are wrapping the actual async method, so it's fine.
        with patch.object(LLMFactory, 'create_llm', wraps=LLMFactory.create_llm) as mock_factory_create:
            mock_factory_create.return_value = AsyncMock() # Ensure the wrapped call can be awaited
            await create_llm_from_env() # Added await
            mock_factory_create.assert_called_once()
            args, kwargs = mock_factory_create.call_args
            assert 'config' in kwargs

    @pytest.mark.asyncio
    async def test_create_llm_from_env_with_config_passed(self, monkeypatch, mock_llm_classes): # Made async
        """Test create_llm_from_env with a base config."""
        monkeypatch.setenv("LLM_API_KEY", "env_key_for_helper_override")

        base_config = ModelConfig(
            provider="openai",
            model_name="base_cfg_model",
            api_key="base_cfg_key"
        )
        base_config_no_key = ModelConfig(provider="openai", model_name="model_no_key")

        await create_llm_from_env(config=base_config_no_key) # Added await
        mock_openai_class = mock_llm_classes['openai']

        args, kwargs = mock_openai_class.call_args
        assert kwargs['config'].api_key == "env_key_for_helper_override"
        assert kwargs['model'] == "model_no_key"

        mock_openai_class.reset_mock()

        await create_llm_from_env(config=base_config) # Added await
        args_cfg, kwargs_cfg = mock_openai_class.call_args
        assert kwargs_cfg['config'].api_key == "base_cfg_key"
        assert kwargs_cfg['model'] == "base_cfg_model"
