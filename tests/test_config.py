"""Tests for the codin.config module."""

import os
import pytest
from unittest.mock import patch, mock_open
import yaml # For creating test config file content
import tempfile # For creating temporary config file
from pathlib import Path

from codin.config import get_config, load_config, get_api_key, CodinConfig, get_default_model_configs
from src.codin.model.config import ModelConfig as ModelClientConfig # Use alias for clarity


class TestConfig:
    """Test cases for the config module."""

    @pytest.fixture(autouse=True)
    def setup_environment(self, monkeypatch):
        """Set up test environment variables."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("MY_CUSTOM_KEY", raising=False) # For custom env_key tests
        monkeypatch.delenv("UNUSED_KEY", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("DEBUG", raising=False)

        # Clear config cache
        import codin.config
        codin.config._config = None
        codin.config._config_file = None # Also clear the stored config file path

    def test_default_config(self):
        """Test that default configuration is loaded correctly."""
        with patch("codin.config.find_config_files", return_value=[]):
            config = load_config()

        assert isinstance(config, CodinConfig)
        assert config.model == "gpt-4o-mini"
        assert config.provider == "openai"
        assert config.debug is False

    def test_environment_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("DEBUG", "true")

        with patch("codin.config.find_config_files", return_value=[]):
            config = load_config()

        assert config.model == "gpt-4o"
        assert config.provider == "anthropic"
        assert config.debug is True
        assert config.verbose is False

    def test_get_config(self, monkeypatch):
        """Test the get_config function."""
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")

        with patch("codin.config.find_config_files", return_value=[]): # Ensure no file loading interference
            config = get_config()
            assert isinstance(config, CodinConfig)
            assert config.model == "gpt-4o"

            config2 = get_config() # Should return cached instance
            assert config is config2

    def test_get_config_reloads_if_file_changes(self, monkeypatch):
        dummy_path_1 = Path("dummy_cfg_1.yaml")
        dummy_path_2 = Path("dummy_cfg_2.yaml")

        with patch("codin.config.find_config_files", return_value=[dummy_path_1]), \
             patch("codin.config.load_config_file", return_value={"model": "model1"}):
            cfg1 = get_config(config_file=dummy_path_1)
            assert cfg1.model == "model1"

        # Clear global _config to simulate new call context for get_config
        import codin.config as AppConfigModule
        AppConfigModule._config = None
        AppConfigModule._config_file = None

        with patch("codin.config.find_config_files", return_value=[dummy_path_2]), \
             patch("codin.config.load_config_file", return_value={"model": "model2"}):
            cfg2 = get_config(config_file=dummy_path_2) # Different file
            assert cfg2.model == "model2"
            assert cfg1 is not cfg2


    def test_get_api_key_uses_defaults(self, monkeypatch):
        """Test retrieving API keys using default model_configs."""
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key-from-env")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key-from-env")

        # Ensure find_config_files returns empty to use get_default_model_configs
        with patch("codin.config.find_config_files", return_value=[]):
            # Clear global _config to force reload defaults
            import codin.config as AppConfigModule
            AppConfigModule._config = None
            AppConfigModule._config_file = None

            assert get_api_key("openai") == "openai-key-from-env"
            assert get_api_key("anthropic") == "anthropic-key-from-env"
            assert get_api_key("unknown") is None

    # New tests for ModelConfig refactoring
    def test_get_default_model_configs_uses_new_model_config(self, monkeypatch):
        """Test get_default_model_configs returns new ModelClientConfig instances."""
        monkeypatch.setenv("OPENAI_API_KEY", "default_openai_key_for_test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "default_anthropic_key_for_test")

        # Call directly, no need to mock find_config_files as it's not used by this func
        default_configs = get_default_model_configs()

        assert "openai" in default_configs
        assert isinstance(default_configs["openai"], ModelClientConfig)
        assert default_configs["openai"].api_key == "default_openai_key_for_test"
        assert default_configs["openai"].provider == "openai"
        assert default_configs["openai"].model_name == "gpt-4o-mini"

        assert "anthropic" in default_configs
        assert isinstance(default_configs["anthropic"], ModelClientConfig)
        assert default_configs["anthropic"].api_key == "default_anthropic_key_for_test"
        assert default_configs["anthropic"].provider == "anthropic"
        assert default_configs["anthropic"].model_name == "claude-3.5-sonnet"
        assert default_configs["anthropic"].api_version == "2023-06-01"

    def _create_temp_config_file(self, content: dict, suffix=".yaml") -> Path:
        """Helper to create a temporary YAML/JSON config file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
        if suffix == ".yaml":
            yaml.dump(content, temp_file)
        elif suffix == ".json":
            json.dump(content, temp_file)
        temp_file.close()
        return Path(temp_file.name)

    def test_load_config_maps_old_model_config_format(self, monkeypatch):
        """Test load_config correctly maps old model_configs format from file."""
        monkeypatch.setenv("MY_FILE_KEY", "file_api_key")

        old_format_content = {
            "model_configs": {
                "custom_provider": {
                    # "name": "Custom Provider", # Old field, not used by new ModelClientConfig directly for provider
                    "base_url": "https://custom.api/v1",
                    "env_key": "MY_FILE_KEY",
                    "models": ["custom-model-1", "custom-model-2"]
                }
            }
        }
        temp_config_path = self._create_temp_config_file(old_format_content)

        try:
            with patch("codin.config.find_config_files", return_value=[temp_config_path]):
                config = load_config()

            custom_cfg = config.model_configs.get("custom_provider")
            assert custom_cfg is not None
            assert isinstance(custom_cfg, ModelClientConfig)
            assert custom_cfg.provider == "custom_provider"
            assert custom_cfg.api_key == "file_api_key"
            assert custom_cfg.base_url == "https://custom.api/v1"
            assert custom_cfg.model_name == "custom-model-1"
        finally:
            os.unlink(temp_config_path)

    def test_load_config_uses_new_model_config_fields_from_file(self, monkeypatch):
        """Test load_config uses new ModelClientConfig fields if present in file."""
        new_format_content = {
            "model_configs": {
                "new_provider": {
                    # provider field in ModelClientConfig is set by the dict key 'new_provider'
                    "model_name": "new-model-x",
                    "api_key": "direct_api_key_in_file", # Direct API key
                    "base_url": "https://new.api/v2",
                    "api_version": "2024-01-01",
                    "timeout": 77.0,
                    # env_key should be ignored if api_key is present
                    "env_key": "SOME_OTHER_KEY_THAT_SHOULD_BE_IGNORED"
                }
            }
        }
        # Ensure the ignored env key is not set, or set to something different
        monkeypatch.setenv("SOME_OTHER_KEY_THAT_SHOULD_BE_IGNORED", "ignored_env_val")
        temp_config_path = self._create_temp_config_file(new_format_content)

        try:
            with patch("codin.config.find_config_files", return_value=[temp_config_path]):
                config = load_config()

            new_cfg = config.model_configs.get("new_provider")
            assert new_cfg is not None
            assert isinstance(new_cfg, ModelClientConfig)
            assert new_cfg.provider == "new_provider"
            assert new_cfg.model_name == "new-model-x"
            assert new_cfg.api_key == "direct_api_key_in_file" # Direct key wins
            assert new_cfg.base_url == "https://new.api/v2"
            assert new_cfg.api_version == "2024-01-01"
            assert new_cfg.timeout == 77.0
        finally:
            os.unlink(temp_config_path)

    def test_get_api_key_with_new_model_config_loaded_from_file(self, monkeypatch):
        """Test get_api_key after config loading with new ModelClientConfig from file."""
        content = {
            "model_configs": {
                "provider_for_get_key": {
                    "api_key": "key_for_get_api_key_test_from_file"
                }
            }
        }
        temp_config_path = self._create_temp_config_file(content)
        try:
            with patch("codin.config.find_config_files", return_value=[temp_config_path]):
                import codin.config as AppConfigModule
                AppConfigModule._config = None # Force reload
                AppConfigModule._config_file = None

                # get_api_key calls get_config which calls load_config
                retrieved_key = get_api_key("provider_for_get_key")
                assert retrieved_key == "key_for_get_api_key_test_from_file"
        finally:
            os.unlink(temp_config_path)
