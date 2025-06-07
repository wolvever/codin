"""Tests for the codin.config module."""

import os
import pytest
from unittest.mock import patch

from codin.config import get_config, load_config, get_api_key, CodinConfig


class TestConfig:
    """Test cases for the config module."""

    @pytest.fixture(autouse=True)
    def setup_environment(self, monkeypatch):
        """Set up test environment variables."""
        # Clear environment variables we'll test
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("DEBUG", raising=False)

        # Clear config cache
        import codin.config

        codin.config._config = None

    def test_default_config(self):
        """Test that default configuration is loaded correctly."""
        # Mock find_config_files to return empty list (no config files)
        with patch("codin.config.find_config_files", return_value=[]):
            config = load_config()

        # Check some default values
        assert isinstance(config, CodinConfig)
        assert config.model == "gpt-4o-mini"
        assert config.provider == "openai"
        assert config.debug is False
        assert config.verbose is False

    def test_environment_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        # Set environment variables
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("DEBUG", "true")

        # Mock find_config_files to return empty list (no config files)
        with patch("codin.config.find_config_files", return_value=[]):
            # Load config
            config = load_config()

        # Check overridden values
        assert config.model == "gpt-4o"
        assert config.provider == "anthropic"
        assert config.debug is True  # Converted to boolean

        # Check that non-overridden values remain as defaults
        assert config.verbose is False

    def test_get_config(self, monkeypatch):
        """Test the get_config function."""
        # Set an environment variable
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")

        # Get the config object
        config = get_config()
        assert isinstance(config, CodinConfig)
        assert config.model == "gpt-4o"

        # Test that it returns the same instance (cached)
        config2 = get_config()
        assert config is config2

    def test_parse_values(self, monkeypatch):
        """Test parsing of different value types from environment variables."""
        # Test boolean parsing for debug flag
        monkeypatch.setenv("DEBUG", "true")
        config = load_config()
        assert config.debug is True

        # Clear cache and test false
        import codin.config

        codin.config._config = None
        monkeypatch.setenv("DEBUG", "false")
        config = load_config()
        assert config.debug is False

        # Test string values
        codin.config._config = None
        monkeypatch.setenv("LLM_MODEL", "claude-3-5-sonnet")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        config = load_config()
        assert config.model == "claude-3-5-sonnet"
        assert config.provider == "anthropic"

    def test_get_api_key(self, monkeypatch):
        """Test retrieving API keys for different providers."""
        # Set API keys
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

        # Get keys for different providers
        assert get_api_key("openai") == "openai-key"
        assert get_api_key("anthropic") == "anthropic-key"

        # Unknown provider
        assert get_api_key("unknown") is None

        # Missing key
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        assert get_api_key("anthropic") is None

    def test_mcp_servers_config(self):
        """Test that MCP servers are loaded from configuration."""
        config = get_config()

        # Check that MCP servers are loaded (from config.yaml)
        assert hasattr(config, "mcp_servers")
        assert isinstance(config.mcp_servers, dict)

        # The actual servers depend on what's in config.yaml
        # Just verify the structure is correct
        for server_name, server_config in config.mcp_servers.items():
            assert hasattr(server_config, "description")
            # Either has command (stdio) or url (sse)
            assert hasattr(server_config, "command") or hasattr(server_config, "url")
