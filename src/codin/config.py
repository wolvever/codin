"""Configuration management for codin.

This module provides configuration loading and management capabilities similar
to the original codex CLI, including support for multiple providers, codin_rules.md
files, and environment-based configuration.
"""

import json
import os
import typing as _t

from enum import Enum
from pathlib import Path

import yaml

from pydantic import BaseModel, Field


# Define ApprovalMode locally to avoid circular import
class ApprovalMode(str, Enum):
    """Unified approval mode for tool execution and CLI."""

    ALWAYS = "always"  # Always ask for approval (same as SUGGEST)
    UNSAFE_ONLY = "unsafe_only"  # Only ask for potentially unsafe operations (same as AUTO_EDIT)
    NEVER = "never"  # Never ask, auto-approve everything (same as FULL_AUTO)


__all__ = [
    "ApprovalMode",
    "CodinConfig",
    "HistoryConfig",
    "MCPServerConfig",
    "ModelConfig",
    "AgentConfig",
    "get_default_model_configs",
    "get_api_key",
    "get_config",
    "load_agents_instructions",
    "load_config",
]


class ModelConfig(BaseModel):
    """Configuration for a model provider."""

    name: str
    base_url: str
    env_key: str
    models: list[str] = Field(default_factory=list)


class HistoryConfig(BaseModel):
    """Configuration for conversation history."""

    max_size: int = 1000
    save_history: bool = True
    sensitive_patterns: list[str] = Field(default_factory=list)


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None  # For SSE servers
    description: str = ""


class AgentConfig(BaseModel):
    """Configuration for an agent using a registered model config."""

    model_config_name: str
    name: str | None = None
    config: dict[str, _t.Any] = Field(default_factory=dict)


class CodinConfig(BaseModel):
    """Main configuration class for codin."""

    # Core settings
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    approval_mode: ApprovalMode = ApprovalMode.ALWAYS

    # Output settings
    verbose: bool = False
    debug: bool = False
    quiet_mode: bool = False
    max_output_lines: int = -1  # Maximum lines to show in CLI output (-1 for unlimited)
    max_output_chars: int = 20000  # Maximum characters for tool output

    # Agent behavior settings
    enable_rules: bool = True
    max_turns: int = 100
    prevent_duplicate_tools: bool = False  # Prevent duplicate tool calls in same turn

    # Model configurations
    model_configs: dict[str, ModelConfig] = Field(default_factory=dict)

    # Agent configurations
    agent_configs: dict[str, AgentConfig] = Field(default_factory=dict)

    # History settings
    history: HistoryConfig = Field(default_factory=HistoryConfig)

    # MCP server configurations
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    # Project documentation settings
    agents_files: list[str] = Field(default_factory=list)


def get_default_model_configs() -> dict[str, ModelConfig]:
    """Get default model configurations matching codex."""
    return {
        "openai": ModelConfig(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            env_key="OPENAI_API_KEY",
            models=["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
        ),
        "azure": ModelConfig(
            name="Azure OpenAI",
            base_url="https://YOUR_PROJECT_NAME.openai.azure.com/openai",
            env_key="AZURE_OPENAI_API_KEY",
            models=["gpt-4", "gpt-35-turbo"],
        ),
        "openrouter": ModelConfig(
            name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            env_key="OPENROUTER_API_KEY",
            models=["anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
        ),
        "gemini": ModelConfig(
            name="Gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            env_key="GEMINI_API_KEY",
            models=["gemini-1.5-pro", "gemini-1.5-flash"],
        ),
        "ollama": ModelConfig(
            name="Ollama",
            base_url="http://localhost:11434/v1",
            env_key="OLLAMA_API_KEY",
            models=["llama2", "codellama", "mistral"],
        ),
        "mistral": ModelConfig(
            name="Mistral",
            base_url="https://api.mistral.ai/v1",
            env_key="MISTRAL_API_KEY",
            models=["mistral-large", "mistral-medium", "mistral-small"],
        ),
        "deepseek": ModelConfig(
            name="DeepSeek",
            base_url="https://api.deepseek.com",
            env_key="DEEPSEEK_API_KEY",
            models=["deepseek-coder", "deepseek-chat"],
        ),
        "xai": ModelConfig(
            name="xAI",
            base_url="https://api.x.ai/v1",
            env_key="XAI_API_KEY",
            models=["grok-1", "grok-beta"],
        ),
        "groq": ModelConfig(
            name="Groq",
            base_url="https://api.groq.com/openai/v1",
            env_key="GROQ_API_KEY",
            models=["llama2-70b-4096", "mixtral-8x7b-32768"],
        ),
        "arceeai": ModelConfig(
            name="ArceeAI",
            base_url="https://conductor.arcee.ai/v1",
            env_key="ARCEEAI_API_KEY",
            models=["arcee-agent", "arcee-nova"],
        ),
        "anthropic": ModelConfig(
            name="Anthropic",
            base_url="https://api.anthropic.com",
            env_key="ANTHROPIC_API_KEY",
            models=["claude-3.5-sonnet", "claude-3-opus", "claude-3-haiku"],
        ),
    }


def load_config_file(config_path: Path) -> dict[str, _t.Any]:
    """Load configuration from a YAML or JSON file."""
    if not config_path.exists():
        return {}

    try:
        content = config_path.read_text(encoding="utf-8")

        if config_path.suffix.lower() in [".yaml", ".yml"]:
            return yaml.safe_load(content) or {}
        if config_path.suffix.lower() == ".json":
            return json.loads(content) or {}
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")


def find_config_files(config_file: Path | None = None) -> list[Path]:
    """Find all possible configuration files in order of precedence.

    Args:
        config_file: Optional custom config file path to use instead of defaults

    Searches for config files in:
    1. ~/.codin/ (global user config)
    2. Current working directory (project-specific config)
    3. Custom config file if specified (highest precedence)

    Later files override earlier ones.
    """
    config_files = []

    # If custom config file is specified, use only that
    if config_file:
        if config_file.exists():
            config_files.append(config_file)
        else:
            raise ValueError(f"Specified config file not found: {config_file}")
        return config_files

    # 1. Global user config in ~/.codin/
    config_dir = Path.home() / ".codin"
    for filename in ["config.yaml", "config.yml", "config.json"]:
        config_path = config_dir / filename
        if config_path.exists():
            config_files.append(config_path)

    # 2. Local project config in current directory
    cwd = Path.cwd()
    for filename in ["config.yaml", "config.yml", "config.json"]:
        config_path = cwd / filename
        if config_path.exists():
            config_files.append(config_path)

    return config_files


def load_agents_instructions() -> str:
    """Load codin_rules.md files from various locations, merging them top-down.

    Looks for codin_rules.md files in:
    1. ~/.codin/codin_rules.md - personal global guidance
    2. codin_rules.md at repo root - shared project notes
    3. codin_rules.md in current working directory - sub-folder specifics

    Returns:
        Combined instructions from all found codin_rules.md files
    """
    instructions_parts = []

    # 1. Global user instructions
    global_agents = Path.home() / ".codin" / "codin_rules.md"
    if global_agents.exists():
        try:
            content = global_agents.read_text(encoding="utf-8").strip()
            if content:
                instructions_parts.append(f"# Global Instructions\n{content}")
        except Exception:
            pass  # Ignore errors reading global instructions

    # 2. Project root instructions
    cwd = Path.cwd()

    # Find git root or use current directory
    git_root = cwd
    current = cwd
    while current != current.parent:
        if (current / ".git").exists():
            git_root = current
            break
        current = current.parent

    project_agents = git_root / "codin_rules.md"
    if project_agents.exists() and project_agents != cwd / "codin_rules.md":
        try:
            content = project_agents.read_text(encoding="utf-8").strip()
            if content:
                instructions_parts.append(f"# Project Instructions\n{content}")
        except Exception:
            pass

    # 3. Current directory instructions
    local_agents = cwd / "codin_rules.md"
    if local_agents.exists():
        try:
            content = local_agents.read_text(encoding="utf-8").strip()
            if content:
                instructions_parts.append(f"# Local Instructions\n{content}")
        except Exception:
            pass

    return "\n\n".join(instructions_parts)


def load_config(config_file: Path | str | None = None) -> CodinConfig:
    """Load configuration from files and environment variables.

    Args:
        config_file: Optional path to a specific config file to use
    """
    # Start with defaults
    config_data = {}

    # Convert string path to Path object if needed
    config_path = None
    if config_file:
        config_path = Path(config_file) if isinstance(config_file, str) else config_file

    # Load from config files
    config_files = find_config_files(config_path)
    for config_file_path in config_files:
        file_config = load_config_file(config_file_path)
        config_data.update(file_config)

    # Override with environment variables
    env_overrides = {}

    # Core settings
    if "LLM_MODEL" in os.environ:
        env_overrides["model"] = os.environ["LLM_MODEL"]

    if "LLM_PROVIDER" in os.environ:
        env_overrides["provider"] = os.environ["LLM_PROVIDER"]

    if "CODIN_APPROVAL_MODE" in os.environ:
        env_overrides["approval_mode"] = os.environ["CODIN_APPROVAL_MODE"]

    if "CODIN_QUIET_MODE" in os.environ:
        env_overrides["quiet_mode"] = os.environ["CODIN_QUIET_MODE"].lower() in ("1", "true", "yes")

    if "CODIN_RULE" in os.environ:
        env_overrides["enable_project_docs"] = os.environ["CODIN_RULE"].lower() in (
            "1",
            "true",
            "yes",
        )

    if "DEBUG" in os.environ:
        env_overrides["debug"] = os.environ["DEBUG"].lower() in ("1", "true", "yes")

    config_data.update(env_overrides)

    # Create config object
    config = CodinConfig()

    # Update with loaded data, handling enum conversions
    for key, value in config_data.items():
        if hasattr(config, key):
            # Special handling for enum fields
            if key == "approval_mode" and isinstance(value, str):
                try:
                    # Convert string to ApprovalMode enum
                    setattr(config, key, ApprovalMode(value))
                except ValueError:
                    # If invalid value, keep default
                    pass
            elif key == "model_configs" and isinstance(value, dict):
                # Convert model_configs dict to ModelConfig objects
                configs = {}
                for cfg_name, cfg_data in value.items():
                    if isinstance(cfg_data, dict):
                        configs[cfg_name] = ModelConfig(
                            name=cfg_data.get("name", cfg_name),
                            base_url=cfg_data.get("base_url", ""),
                            env_key=cfg_data.get("env_key", ""),
                            models=cfg_data.get("models", []),
                        )
                    else:
                        configs[cfg_name] = cfg_data
                setattr(config, key, configs)
            elif key == "agent_configs" and isinstance(value, dict):
                # Convert agent_configs dict to AgentConfig objects
                agents = {}
                for agent_name, agent_data in value.items():
                    if isinstance(agent_data, dict):
                        agents[agent_name] = AgentConfig(
                            model_config_name=agent_data.get("model_config", ""),
                            name=agent_data.get("name"),
                            config=agent_data.get("config", {}),
                        )
                    else:
                        agents[agent_name] = agent_data
                setattr(config, key, agents)
            elif key == "history" and isinstance(value, dict):
                # Convert history dict to HistoryConfig object
                history_config = HistoryConfig(
                    max_size=value.get("max_size", 1000),
                    save_history=value.get("save_history", True),
                    sensitive_patterns=value.get("sensitive_patterns", []),
                )
                setattr(config, key, history_config)
            elif key == "mcp_servers" and isinstance(value, dict):
                # Convert mcp_servers dict to MCPServerConfig objects
                mcp_servers = {}
                for server_name, server_data in value.items():
                    if isinstance(server_data, dict):
                        # Convert dict to MCPServerConfig
                        mcp_servers[server_name] = MCPServerConfig(
                            command=server_data.get("command", ""),
                            args=server_data.get("args", []),
                            env=server_data.get("env", {}),
                            url=server_data.get("url"),
                            description=server_data.get("description", ""),
                        )
                    else:
                        # Already a MCPServerConfig object
                        mcp_servers[server_name] = server_data
                setattr(config, key, mcp_servers)
            else:
                setattr(config, key, value)

    # Set up default model configurations if not configured
    if not config.model_configs:
        config.model_configs = get_default_model_configs()

    return config


# Global config instance
_config: CodinConfig | None = None
_config_file: Path | str | None = None


def get_config(config_file: Path | str | None = None) -> CodinConfig:
    """Get the global configuration instance.

    Args:
        config_file: Optional path to a specific config file to use
    """
    global _config, _config_file

    # Reload config if a different config file is specified
    if config_file != _config_file or _config is None:
        _config = load_config(config_file)
        _config_file = config_file

    return _config


def get_api_key(provider: str | None = None) -> str | None:
    """Get API key for a provider/model configuration."""
    config = get_config()

    if provider is None:
        provider = config.provider

    provider_config = config.model_configs.get(provider)
    if not provider_config:
        return None

    return os.environ.get(provider_config.env_key)
