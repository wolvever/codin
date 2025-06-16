"""Unified agent configuration using endpoint system."""

import os
import typing as _t
from dataclasses import dataclass, field

from codin.endpoint import EndpointConfig
from codin.model.endpoint_config import ModelEndpointConfig
from codin.config import ApprovalMode


@dataclass
class AgentEndpointConfig:
    """Unified endpoint configuration for agents."""
    
    # Model configuration
    model_config: ModelEndpointConfig | None = None
    
    # Memory endpoint
    memory_endpoint: EndpointConfig | None = None
    
    # Tool endpoint
    tool_endpoint: EndpointConfig | None = None
    
    # Prompt endpoint  
    prompt_endpoint: EndpointConfig | None = None
    
    # Replay endpoint
    replay_endpoint: EndpointConfig | None = None
    
    # Agent configuration
    name: str = "CodeAgent"
    description: str = "Python agent with sandbox + tools support and iterative execution"
    approval_mode: ApprovalMode = ApprovalMode.UNSAFE_ONLY
    max_turns: int = 100
    rules: str | None = None
    debug: bool = False
    
    def __post_init__(self):
        """Initialize endpoint configurations from environment if not provided."""
        if self.model_config is None:
            self.model_config = ModelEndpointConfig.from_env()
        
        if self.memory_endpoint is None:
            self.memory_endpoint = EndpointConfig.memory()
        
        if self.tool_endpoint is None:
            self.tool_endpoint = EndpointConfig.tool()
        
        if self.prompt_endpoint is None:
            self.prompt_endpoint = EndpointConfig.prompt()
        
        if self.replay_endpoint is None:
            self.replay_endpoint = EndpointConfig.replay()
    
    @classmethod
    def from_env(cls) -> "AgentEndpointConfig":
        """Create agent configuration from environment variables."""
        return cls(
            name=os.getenv("CODIN_AGENT_NAME", "CodeAgent"),
            description=os.getenv("CODIN_AGENT_DESCRIPTION", "Python agent with sandbox + tools support"),
            max_turns=int(os.getenv("CODIN_AGENT_MAX_TURNS", "100")),
            rules=os.getenv("CODIN_AGENT_RULES"),
            debug=os.getenv("CODIN_AGENT_DEBUG", "false").lower() == "true",
        )
    
    @classmethod
    def local_dev(cls) -> "AgentEndpointConfig":
        """Create configuration for local development."""
        return cls(
            model_config=ModelEndpointConfig.openai(),
            memory_endpoint=EndpointConfig(endpoint="fs://./data/memory"),
            tool_endpoint=EndpointConfig(endpoint="fs://./tools"),
            prompt_endpoint=EndpointConfig(endpoint="fs://./prompt_templates"),
            replay_endpoint=EndpointConfig(endpoint="fs://./data/replay"),
            debug=True,
        )
    
    @classmethod
    def remote_prod(
        cls, 
        model_base_url: str,
        memory_service_url: str,
        tool_service_url: str | None = None,
        prompt_service_url: str | None = None
    ) -> "AgentEndpointConfig":
        """Create configuration for remote production environment."""
        return cls(
            model_config=ModelEndpointConfig.openai(base_url=model_base_url),
            memory_endpoint=EndpointConfig(endpoint=memory_service_url),
            tool_endpoint=EndpointConfig(endpoint=tool_service_url or "fs://./tools"),
            prompt_endpoint=EndpointConfig(endpoint=prompt_service_url or "fs://./prompt_templates"),
            replay_endpoint=EndpointConfig(endpoint="fs://./data/replay"),
        )
    
    @classmethod
    def hybrid(
        cls,
        model_base_url: str,
        local_memory: bool = True,
        local_tools: bool = True,
        local_prompts: bool = True,
    ) -> "AgentEndpointConfig":
        """Create hybrid configuration with mix of local and remote services."""
        return cls(
            model_config=ModelEndpointConfig.openai(base_url=model_base_url),
            memory_endpoint=EndpointConfig(endpoint="fs://./data/memory") if local_memory else EndpointConfig.memory(),
            tool_endpoint=EndpointConfig(endpoint="fs://./tools") if local_tools else EndpointConfig.tool(),
            prompt_endpoint=EndpointConfig(endpoint="fs://./prompt_templates") if local_prompts else EndpointConfig.prompt(),
            replay_endpoint=EndpointConfig(endpoint="fs://./data/replay"),
        )