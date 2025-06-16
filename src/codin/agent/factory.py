"""Agent factory with unified endpoint configuration."""

import logging
import typing as _t

from codin.memory import Memory
from codin.model.factory import LLMFactory
from codin.prompt.registry import PromptRegistry
from codin.sandbox.base import Sandbox
from codin.sandbox.local import LocalSandbox
from codin.tool.registry import ToolRegistry, ToolEndpoint
from codin.tool.base import Toolset

from .config import AgentEndpointConfig
from .code_agent import CodeAgent

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating agents with unified endpoint configuration."""
    
    @classmethod
    async def create_code_agent(
        cls,
        config: AgentEndpointConfig | None = None,
        sandbox: Sandbox | None = None,
        additional_toolsets: list[Toolset] | None = None,
    ) -> CodeAgent:
        """Create a CodeAgent with unified endpoint configuration.
        
        Args:
            config: Agent endpoint configuration (defaults to environment-based config)
            sandbox: Custom sandbox instance (defaults to LocalSandbox)
            additional_toolsets: Additional toolsets to register
            
        Returns:
            Configured CodeAgent instance
        """
        config = config or AgentEndpointConfig.from_env()
        
        # Create LLM from model configuration
        llm = await LLMFactory.create_llm(endpoint_config=config.model_config)
        
        # Create memory service
        memory_service = Memory(config.memory_endpoint)
        
        # Create tool registry
        tool_registry = await cls._create_tool_registry(config)
        
        # Set up prompt registry
        PromptRegistry.set_endpoint(config.prompt_endpoint.endpoint)
        
        # Create sandbox
        sandbox = sandbox or LocalSandbox()
        
        # Create agent
        agent = CodeAgent(
            name=config.name,
            description=config.description,
            llm_model=None,  # We'll set the LLM directly
            sandbox=sandbox,
            tool_registry=tool_registry,
            toolsets=additional_toolsets,
            approval_mode=config.approval_mode,
            max_turns=config.max_turns,
            rules=config.rules,
            memory_system=memory_service,
            debug=config.debug,
        )
        
        # Set the pre-configured LLM
        agent.llm = llm
        
        logger.info(f"Created CodeAgent '{config.name}' with unified endpoint configuration")
        return agent
    
    @classmethod
    async def _create_tool_registry(cls, config: AgentEndpointConfig) -> ToolRegistry:
        """Create tool registry from endpoint configuration."""
        tool_endpoint = ToolEndpoint(
            endpoint=config.tool_endpoint.endpoint,
            fallback_endpoint=config.tool_endpoint.fallback_endpoint,
            auth=config.tool_endpoint.auth,
            timeout=config.tool_endpoint.timeout,
        )
        
        # Create registry from endpoint
        registry = await ToolRegistry.from_endpoint(
            endpoint=tool_endpoint.endpoint,
            auth=tool_endpoint.auth,
            timeout=tool_endpoint.timeout,
        )
        
        return registry
    
    @classmethod
    async def create_local_agent(
        cls,
        model_name: str = "gpt-4",
        sandbox: Sandbox | None = None,
    ) -> CodeAgent:
        """Create an agent configured for local development.
        
        Args:
            model_name: Model to use for the agent
            sandbox: Custom sandbox instance
            
        Returns:
            CodeAgent configured for local development
        """
        config = AgentEndpointConfig.local_dev()
        config.model_config.model_name = model_name
        
        return await cls.create_code_agent(config, sandbox)
    
    @classmethod
    async def create_remote_agent(
        cls,
        model_base_url: str,
        memory_service_url: str,
        model_name: str = "gpt-4",
        sandbox: Sandbox | None = None,
    ) -> CodeAgent:
        """Create an agent configured for remote services.
        
        Args:
            model_base_url: Base URL for the model service
            memory_service_url: URL for the memory service
            model_name: Model to use
            sandbox: Custom sandbox instance
            
        Returns:
            CodeAgent configured for remote services
        """
        config = AgentEndpointConfig.remote_prod(
            model_base_url=model_base_url,
            memory_service_url=memory_service_url,
        )
        config.model_config.model_name = model_name
        
        return await cls.create_code_agent(config, sandbox)
    
    @classmethod
    async def create_hybrid_agent(
        cls,
        model_base_url: str,
        model_name: str = "gpt-4",
        local_memory: bool = True,
        local_tools: bool = True,
        local_prompts: bool = True,
        sandbox: Sandbox | None = None,
    ) -> CodeAgent:
        """Create an agent with hybrid local/remote configuration.
        
        Args:
            model_base_url: Base URL for the model service
            model_name: Model to use
            local_memory: Use local memory storage
            local_tools: Use local tool registry
            local_prompts: Use local prompt storage
            sandbox: Custom sandbox instance
            
        Returns:
            CodeAgent with hybrid configuration
        """
        config = AgentEndpointConfig.hybrid(
            model_base_url=model_base_url,
            local_memory=local_memory,
            local_tools=local_tools,
            local_prompts=local_prompts,
        )
        config.model_config.model_name = model_name
        
        return await cls.create_code_agent(config, sandbox)


# Convenience functions
async def create_agent(
    config: AgentEndpointConfig | None = None,
    sandbox: Sandbox | None = None,
) -> CodeAgent:
    """Create an agent with the given configuration."""
    return await AgentFactory.create_code_agent(config, sandbox)


async def create_local_agent(
    model_name: str = "gpt-4",
    sandbox: Sandbox | None = None,
) -> CodeAgent:
    """Create an agent for local development."""
    return await AgentFactory.create_local_agent(model_name, sandbox)


async def create_remote_agent(
    model_base_url: str,
    memory_service_url: str,
    model_name: str = "gpt-4",
    sandbox: Sandbox | None = None,
) -> CodeAgent:
    """Create an agent for remote services."""
    return await AgentFactory.create_remote_agent(
        model_base_url, memory_service_url, model_name, sandbox
    )