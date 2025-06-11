from dataclasses import dataclass, field
import typing as _t

@dataclass
class ModelConfig:
    """Configuration for language and embedding models."""
    model_name: _t.Optional[str] = None
    api_key: _t.Optional[str] = None
    base_url: _t.Optional[str] = None
    api_version: _t.Optional[str] = None  # Specific to some models like Anthropic

    # For OpenAI/Azure specific fields, could be added later if needed
    # e.g., deployment_name: _t.Optional[str] = None

    # Timeout settings, could also be part of a more general ClientConfig if separated
    timeout: _t.Optional[float] = None
    connect_timeout: _t.Optional[float] = None

    # Retry settings
    max_retries: _t.Optional[int] = None
    retry_min_wait: _t.Optional[float] = None
    retry_max_wait: _t.Optional[float] = None
    retry_on_status_codes: _t.Optional[list[int]] = None

    # Provider, could be used by factory to determine which model class to use
    provider: _t.Optional[str] = None

    # Any other model-specific parameters can be added as needed
    # For example, temperature, max_tokens for LLMs, but these are usually runtime params
    # Sticking to initialization/client config parameters here.

    # TODO: Consider if we need a separate ClientConfig that ModelConfig might hold.
    # For now, merging them for simplicity as OpenAILLM's ClientConfig was quite tied to model params.
    # OpenAILLM's ClientConfig fields:
    # base_url, timeout, connect_timeout, default_headers, max_retries,
    # retry_min_wait, retry_max_wait, retry_on_status_codes.
    # default_headers will be constructed from api_key, api_version etc.

    def get_client_config_kwargs(self) -> dict[str, _t.Any]:
        """Helper to extract keyword arguments relevant for ClientConfig."""
        kwargs = {}
        if self.base_url is not None:
            kwargs['base_url'] = self.base_url
        if self.timeout is not None:
            kwargs['timeout'] = self.timeout
        if self.connect_timeout is not None:
            kwargs['connect_timeout'] = self.connect_timeout
        if self.max_retries is not None:
            kwargs['max_retries'] = self.max_retries
        if self.retry_min_wait is not None:
            kwargs['retry_min_wait'] = self.retry_min_wait
        if self.retry_max_wait is not None:
            kwargs['retry_max_wait'] = self.retry_max_wait
        if self.retry_on_status_codes is not None:
            kwargs['retry_on_status_codes'] = self.retry_on_status_codes
        return kwargs

# Example of how it might be used:
# common_config = ModelConfig(api_key="...", model_name="gpt-4")
# openai_model = OpenAILLM(config=common_config)

# For a model requiring a specific version:
# anthropic_config = ModelConfig(api_key="...", model_name="claude-2", api_version="2023-06-01")
# anthropic_model = AnthropicLLM(config=anthropic_config)
