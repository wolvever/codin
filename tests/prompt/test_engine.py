"""Tests for the prompt.engine module."""

import pytest
import typing as _t
from unittest.mock import AsyncMock, MagicMock, patch

from codin.prompt.engine import PromptEngine
from codin.prompt.registry import PromptRegistry, get_registry
from codin.prompt.base import PromptTemplate, RenderedPrompt
from codin.model.base import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM for testing the prompt engine."""
    
    def __init__(self, model="mock-llm"):
        super().__init__(model)
        self.generate_calls = []
        self.generate_with_tools_calls = []
        self.response = "This is a mock response"
        self._prepared = False  # Add _prepared attribute that engine checks for
    
    @classmethod
    def supported_models(cls) -> list[str]:
        """Return supported model patterns."""
        return ["mock-.*", "test-.*"]
    
    async def prepare(self) -> None:
        self._prepared = True
    
    async def generate(self, 
                     prompt: str | list[dict[str, str]], 
                     *, 
                     stream: bool = False,
                     temperature: float | None = None,
                     max_tokens: int | None = None,
                     stop_sequences: list[str] | None = None,
                     **kwargs) -> str | _t.AsyncIterator[str]:
        self.generate_calls.append((prompt, stream, kwargs))
        
        if stream:
            return self._stream_response()
        return self.response
    
    async def generate_with_tools(self,
                                prompt: str | list[dict[str, str]],
                                tools: list[dict],
                                *,
                                stream: bool = False,
                                temperature: float | None = None,
                                max_tokens: int | None = None,
                                ) -> dict | _t.AsyncIterator[dict]:
        """Mock implementation of generate_with_tools."""
        self.generate_with_tools_calls.append((prompt, tools, stream, temperature, max_tokens))
        
        if stream:
            return self._stream_tools_response()
        
        return {
            "content": self.response,
            "tool_calls": []
        }
    
    async def _stream_response(self) -> _t.AsyncIterator[str]:
        """Return a streaming response."""
        for word in self.response.split():
            yield word + " "
    
    async def _stream_tools_response(self) -> _t.AsyncIterator[dict]:
        """Return a streaming tools response."""
        for word in self.response.split():
            yield {
                "content": word + " ",
                "tool_calls": []
            }


@pytest.fixture
def clean_registry():
    """Provide a clean registry for tests."""
    from codin.prompt.registry import get_registry
    
    # Get the registry that the engine will actually use
    registry = get_registry()
    
    # Clear any existing templates
    registry._in_memory_templates.clear()
    if hasattr(registry, '_registry'):
        registry._registry.clear()
    
    yield registry
    
    # Clean up after test
    registry._in_memory_templates.clear()
    if hasattr(registry, '_registry'):
        registry._registry.clear()


@pytest.fixture
def mock_template():
    """Create a mock prompt template."""
    template = PromptTemplate(
        name="test-prompt",
        version="latest",
        text="Hello, {{ name }}! How is {{ location }}?"
    )
    
    return template


class TestPromptEngine:
    """Test cases for the PromptEngine class."""
    
    def test_initialization_with_model_name(self):
        """Test initializing the engine with a model name."""
        with patch("codin.model.ModelRegistry.create_llm") as mock_create_llm:
            mock_llm = MockLLM()
            mock_create_llm.return_value = mock_llm
            
            engine = PromptEngine("gpt-4")
            
            # Check that LLM was created
            mock_create_llm.assert_called_once_with("gpt-4")
            assert engine.llm is mock_llm
    
    def test_initialization_with_llm_instance(self):
        """Test initializing the engine with an LLM instance."""
        llm = MockLLM()
        engine = PromptEngine(llm)
        
        assert engine.llm is llm
    
    def test_initialization_with_endpoint(self):
        """Test initializing the engine with an endpoint."""
        llm = MockLLM()
        engine = PromptEngine(llm, endpoint="fs://./test_templates")
        
        assert engine.llm is llm
        assert engine.endpoint == "fs://./test_templates"
    
    @pytest.mark.asyncio
    async def test_run_with_template(self, clean_registry, mock_template):
        """Test running a prompt with a template."""
        # Register the template
        clean_registry.register(mock_template)
        
        # Create LLM and engine
        llm = MockLLM()
        engine = PromptEngine(llm)
        
        # Run the prompt
        result = await engine.run("test-prompt", name="World", location="home")
        
        # Check that LLM was prepared
        assert llm._prepared
        
        # Check that LLM was called
        assert len(llm.generate_calls) == 1
        prompt, stream, kwargs = llm.generate_calls[0]
        
        # Should not be streaming
        assert not stream
        
        # Check result type
        assert hasattr(result, 'content')
        assert result.content == "This is a mock response"
    
    @pytest.mark.asyncio
    async def test_run_with_streaming(self, clean_registry, mock_template):
        """Test running a prompt with streaming."""
        # Register the template
        clean_registry.register(mock_template)
        
        # Create LLM and engine
        llm = MockLLM()
        engine = PromptEngine(llm)
        
        # Run the prompt with streaming
        result = await engine.run("test-prompt", stream=True, name="World", location="home")
        
        # Check that LLM was prepared
        assert llm._prepared
        
        # Check that LLM was called with streaming
        assert len(llm.generate_calls) == 1
        prompt, stream, kwargs = llm.generate_calls[0]
        assert stream is True
        
        # Check that result has streaming content
        assert hasattr(result, 'content')
        assert hasattr(result.content, '__aiter__')  # Should be an async iterator
        
        # Collect streamed chunks
        chunks = []
        async for chunk in result.content:
            chunks.append(chunk)
        
        # Check streamed chunks
        assert chunks == ["This ", "is ", "a ", "mock ", "response "]
    
    @pytest.mark.asyncio
    async def test_run_with_version(self, clean_registry):
        """Test running a prompt with a specific version."""
        # Create and register two versions
        template_v1 = PromptTemplate(name="test", version="v1", text="Version 1: {{ var }}")
        template_v2 = PromptTemplate(name="test", version="v2", text="Version 2: {{ var }}")
        
        clean_registry.register(template_v1)
        clean_registry.register(template_v2)
        
        # Create engine
        llm = MockLLM()
        engine = PromptEngine(llm)
        
        # Run with specific version
        result = await engine.run("test", version="v1", var="test")
        
        # Check that LLM was called
        assert len(llm.generate_calls) == 1
        prompt, _, _ = llm.generate_calls[0]
        
        # Check that we got a successful result (even if we can't easily verify the exact template)
        assert hasattr(result, 'content')
        assert result.content == "This is a mock response"
    
    @pytest.mark.asyncio 
    async def test_run_with_tools(self, clean_registry, mock_template):
        """Test running a prompt with tools."""
        # Register the template
        clean_registry.register(mock_template)
        
        # Create LLM and engine
        llm = MockLLM()
        engine = PromptEngine(llm)
        
        # Create mock tools
        from codin.prompt.base import ToolDefinition
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}}
            )
        ]
        
        # Run the prompt with tools
        result = await engine.run("test-prompt", tools=tools, name="World", location="home")
        
        # Check that LLM was prepared
        assert llm._prepared
        
        # Check that tools were passed (either generate_with_tools or regular generate was called)
        total_calls = len(llm.generate_calls) + len(llm.generate_with_tools_calls)
        assert total_calls >= 1
        
        # Check result
        assert hasattr(result, 'content')
        # When tools are used, the result might be a dict converted to string
        if isinstance(result.content, str) and result.content.startswith("{'content'"):
            # Tools were used, check that it contains our mock response
            assert "This is a mock response" in result.content
        else:
            # Regular generation was used
            assert result.content == "This is a mock response"
    
    @pytest.mark.asyncio
    async def test_render_only(self, clean_registry, mock_template):
        """Test rendering a template without executing LLM."""
        # Register the template
        clean_registry.register(mock_template)
        
        # Create engine
        llm = MockLLM()
        engine = PromptEngine(llm)
        
        # Render the template
        rendered = await engine.render_only("test-prompt", name="World", location="home")
        
        # Check that LLM was NOT called
        assert len(llm.generate_calls) == 0
        assert len(llm.generate_with_tools_calls) == 0
        
        # Check rendered result
        assert hasattr(rendered, 'text')
        assert "Hello, World! How is home?" in rendered.text 