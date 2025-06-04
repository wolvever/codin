"""Tests for the model.registry module."""

import re
import pytest
import typing as _t
from unittest.mock import MagicMock, patch

from codin.model.base import BaseModel, BaseLLM, BaseEmbedding, BaseReranker, ModelType
from codin.model.registry import ModelRegistry


class MockModel(BaseModel):
    """Mock model for testing."""
    
    model_type = ModelType.LLM
    
    def __init__(self, model: str):
        super().__init__(model)
    
    @classmethod
    def supported_models(cls) -> list[str]:
        return ["mock-.*", "test-model"]
    
    async def prepare(self) -> None:
        pass


class MockLLM(BaseLLM):
    """Mock LLM for testing."""
    
    def __init__(self, model: str):
        super().__init__(model)
    
    @classmethod
    def supported_models(cls) -> list[str]:
        return ["gpt-test-.*", "llm-test"]
    
    async def prepare(self) -> None:
        pass
    
    async def generate(self, prompt: str | list[dict[str, str]], **kwargs) -> str:
        return f"Response from {self.model}"
    
    async def generate_with_tools(self,
                                prompt: str | list[dict[str, str]],
                                tools: list[dict],
                                *,
                                stream: bool = False,
                                temperature: float | None = None,
                                max_tokens: int | None = None,
                                ) -> dict | _t.AsyncIterator[dict]:
        """Mock implementation of generate_with_tools."""
        result = {
            "content": f"Mock tool response from {self.model}",
            "tool_calls": [
                {
                    "id": "mock_call_123",
                    "type": "function",
                    "function": {
                        "name": "mock_function",
                        "arguments": '{"param": "value"}'
                    }
                }
            ]
        }
        
        if stream:
            async def mock_stream():
                yield result
            return mock_stream()
        else:
            return result


class MockEmbedding(BaseEmbedding):
    """Mock embedding model for testing."""
    
    def __init__(self, model: str):
        super().__init__(model)
    
    @classmethod
    def supported_models(cls) -> list[str]:
        return ["embed-.*"]
    
    async def prepare(self) -> None:
        pass
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture(autouse=True)
def clear_model_registry():
    """Clear the model registry between tests."""
    from codin.model import registry
    
    # Save original registry
    original_registry = dict(registry._model_registry)
    
    yield
    
    # Restore original registry
    registry._model_registry.clear()
    registry._model_registry.update(original_registry)


class TestModelRegistry:
    """Test cases for the ModelRegistry."""
    
    def test_register_decorator(self):
        """Test registering models using the decorator."""
        from codin.model import registry
        
        # Register mock models
        ModelRegistry.register(MockModel)
        ModelRegistry.register(MockLLM)
        ModelRegistry.register(MockEmbedding)
        
        # Check registration
        model_registry = registry._model_registry
        assert "mock-.*" in model_registry
        assert "test-model" in model_registry
        assert "gpt-test-.*" in model_registry
        assert "llm-test" in model_registry
        assert "embed-.*" in model_registry
        
        # Check correct types
        assert model_registry["mock-.*"][0] == ModelType.LLM
        assert model_registry["mock-.*"][1] == MockModel
        
        assert model_registry["gpt-test-.*"][0] == ModelType.LLM
        assert model_registry["gpt-test-.*"][1] == MockLLM
        
        assert model_registry["embed-.*"][0] == ModelType.EMBEDDING
        assert model_registry["embed-.*"][1] == MockEmbedding
    
    def test_resolve_model_class(self):
        """Test resolving model class from model name."""
        # Register mock models
        ModelRegistry.register(MockModel)
        ModelRegistry.register(MockLLM)
        ModelRegistry.register(MockEmbedding)
        
        # Test direct matches
        assert ModelRegistry.resolve_model_class("test-model") == MockModel
        assert ModelRegistry.resolve_model_class("llm-test") == MockLLM
        
        # Test pattern matches
        assert ModelRegistry.resolve_model_class("mock-model") == MockModel
        assert ModelRegistry.resolve_model_class("mock-123") == MockModel
        assert ModelRegistry.resolve_model_class("gpt-test-4") == MockLLM
        assert ModelRegistry.resolve_model_class("embed-ada") == MockEmbedding
        
        # Test with type filter
        assert ModelRegistry.resolve_model_class("mock-model", ModelType.LLM) == MockModel
        assert ModelRegistry.resolve_model_class("embed-ada", ModelType.EMBEDDING) == MockEmbedding
        
        # Test non-existing model
        with pytest.raises(ValueError):
            ModelRegistry.resolve_model_class("unknown-model")
        
        # Test type mismatch
        with pytest.raises(ValueError):
            ModelRegistry.resolve_model_class("embed-ada", ModelType.LLM)
    
    def test_create_llm(self):
        """Test creating an LLM instance."""
        # Register mock LLM
        ModelRegistry.register(MockLLM)
        
        # Create an instance
        llm = ModelRegistry.create_llm("gpt-test-turbo")
        
        assert isinstance(llm, MockLLM)
        assert llm.model == "gpt-test-turbo"
        assert llm.model_type == ModelType.LLM
    
    def test_create_embedding(self):
        """Test creating an embedding model instance."""
        # Register mock embedding
        ModelRegistry.register(MockEmbedding)
        
        # Create an instance
        embedding = ModelRegistry.create_embedding("embed-test")
        
        assert isinstance(embedding, MockEmbedding)
        assert embedding.model == "embed-test"
        assert embedding.model_type == ModelType.EMBEDDING
    
    def test_create_reranker(self):
        """Test creating a reranker model instance."""
        # Create a mock reranker
        class MockReranker(BaseReranker):
            @classmethod
            def supported_models(cls) -> list[str]:
                return ["rerank-.*"]
            
            async def prepare(self) -> None:
                pass
            
            async def rerank(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
                return [(doc, 0.9) for doc in documents]
        
        # Register mock reranker
        ModelRegistry.register(MockReranker)
        
        # Create an instance
        reranker = ModelRegistry.create_reranker("rerank-test")
        
        assert isinstance(reranker, MockReranker)
        assert reranker.model == "rerank-test"
        assert reranker.model_type == ModelType.RERANKER
    
    def test_list_supported_models(self):
        """Test listing supported model patterns."""
        # Register mock models
        ModelRegistry.register(MockModel)
        ModelRegistry.register(MockLLM)
        ModelRegistry.register(MockEmbedding)
        
        # List all models
        all_models = ModelRegistry.list_supported_models()
        assert "mock-.*" in all_models
        assert "test-model" in all_models
        assert "gpt-test-.*" in all_models
        assert "llm-test" in all_models
        assert "embed-.*" in all_models
        
        # List specific types
        llms = ModelRegistry.list_supported_models(ModelType.LLM)
        assert "mock-.*" in llms
        assert "test-model" in llms
        assert "gpt-test-.*" in llms
        assert "llm-test" in llms
        assert "embed-.*" not in llms
        
        embeddings = ModelRegistry.list_supported_models(ModelType.EMBEDDING)
        assert "embed-.*" in embeddings
        assert "mock-.*" not in embeddings 