import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.codin.model.facade_client import ModelFacade
from src.codin.model.base import BaseLLM, BaseEmbedding # BaseReranker if used
from src.codin.model.config import ModelConfig
from src.codin.model.factory import LLMFactory
from src.codin.model.registry import ModelRegistry
from src.codin.model.openai_embedding import OpenAIEmbedding # For mocking specific embedding

# Mock for the LLM instance
@pytest.fixture
def mock_llm_instance():
    llm = AsyncMock(spec=BaseLLM)
    llm.model = "mocked-llm"
    llm.config = ModelConfig(provider="mock_provider", model_name="mocked-llm")
    llm.generate = AsyncMock(return_value="llm_response")
    llm.generate_with_tools = AsyncMock(return_value={"content": "llm_tool_response"})
    llm.close = AsyncMock()
    return llm

# Mock for the Embedding instance
@pytest.fixture
def mock_embedding_instance():
    embedding = AsyncMock(spec=BaseEmbedding)
    embedding.model = "mocked-embedding"
    embedding.embed = AsyncMock(return_value=[[0.1, 0.2]])
    embedding.close = AsyncMock()
    return embedding

# Mocks for Factory and Registry
@pytest.fixture
def mock_llm_factory(mock_llm_instance):
    factory = MagicMock(spec=LLMFactory)
    factory.create_llm = AsyncMock(return_value=mock_llm_instance)
    return factory

@pytest.fixture
def mock_model_registry(mock_embedding_instance):
    registry = MagicMock(spec=ModelRegistry)
    # Configure create_embedding to return our async mock embedding instance
    registry.create_embedding = AsyncMock(return_value=mock_embedding_instance)
    return registry


class TestModelFacade:

    @pytest.mark.asyncio
    async def test_create_with_llm_only(self, mock_llm_factory, mock_model_registry):
        """Test ModelFacade.create instantiates LLM, embedding might be None."""
        facade = await ModelFacade.create(
            provider="mock_provider",
            model_name="mock_llm",
            config=ModelConfig(provider="mock_provider"), # Pass a config with provider
            llm_factory=mock_llm_factory,
            model_registry=mock_model_registry
        )
        mock_llm_factory.create_llm.assert_called_once()
        # Embedding might be None if provider is not 'openai' and no embedding_model_name
        if facade._embedding is not None:
            mock_model_registry.create_embedding.assert_called_once()

        assert facade._llm is not None
        assert facade._llm.model == "mocked-llm" # From mock_llm_instance

    @pytest.mark.asyncio
    async def test_create_with_openai_provider_instantiates_default_embedding(self, mock_llm_factory, mock_model_registry, mock_llm_instance):
        """Test ModelFacade.create instantiates default OpenAI embedding for openai LLM."""
        # Ensure the mock_llm_instance created by factory has 'openai' provider in its config
        mock_llm_instance.config = ModelConfig(provider="openai", model_name="mocked-gpt")
        mock_llm_factory.create_llm.return_value = mock_llm_instance # Update factory to return this

        facade = await ModelFacade.create(
            provider="openai",
            model_name="mocked-gpt",
            config=ModelConfig(provider="openai"), # Crucial for provider detection
            llm_factory=mock_llm_factory,
            model_registry=mock_model_registry
        )
        mock_llm_factory.create_llm.assert_called_once()
        mock_model_registry.create_embedding.assert_called_once()
        assert mock_model_registry.create_embedding.call_args.kwargs['model_name'] == ModelFacade.DEFAULT_OPENAI_EMBEDDING_MODEL
        assert facade._embedding is not None
        assert facade._embedding.model == "mocked-embedding"

    @pytest.mark.asyncio
    async def test_create_with_explicit_embedding_model(self, mock_llm_factory, mock_model_registry):
        """Test ModelFacade.create instantiates specified embedding model."""
        explicit_embed_model = "custom/my-embedding-model"
        facade = await ModelFacade.create(
            provider="any_provider",
            model_name="any_llm",
            llm_factory=mock_llm_factory,
            model_registry=mock_model_registry,
            embedding_model_name=explicit_embed_model
        )
        mock_model_registry.create_embedding.assert_called_once()
        assert mock_model_registry.create_embedding.call_args.kwargs['model_name'] == explicit_embed_model
        assert facade._embedding is not None

    @pytest.mark.asyncio
    async def test_create_embedding_instantiation_failure(self, mock_llm_factory, mock_model_registry, caplog):
        """Test warning is logged if embedding model instantiation fails."""
        mock_model_registry.create_embedding.side_effect = Exception("Embedding init failed")

        facade = await ModelFacade.create(
            provider="openai", # To trigger embedding creation attempt
            model_name="any_llm",
            llm_factory=mock_llm_factory,
            model_registry=mock_model_registry
        )
        assert facade._embedding is None
        assert "Could not instantiate embedding model" in caplog.text
        assert "Embedding init failed" in caplog.text

    @pytest.mark.asyncio
    async def test_generate_delegates_to_llm(self, mock_llm_instance):
        facade = ModelFacade(llm=mock_llm_instance) # Use direct constructor for this test
        prompt = "Test prompt"
        kwargs = {"temperature": 0.5}
        await facade.generate(prompt, **kwargs)
        mock_llm_instance.generate.assert_called_once_with(prompt=prompt, **kwargs)

    @pytest.mark.asyncio
    async def test_generate_with_tools_delegates_to_llm(self, mock_llm_instance):
        facade = ModelFacade(llm=mock_llm_instance)
        prompt = "Tool prompt"
        tools = [{"type": "function", "function": {"name": "test"}}]
        kwargs = {"tool_choice": "auto"}
        await facade.generate_with_tools(prompt, tools, **kwargs)
        mock_llm_instance.generate_with_tools.assert_called_once_with(prompt=prompt, tools=tools, **kwargs)

    @pytest.mark.asyncio
    async def test_embed_delegates_to_embedding_model(self, mock_llm_instance, mock_embedding_instance):
        facade = ModelFacade(llm=mock_llm_instance, embedding=mock_embedding_instance)
        texts = ["text1", "text2"]
        kwargs = {"some_embed_param": True}
        await facade.embed(texts, **kwargs)
        mock_embedding_instance.embed.assert_called_once_with(texts=texts, **kwargs)

    @pytest.mark.asyncio
    async def test_embed_raises_if_no_embedding_model(self, mock_llm_instance):
        facade = ModelFacade(llm=mock_llm_instance, embedding=None)
        with pytest.raises(RuntimeError, match="Embedding client not configured"):
            await facade.embed(["text"])

    @pytest.mark.asyncio
    async def test_rerank_raises_if_no_reranker_model(self, mock_llm_instance):
        facade = ModelFacade(llm=mock_llm_instance, reranker=None)
        with pytest.raises(RuntimeError, match="Reranker client not configured"):
            await facade.rerank("query", ["doc1"])


    @pytest.mark.asyncio
    async def test_close_calls_close_on_clients(self, mock_llm_instance, mock_embedding_instance):
        # Mock a reranker instance as well for complete test
        mock_reranker_instance = AsyncMock(spec=BaseReranker)
        mock_reranker_instance.close = AsyncMock()

        facade = ModelFacade(llm=mock_llm_instance, embedding=mock_embedding_instance, reranker=mock_reranker_instance)
        await facade.close()

        mock_llm_instance.close.assert_called_once()
        mock_embedding_instance.close.assert_called_once()
        mock_reranker_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_llm_instance, mock_embedding_instance):
        facade_instance = ModelFacade(llm=mock_llm_instance, embedding=mock_embedding_instance)
        with patch.object(facade_instance, 'close', new_callable=AsyncMock) as mock_close:
            async with facade_instance as facade:
                assert facade is facade_instance
                # Test some operation
                await facade.generate("hello")
                mock_llm_instance.generate.assert_called_once_with(prompt="hello")
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_properties_access_clients(self, mock_llm_instance, mock_embedding_instance):
        facade = ModelFacade(llm=mock_llm_instance, embedding=mock_embedding_instance)
        assert facade.llm is mock_llm_instance
        assert facade.embedding_model is mock_embedding_instance
        with pytest.raises(RuntimeError, match="Reranker client not configured"):
            _ = facade.reranker_model # Access property

        facade_no_embed = ModelFacade(llm=mock_llm_instance)
        with pytest.raises(RuntimeError, match="Embedding client not configured"):
            _ = facade_no_embed.embedding_model

    # Test case for create when LLM instantiation fails
    @pytest.mark.asyncio
    async def test_create_llm_instantiation_failure(self, mock_llm_factory, mock_model_registry):
        mock_llm_factory.create_llm.side_effect = Exception("LLM init failed")

        with pytest.raises(Exception, match="LLM init failed"):
            await ModelFacade.create(
                provider="failing_provider",
                model_name="failing_llm",
                llm_factory=mock_llm_factory,
                model_registry=mock_model_registry
            )
        mock_model_registry.create_embedding.assert_not_called() # Embedding should not be attempted

    # Test case for when embedding_model_name is given but provider is not openai
    @pytest.mark.asyncio
    async def test_create_with_explicit_embedding_non_openai_provider(self, mock_llm_factory, mock_model_registry, mock_llm_instance):
        mock_llm_instance.config = ModelConfig(provider="custom_provider", model_name="custom_llm")
        mock_llm_factory.create_llm.return_value = mock_llm_instance

        explicit_embed_model = "custom/my-embedding-model"
        facade = await ModelFacade.create(
            provider="custom_provider",
            model_name="custom_llm",
            llm_factory=mock_llm_factory,
            model_registry=mock_model_registry,
            embedding_model_name=explicit_embed_model
        )
        mock_model_registry.create_embedding.assert_called_once()
        assert mock_model_registry.create_embedding.call_args.kwargs['model_name'] == explicit_embed_model
        assert facade._embedding is not None

    # Test case for when provider is not OpenAI and no explicit embedding model is given
    @pytest.mark.asyncio
    async def test_create_no_default_embedding_for_non_openai_provider(self, mock_llm_factory, mock_model_registry, mock_llm_instance):
        mock_llm_instance.config = ModelConfig(provider="custom_provider", model_name="custom_llm")
        mock_llm_factory.create_llm.return_value = mock_llm_instance

        facade = await ModelFacade.create(
            provider="custom_provider",
            model_name="custom_llm",
            llm_factory=mock_llm_factory,
            model_registry=mock_model_registry
            # No embedding_model_name
        )
        mock_model_registry.create_embedding.assert_not_called()
        assert facade._embedding is None
