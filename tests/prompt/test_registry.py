"""Tests for the prompt.registry module."""

import os
import pathlib
import pytest
import typing as _t
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from codin.prompt.registry import PromptRegistry
from codin.prompt.base import PromptTemplate


class TestPromptRegistry:
    """Test cases for the PromptRegistry class."""
    
    @pytest.fixture
    def clean_registry(self):
        """Provide a clean registry for tests."""
        # Save original instance
        original_instance = PromptRegistry._instance
        # Reset the instance
        PromptRegistry._instance = None
        
        yield PromptRegistry.get_instance()
        
        # Restore original instance
        PromptRegistry._instance = original_instance
    
    def test_singleton_pattern(self):
        """Test that PromptRegistry is a singleton."""
        instance1 = PromptRegistry.get_instance()
        instance2 = PromptRegistry.get_instance()
        
        assert instance1 is instance2
        
        # Direct instantiation should work (creates a new instance)
        direct_instance = PromptRegistry()
        assert direct_instance is not None
        assert isinstance(direct_instance, PromptRegistry)
    
    def test_register_and_get_template(self, clean_registry):
        """Test registering and retrieving a template."""
        # Create and register a template
        template = PromptTemplate(
            name="test",
            version="v1",
            text="Hello, {{ name }}!"
        )
        
        clean_registry.register(template)
        
        # Retrieve the template
        retrieved = clean_registry.get("test", "v1")
        
        assert retrieved is template
        assert retrieved.name == "test"
        assert retrieved.version == "v1"
        assert retrieved.text == "Hello, {{ name }}!"
    
    def test_get_latest_version(self, clean_registry):
        """Test getting the latest version of a template."""
        # Create and register multiple versions
        template1 = PromptTemplate(name="test", version="v1", text="Version 1")
        template2 = PromptTemplate(name="test", version="v2", text="Version 2")
        template3 = PromptTemplate(name="test", version="v3", text="Version 3")
        
        clean_registry.register(template1)
        clean_registry.register(template2)
        clean_registry.register(template3)
        
        # Retrieve without specifying version (should get latest)
        retrieved = clean_registry.get("test")
        
        assert retrieved is template3
        assert retrieved.version == "v3"
    
    def test_get_nonexistent_template(self, clean_registry):
        """Test getting a non-existent template."""
        with pytest.raises(KeyError):
            clean_registry.get("nonexistent")
            
        with pytest.raises(KeyError):
            clean_registry.get("test", "nonexistent")
    
    def test_list_templates(self, clean_registry):
        """Test listing templates."""
        # Create and register multiple templates
        template1 = PromptTemplate(name="test1", version="v1", text="Test 1")
        template2 = PromptTemplate(name="test2", version="v1", text="Test 2")
        template3 = PromptTemplate(name="test1", version="v2", text="Test 1 v2")
        
        clean_registry.register(template1)
        clean_registry.register(template2)
        clean_registry.register(template3)
        
        # List all templates
        all_templates = clean_registry.list()
        assert len(all_templates) == 3
        assert template1 in all_templates
        assert template2 in all_templates
        assert template3 in all_templates
        
        # List templates by name
        test1_templates = clean_registry.list("test1")
        assert len(test1_templates) == 2
        assert template1 in test1_templates
        assert template3 in test1_templates
        assert template2 not in test1_templates
    
    def test_prompt_decorator(self):
        """Test the @PromptRegistry.prompt decorator."""
        # Save original registry state
        original_instance = PromptRegistry._instance
        original_registry = {}
        if original_instance:
            original_registry = {k: v for k, v in original_instance._registry.items()}
        
        try:
            # Reset registry
            PromptRegistry._instance = None
            registry = PromptRegistry.get_instance()
            
            # Use the decorator
            @PromptRegistry.prompt("greeting", version="v1")
            def greeting_template():
                return "Hello, {{ name }}!"
            
            # Check that template was registered
            template = registry.get("greeting", "v1")
            assert template.name == "greeting"
            assert template.version == "v1"
            assert template.text == "Hello, {{ name }}!"
            
            # Render the template
            rendered = template.render(name="World")
            assert rendered.text == "Hello, World!"
            
        finally:
            # Restore original registry state
            PromptRegistry._instance = original_instance
            if original_instance:
                original_instance._registry = original_registry
    
    def test_set_and_get_run_mode(self, clean_registry):
        """Test setting and getting the run mode."""
        # Default should be local
        assert clean_registry.run_mode == "local"
        
        # Set to remote
        clean_registry.set_run_mode("remote")
        assert clean_registry.run_mode == "remote"
        
        # Invalid mode should raise ValueError
        with pytest.raises(ValueError):
            clean_registry.set_run_mode("invalid")
    
    def test_class_methods(self):
        """Test the class-level methods."""
        # Save original registry state
        original_instance = PromptRegistry._instance
        original_registry = {}
        if original_instance:
            original_registry = {k: v for k, v in original_instance._registry.items()}
        original_mode = original_instance.run_mode if original_instance else "local"
        
        try:
            # Reset registry
            PromptRegistry._instance = None
            
            # Create templates
            template1 = PromptTemplate(name="test1", version="v1", text="Test 1")
            template2 = PromptTemplate(name="test2", version="v1", text="Test 2")
            
            # Use class methods
            PromptRegistry.register_template(template1)
            PromptRegistry.register_template(template2)
            
            # Get templates
            assert PromptRegistry.get_template("test1") is template1
            assert PromptRegistry.get_template("test2") is template2
            
            # List templates
            all_templates = PromptRegistry.list_templates()
            assert len(all_templates) == 2
            assert template1 in all_templates
            assert template2 in all_templates
            
            # Set and get mode
            assert PromptRegistry.get_mode() == "local"
            PromptRegistry.set_mode("remote")
            assert PromptRegistry.get_mode() == "remote"
            
        finally:
            # Restore original registry state
            PromptRegistry._instance = original_instance
            if original_instance:
                original_instance._registry = original_registry
                original_instance.run_mode = original_mode
    
    @pytest.mark.asyncio
    @patch("builtins.open", new_callable=lambda: mock_open(read_data="name: greeting\nversion: latest\nvariants:\n  - text: 'Hello, {{ name }}!'"))
    @patch("pathlib.Path.exists")
    async def test_lazy_load_local(self, mock_exists, mock_file_open, clean_registry, monkeypatch):
        """Test lazy loading templates from local files."""
        # Set up environment
        monkeypatch.setenv("PROMPT_TEMPLATE_DIR", "/templates")
        monkeypatch.setenv("PROMPT_RUN_MODE", "local")
        
        # Create a new registry that will pick up the environment variable
        PromptRegistry._instance = None
        registry = PromptRegistry.get_instance()
        
        # Set up mocks
        mock_exists.return_value = True
        
        # Try to get a template that's not registered yet
        template = registry.get("greeting")
        
        # Check mock calls
        mock_exists.assert_called()
        mock_file_open.assert_called()
        
        # Check template
        assert template.name == "greeting"
        assert template.text == "Hello, {{ name }}!"
    
    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_lazy_load_remote(self, mock_client, clean_registry, monkeypatch):
        """Test lazy loading templates from remote."""
        # Set up environment
        monkeypatch.setenv("PROMPT_RUN_MODE", "remote")
        monkeypatch.setenv("PROMPT_REMOTE_BASE_URL", "https://api.example.com")
        
        # Create a new registry with HTTP endpoint
        PromptRegistry._instance = None
        registry = PromptRegistry("https://api.example.com/prompts")
        
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "greeting",
            "version": "v1",
            "variants": [{"text": "Hello, {{ name }}!"}]
        }
        mock_response.headers = {}  # No ETag
        
        # Set up mock client
        client_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = client_instance
        client_instance.get.return_value = mock_response
        
        # Try to get a template that's not registered yet
        template = registry.get("greeting")
        
        # Check template
        assert template.name == "greeting"
        assert template.version == "v1"
        assert template.text == "Hello, {{ name }}!" 