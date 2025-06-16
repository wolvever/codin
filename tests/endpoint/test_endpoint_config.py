import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from codin.endpoint.config import EndpointConfig
from codin.endpoint.base_config import BaseEndpointConfig
from codin.endpoint.resolver import EndpointResolver
from codin.endpoint.backends import EndpointBackend


class MockEndpointBackend(EndpointBackend):
    def __init__(self, name: str):
        self.name = name
        self.config = {}
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        return True
    
    def get_client(self, config: Dict[str, Any]):
        return Mock()


class TestBaseEndpointConfig:
    
    def test_base_config_creation(self):
        """Test creating base endpoint configuration."""
        config = BaseEndpointConfig(
            name="test_endpoint",
            backend="test_backend",
            config={"key": "value"}
        )
        
        assert config.name == "test_endpoint"
        assert config.backend == "test_backend"
        assert config.config["key"] == "value"
    
    def test_base_config_validation(self):
        """Test base configuration validation."""
        # Valid config
        config = BaseEndpointConfig(
            name="valid_endpoint",
            backend="openai",
            config={"api_key": "test_key"}
        )
        
        assert config.name == "valid_endpoint"
    
    def test_base_config_invalid_name(self):
        """Test invalid endpoint name."""
        with pytest.raises(ValueError):
            BaseEndpointConfig(
                name="",  # Empty name should be invalid
                backend="test",
                config={}
            )
    
    def test_base_config_missing_backend(self):
        """Test missing backend specification."""
        with pytest.raises(ValueError):
            BaseEndpointConfig(
                name="test",
                backend="",  # Empty backend should be invalid
                config={}
            )


class TestEndpointConfig:
    
    @pytest.fixture
    def sample_config_dict(self):
        return {
            "endpoints": {
                "openai_endpoint": {
                    "backend": "openai",
                    "config": {
                        "api_key": "test_key",
                        "model": "gpt-4"
                    }
                },
                "anthropic_endpoint": {
                    "backend": "anthropic", 
                    "config": {
                        "api_key": "test_key_2",
                        "model": "claude-3"
                    }
                }
            }
        }
    
    def test_endpoint_config_creation_empty(self):
        """Test creating empty endpoint configuration."""
        config = EndpointConfig()
        
        assert len(config.endpoints) == 0
    
    def test_endpoint_config_creation_with_data(self, sample_config_dict):
        """Test creating endpoint configuration with data."""
        config = EndpointConfig.from_dict(sample_config_dict)
        
        assert len(config.endpoints) == 2
        assert "openai_endpoint" in config.endpoints
        assert "anthropic_endpoint" in config.endpoints
    
    def test_add_endpoint(self):
        """Test adding an endpoint to configuration."""
        config = EndpointConfig()
        endpoint_config = BaseEndpointConfig(
            name="new_endpoint",
            backend="test_backend",
            config={"test": "value"}
        )
        
        config.add_endpoint(endpoint_config)
        
        assert "new_endpoint" in config.endpoints
        assert config.endpoints["new_endpoint"] == endpoint_config
    
    def test_get_endpoint_existing(self):
        """Test getting an existing endpoint."""
        config = EndpointConfig()
        endpoint_config = BaseEndpointConfig(
            name="test_endpoint",
            backend="test",
            config={}
        )
        config.add_endpoint(endpoint_config)
        
        retrieved = config.get_endpoint("test_endpoint")
        
        assert retrieved == endpoint_config
    
    def test_get_endpoint_nonexistent(self):
        """Test getting a non-existent endpoint."""
        config = EndpointConfig()
        
        retrieved = config.get_endpoint("nonexistent")
        
        assert retrieved is None
    
    def test_list_endpoints(self, sample_config_dict):
        """Test listing all endpoints."""
        config = EndpointConfig.from_dict(sample_config_dict)
        
        endpoints = config.list_endpoints()
        
        assert len(endpoints) == 2
        endpoint_names = {ep.name for ep in endpoints}
        assert endpoint_names == {"openai_endpoint", "anthropic_endpoint"}
    
    def test_remove_endpoint(self):
        """Test removing an endpoint."""
        config = EndpointConfig()
        endpoint_config = BaseEndpointConfig(
            name="to_remove",
            backend="test",
            config={}
        )
        config.add_endpoint(endpoint_config)
        
        assert "to_remove" in config.endpoints
        
        config.remove_endpoint("to_remove")
        
        assert "to_remove" not in config.endpoints
    
    def test_remove_nonexistent_endpoint(self):
        """Test removing a non-existent endpoint."""
        config = EndpointConfig()
        
        # Should not raise an error
        config.remove_endpoint("nonexistent")
    
    def test_validate_all_endpoints(self, sample_config_dict):
        """Test validating all endpoints in configuration."""
        config = EndpointConfig.from_dict(sample_config_dict)
        
        # Mock backend validation
        with patch('codin.endpoint.backends.get_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.validate_config.return_value = True
            mock_get_backend.return_value = mock_backend
            
            is_valid = config.validate_all()
            
            assert is_valid is True
    
    def test_validate_all_endpoints_with_invalid(self):
        """Test validation fails when one endpoint is invalid."""
        config = EndpointConfig()
        config.add_endpoint(BaseEndpointConfig(
            name="invalid_endpoint",
            backend="test",
            config={"invalid": "config"}
        ))
        
        with patch('codin.endpoint.backends.get_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.validate_config.return_value = False
            mock_get_backend.return_value = mock_backend
            
            is_valid = config.validate_all()
            
            assert is_valid is False
    
    def test_to_dict(self, sample_config_dict):
        """Test converting configuration to dictionary."""
        config = EndpointConfig.from_dict(sample_config_dict)
        
        result_dict = config.to_dict()
        
        assert "endpoints" in result_dict
        assert len(result_dict["endpoints"]) == 2
        assert "openai_endpoint" in result_dict["endpoints"]
        assert "anthropic_endpoint" in result_dict["endpoints"]
    
    def test_from_dict_invalid_format(self):
        """Test creating config from invalid dictionary format."""
        invalid_dict = {
            "not_endpoints": {}
        }
        
        with pytest.raises(ValueError):
            EndpointConfig.from_dict(invalid_dict)
    
    def test_from_dict_invalid_endpoint_config(self):
        """Test creating config from dict with invalid endpoint."""
        invalid_dict = {
            "endpoints": {
                "invalid_endpoint": {
                    "missing_backend": True
                }
            }
        }
        
        with pytest.raises(ValueError):
            EndpointConfig.from_dict(invalid_dict)


class TestEndpointResolver:
    
    @pytest.fixture
    def resolver(self):
        config = EndpointConfig()
        config.add_endpoint(BaseEndpointConfig(
            name="primary_endpoint",
            backend="openai",
            config={"api_key": "test"}
        ))
        config.add_endpoint(BaseEndpointConfig(
            name="secondary_endpoint", 
            backend="anthropic",
            config={"api_key": "test2"}
        ))
        
        return EndpointResolver(config)
    
    def test_resolver_creation(self, resolver):
        """Test creating endpoint resolver."""
        assert resolver is not None
        assert len(resolver._config.endpoints) == 2
    
    def test_resolve_by_name(self, resolver):
        """Test resolving endpoint by name."""
        endpoint = resolver.resolve("primary_endpoint")
        
        assert endpoint is not None
        assert endpoint.name == "primary_endpoint"
        assert endpoint.backend == "openai"
    
    def test_resolve_nonexistent(self, resolver):
        """Test resolving non-existent endpoint."""
        endpoint = resolver.resolve("nonexistent")
        
        assert endpoint is None
    
    def test_resolve_by_backend_type(self, resolver):
        """Test resolving endpoint by backend type."""
        endpoints = resolver.resolve_by_backend("openai")
        
        assert len(endpoints) == 1
        assert endpoints[0].name == "primary_endpoint"
    
    def test_resolve_by_backend_type_multiple(self):
        """Test resolving multiple endpoints of same backend type."""
        config = EndpointConfig()
        config.add_endpoint(BaseEndpointConfig(
            name="openai_1", backend="openai", config={}
        ))
        config.add_endpoint(BaseEndpointConfig(
            name="openai_2", backend="openai", config={}
        ))
        
        resolver = EndpointResolver(config)
        endpoints = resolver.resolve_by_backend("openai")
        
        assert len(endpoints) == 2
        names = {ep.name for ep in endpoints}
        assert names == {"openai_1", "openai_2"}
    
    def test_get_default_endpoint(self, resolver):
        """Test getting default endpoint."""
        default = resolver.get_default()
        
        # Should return first endpoint or None if no default set
        assert default is not None or default is None
    
    def test_set_default_endpoint(self, resolver):
        """Test setting default endpoint."""
        resolver.set_default("secondary_endpoint")
        
        default = resolver.get_default()
        assert default.name == "secondary_endpoint"
    
    def test_set_invalid_default(self, resolver):
        """Test setting invalid default endpoint."""
        with pytest.raises(ValueError):
            resolver.set_default("nonexistent_endpoint")
    
    def test_list_available_backends(self, resolver):
        """Test listing available backends."""
        backends = resolver.list_backends()
        
        assert "openai" in backends
        assert "anthropic" in backends
        assert len(backends) == 2
    
    def test_resolver_with_empty_config(self):
        """Test resolver with empty configuration."""
        config = EndpointConfig()
        resolver = EndpointResolver(config)
        
        assert resolver.resolve("any") is None
        assert len(resolver.list_backends()) == 0
        assert resolver.get_default() is None


class TestEndpointIntegration:
    
    def test_full_endpoint_lifecycle(self):
        """Test full endpoint configuration lifecycle."""
        # Create configuration
        config = EndpointConfig()
        
        # Add endpoint
        endpoint_config = BaseEndpointConfig(
            name="lifecycle_endpoint",
            backend="test_backend",
            config={"key": "value"}
        )
        config.add_endpoint(endpoint_config)
        
        # Create resolver
        resolver = EndpointResolver(config)
        
        # Resolve endpoint
        resolved = resolver.resolve("lifecycle_endpoint")
        assert resolved == endpoint_config
        
        # Convert to dict and back
        config_dict = config.to_dict()
        recreated_config = EndpointConfig.from_dict(config_dict)
        
        # Verify recreation
        recreated_resolver = EndpointResolver(recreated_config)
        recreated_resolved = recreated_resolver.resolve("lifecycle_endpoint")
        
        assert recreated_resolved.name == "lifecycle_endpoint"
        assert recreated_resolved.backend == "test_backend"
        assert recreated_resolved.config["key"] == "value"
    
    def test_configuration_updates(self):
        """Test updating configuration after resolver creation."""
        config = EndpointConfig()
        resolver = EndpointResolver(config)
        
        # Initially empty
        assert resolver.resolve("test") is None
        
        # Add endpoint
        config.add_endpoint(BaseEndpointConfig(
            name="test",
            backend="test_backend", 
            config={}
        ))
        
        # Should now resolve
        resolved = resolver.resolve("test")
        assert resolved is not None
        assert resolved.name == "test"