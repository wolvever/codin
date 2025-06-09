import unittest
from src.codin.model.config import ModelConfig

class TestModelConfig(unittest.TestCase):

    def test_default_initialization(self):
        """Test ModelConfig initializes with default (None) values."""
        config = ModelConfig()
        self.assertIsNone(config.model_name)
        self.assertIsNone(config.api_key)
        self.assertIsNone(config.base_url)
        self.assertIsNone(config.api_version)
        self.assertIsNone(config.timeout)
        self.assertIsNone(config.connect_timeout)
        self.assertIsNone(config.max_retries)
        self.assertIsNone(config.retry_min_wait)
        self.assertIsNone(config.retry_max_wait)
        self.assertIsNone(config.retry_on_status_codes)
        self.assertIsNone(config.provider)

    def test_initialization_with_values(self):
        """Test ModelConfig initializes with provided values."""
        config = ModelConfig(
            model_name="test_model",
            api_key="test_key",
            base_url="http://localhost:1234",
            api_version="v1",
            timeout=60.0,
            connect_timeout=10.0,
            max_retries=5,
            retry_min_wait=1.0,
            retry_max_wait=30.0,
            retry_on_status_codes=[500, 503],
            provider="test_provider"
        )
        self.assertEqual(config.model_name, "test_model")
        self.assertEqual(config.api_key, "test_key")
        self.assertEqual(config.base_url, "http://localhost:1234")
        self.assertEqual(config.api_version, "v1")
        self.assertEqual(config.timeout, 60.0)
        self.assertEqual(config.connect_timeout, 10.0)
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.retry_min_wait, 1.0)
        self.assertEqual(config.retry_max_wait, 30.0)
        self.assertEqual(config.retry_on_status_codes, [500, 503])
        self.assertEqual(config.provider, "test_provider")

    def test_get_client_config_kwargs_empty(self):
        """Test get_client_config_kwargs returns empty dict for default config."""
        config = ModelConfig()
        kwargs = config.get_client_config_kwargs()
        self.assertEqual(kwargs, {})

    def test_get_client_config_kwargs_partial(self):
        """Test get_client_config_kwargs with some values set."""
        config = ModelConfig(
            base_url="http://partial:5678",
            timeout=45.0,
            max_retries=2
        )
        kwargs = config.get_client_config_kwargs()
        expected_kwargs = {
            "base_url": "http://partial:5678",
            "timeout": 45.0,
            "max_retries": 2
        }
        self.assertEqual(kwargs, expected_kwargs)

    def test_get_client_config_kwargs_all(self):
        """Test get_client_config_kwargs with all relevant values set."""
        config = ModelConfig(
            model_name="should_not_be_in_kwargs", # This field is not for ClientConfig
            api_key="should_not_be_in_kwargs",    # This field is not for ClientConfig
            base_url="http://all_values:8000",
            api_version="should_not_be_in_kwargs",
            timeout=120.0,
            connect_timeout=20.0,
            max_retries=3,
            retry_min_wait=0.5,
            retry_max_wait=60.0,
            retry_on_status_codes=[429, 502, 504],
            provider="should_not_be_in_kwargs"
        )
        kwargs = config.get_client_config_kwargs()
        expected_kwargs = {
            "base_url": "http://all_values:8000",
            "timeout": 120.0,
            "connect_timeout": 20.0,
            "max_retries": 3,
            "retry_min_wait": 0.5,
            "retry_max_wait": 60.0,
            "retry_on_status_codes": [429, 502, 504]
        }
        self.assertEqual(kwargs, expected_kwargs)

if __name__ == '__main__':
    unittest.main()
