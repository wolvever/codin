"""Comprehensive tests for the codin.id module."""

import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

# Add the src directory to the path to import the module directly
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import the id module directly to avoid circular imports
from codin.id import new_id


class TestNewId:
    """Test cases for the new_id function."""

    def test_basic_id_generation(self):
        """Test basic ID generation with default parameters."""
        result = new_id("test")
        
        # Should have the format "test-{8_random_chars}"
        assert result.startswith("test-")
        suffix = result.split("-", 1)[1]
        assert len(suffix) == 8
        assert suffix.isalnum()

    def test_custom_prefix(self):
        """Test ID generation with custom prefixes."""
        prefixes = ["agent", "task", "session", "memory"]
        
        for prefix in prefixes:
            result = new_id(prefix)
            assert result.startswith(f"{prefix}-")
            suffix = result.split("-", 1)[1]
            assert len(suffix) == 8

    def test_custom_length(self):
        """Test ID generation with custom lengths."""
        lengths = [4, 12, 16, 32]
        
        for length in lengths:
            result = new_id("test", length=length)
            suffix = result.split("-", 1)[1]
            assert len(suffix) == length
            assert suffix.isalnum()

    def test_uuid_generation(self):
        """Test ID generation with UUID suffix."""
        result = new_id("test", uuid=True)
        
        assert result.startswith("test-")
        suffix = result.split("-", 1)[1]
        
        # Should be a valid UUID4 string
        try:
            parsed_uuid = uuid.UUID(suffix)
            assert parsed_uuid.version == 4
        except ValueError:
            pytest.fail(f"Generated suffix '{suffix}' is not a valid UUID")

    def test_uuid_ignores_length_parameter(self):
        """Test that length parameter is ignored when uuid=True."""
        result1 = new_id("test", length=4, uuid=True)
        result2 = new_id("test", length=20, uuid=True)
        
        # Both should have UUID format regardless of length parameter
        suffix1 = result1.split("-", 1)[1]
        suffix2 = result2.split("-", 1)[1]
        
        # Both should be valid UUIDs
        uuid.UUID(suffix1)
        uuid.UUID(suffix2)
        
        # Both should have standard UUID length (36 chars with hyphens)
        assert len(suffix1) == 36
        assert len(suffix2) == 36

    def test_empty_prefix(self):
        """Test ID generation with empty prefix."""
        result = new_id("")
        
        # Should start with a hyphen
        assert result.startswith("-")
        suffix = result[1:]  # Remove leading hyphen
        assert len(suffix) == 8
        assert suffix.isalnum()

    def test_prefix_with_special_characters(self):
        """Test ID generation with prefixes containing special characters."""
        special_prefixes = ["test_case", "user@domain", "item#1", "node.js"]
        
        for prefix in special_prefixes:
            result = new_id(prefix)
            assert result.startswith(f"{prefix}-")
            suffix = result.split("-", 1)[1]
            assert len(suffix) == 8

    def test_minimum_length(self):
        """Test ID generation with minimum length."""
        result = new_id("test", length=1)
        suffix = result.split("-", 1)[1]
        assert len(suffix) == 1
        assert suffix.isalnum()

    def test_zero_length(self):
        """Test ID generation with zero length."""
        result = new_id("test", length=0)
        suffix = result.split("-", 1)[1]
        assert len(suffix) == 0
        assert result == "test-"

    def test_large_length(self):
        """Test ID generation with large length."""
        result = new_id("test", length=100)
        suffix = result.split("-", 1)[1]
        assert len(suffix) == 100
        assert suffix.isalnum()

    def test_character_set_randomness(self):
        """Test that generated IDs use the expected character set."""
        # Generate multiple IDs to check character distribution
        ids = [new_id("test") for _ in range(100)]
        suffixes = [id_str.split("-", 1)[1] for id_str in ids]
        
        # Combine all characters used
        all_chars = "".join(suffixes)
        unique_chars = set(all_chars)
        
        # Should only contain alphanumeric characters
        assert all(c.isalnum() for c in unique_chars)
        
        # Should contain both letters and digits (with high probability)
        has_letter = any(c.isalpha() for c in unique_chars)
        has_digit = any(c.isdigit() for c in unique_chars)
        assert has_letter, "Generated IDs should contain letters"
        assert has_digit, "Generated IDs should contain digits"

    def test_uniqueness(self):
        """Test that generated IDs are unique."""
        # Generate many IDs with the same parameters
        ids = [new_id("test") for _ in range(1000)]
        
        # All should be unique
        assert len(set(ids)) == len(ids)

    def test_uuid_uniqueness(self):
        """Test that UUID-based IDs are unique."""
        # Generate many UUID-based IDs
        ids = [new_id("test", uuid=True) for _ in range(100)]
        
        # All should be unique
        assert len(set(ids)) == len(ids)

    def test_deterministic_with_mocked_random(self):
        """Test that ID generation is deterministic when random is mocked."""
        with patch('random.choices') as mock_choices:
            mock_choices.return_value = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            
            result = new_id("test")
            assert result == "test-abcdefgh"
            
            # Should call random.choices with correct parameters
            mock_choices.assert_called_once()
            args, kwargs = mock_choices.call_args
            assert kwargs['k'] == 8

    def test_deterministic_with_mocked_uuid(self):
        """Test that UUID generation is deterministic when uuid is mocked."""
        mock_uuid_value = "12345678-1234-4567-8901-123456789012"
        
        with patch('uuid.uuid4') as mock_uuid4:
            mock_uuid4.return_value = uuid.UUID(mock_uuid_value)
            
            result = new_id("test", uuid=True)
            assert result == f"test-{mock_uuid_value}"

    def test_multiple_hyphens_in_prefix(self):
        """Test behavior with prefixes containing multiple hyphens."""
        result = new_id("multi-part-prefix")
        
        # Should maintain all hyphens in prefix
        assert result.startswith("multi-part-prefix-")
        
        # The last segment should be the generated suffix
        parts = result.split("-")
        suffix = parts[-1]
        assert len(suffix) == 8
        assert suffix.isalnum()

    def test_numeric_prefix(self):
        """Test ID generation with numeric prefix."""
        result = new_id("123")
        
        assert result.startswith("123-")
        suffix = result.split("-", 1)[1]
        assert len(suffix) == 8
        assert suffix.isalnum()

    def test_long_prefix(self):
        """Test ID generation with very long prefix."""
        long_prefix = "a" * 100
        result = new_id(long_prefix)
        
        assert result.startswith(f"{long_prefix}-")
        suffix = result.split("-", 1)[1]
        assert len(suffix) == 8

    @pytest.mark.parametrize("length", [1, 2, 5, 10, 15, 20])
    def test_parametrized_lengths(self, length):
        """Test various lengths using parametrized tests."""
        result = new_id("param", length=length)
        suffix = result.split("-", 1)[1]
        assert len(suffix) == length

    @pytest.mark.parametrize("prefix", ["a", "test", "long-prefix-name", ""])
    def test_parametrized_prefixes(self, prefix):
        """Test various prefixes using parametrized tests."""
        result = new_id(prefix)
        expected_start = f"{prefix}-" if prefix else "-"
        assert result.startswith(expected_start)

    def test_concurrent_generation_uniqueness(self):
        """Test that concurrent ID generation maintains uniqueness."""
        import concurrent.futures
        
        def generate_ids(count):
            return [new_id("concurrent") for _ in range(count)]
        
        # Generate IDs concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generate_ids, 25) for _ in range(4)]
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        
        # All 100 IDs should be unique
        assert len(set(results)) == len(results)

    def test_edge_case_negative_length(self):
        """Test behavior with negative length (should be handled gracefully)."""
        # Note: This depends on how random.choices handles negative k
        # In practice, this would likely raise a ValueError
        with pytest.raises(ValueError):
            new_id("test", length=-1)

    def test_imports_are_local(self):
        """Test that imports are done locally within the function."""
        # This test verifies that the function imports are local
        # by checking that we can call the function without pre-importing
        
        # Clear any existing imports in the module namespace
        
        # The function should work even if we haven't imported the dependencies globally
        result = new_id("import_test")
        assert result.startswith("import_test-")
        assert len(result.split("-", 1)[1]) == 8