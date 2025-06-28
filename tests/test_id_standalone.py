#!/usr/bin/env python3
"""Standalone test script for the id module."""

import sys
import uuid
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Load the id module directly
import importlib.util

spec = importlib.util.spec_from_file_location("id_module", src_path / "codin" / "id.py")
id_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(id_module)

new_id = id_module.new_id

def test_basic_functionality():
    """Test basic functionality of new_id."""
    print("Testing basic functionality...")
    
    # Test basic ID generation
    result = new_id("test")
    assert result.startswith("test-")
    suffix = result.split("-", 1)[1]
    assert len(suffix) == 8
    assert suffix.isalnum()
    print(f"✓ Basic ID: {result}")
    
    # Test custom length
    result = new_id("test", length=12)
    suffix = result.split("-", 1)[1]
    assert len(suffix) == 12
    print(f"✓ Custom length: {result}")
    
    # Test UUID generation
    result = new_id("test", uuid=True)
    assert result.startswith("test-")
    suffix = result.split("-", 1)[1]
    # Should be a valid UUID
    try:
        parsed_uuid = uuid.UUID(suffix)
        assert parsed_uuid.version == 4
        print(f"✓ UUID ID: {result}")
    except ValueError:
        raise AssertionError(f"Invalid UUID: {suffix}")
    
    # Test uniqueness
    ids = [new_id("unique") for _ in range(100)]
    assert len(set(ids)) == len(ids)
    print("✓ Uniqueness test passed")
    
    # Test edge cases
    result = new_id("", length=0)
    assert result == "-"
    print(f"✓ Edge case (empty prefix, zero length): {result}")
    
    result = new_id("test", length=1)
    suffix = result.split("-", 1)[1]
    assert len(suffix) == 1
    print(f"✓ Edge case (length=1): {result}")
    
    print("All tests passed! ✓")

if __name__ == "__main__":
    test_basic_functionality()