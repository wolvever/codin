"""Basic tests that don't depend on imports."""

import pytest


def test_basic_functionality():
    """Test that pytest itself is working."""
    assert True


@pytest.mark.parametrize("x,y,expected", [
    (1, 1, 2),
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_addition(x, y, expected):
    """Test basic addition to demonstrate parametrized tests."""
    assert x + y == expected 