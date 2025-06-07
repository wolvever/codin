from __future__ import annotations

"""Tests for ApprovalMode enum."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.codin.config import ApprovalMode


class TestApprovalMode:
    """Test ApprovalMode enum functionality."""

    def test_approval_mode_values(self):
        """Test ApprovalMode enum values."""
        assert ApprovalMode.ALWAYS.value == "always"
        assert ApprovalMode.NEVER.value == "never"
        assert ApprovalMode.UNSAFE_ONLY.value == "unsafe_only"

    def test_approval_mode_comparison(self):
        """Test ApprovalMode enum comparison."""
        assert ApprovalMode.ALWAYS == ApprovalMode.ALWAYS
        assert ApprovalMode.ALWAYS != ApprovalMode.NEVER
        assert ApprovalMode.NEVER != ApprovalMode.UNSAFE_ONLY

    def test_approval_mode_string_representation(self):
        """Test ApprovalMode string representation."""
        assert str(ApprovalMode.ALWAYS) == "ApprovalMode.ALWAYS"
        assert str(ApprovalMode.NEVER) == "ApprovalMode.NEVER"
        assert str(ApprovalMode.UNSAFE_ONLY) == "ApprovalMode.UNSAFE_ONLY"

    def test_approval_mode_membership(self):
        """Test ApprovalMode membership."""
        modes = list(ApprovalMode)
        assert ApprovalMode.ALWAYS in modes
        assert ApprovalMode.NEVER in modes
        assert ApprovalMode.UNSAFE_ONLY in modes
        assert len(modes) == 3

    def test_approval_mode_from_string(self):
        """Test creating ApprovalMode from string values."""
        # Test valid string values
        always_from_str = ApprovalMode("always")
        never_from_str = ApprovalMode("never")
        unsafe_from_str = ApprovalMode("unsafe_only")

        assert always_from_str == ApprovalMode.ALWAYS
        assert never_from_str == ApprovalMode.NEVER
        assert unsafe_from_str == ApprovalMode.UNSAFE_ONLY

    def test_approval_mode_invalid_value(self):
        """Test ApprovalMode with invalid value."""
        with pytest.raises(ValueError):
            ApprovalMode("invalid_mode") 