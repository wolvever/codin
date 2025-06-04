"""Test cases for tool argument formatting in REPL."""

import pytest
from unittest.mock import Mock, patch
from codin.cli.repl import ReplSession
from codin.config import CodinConfig


class TestToolFormatting:
    """Test tool argument formatting functionality."""
    
    @pytest.fixture
    def repl_session(self):
        """Create a test REPL session."""
        config = CodinConfig()
        return ReplSession(verbose=False, debug=False)
    
    @patch('click.echo')
    def test_format_file_arguments(self, mock_echo, repl_session):
        """Test formatting of file-related arguments."""
        repl_session._format_tool_arguments("test_tool", {
            "path": "/home/user/file.txt",
            "content": "Hello world"
        })
        
        # Verify the calls
        assert mock_echo.call_count == 2
        calls = [call.args[0] for call in mock_echo.call_args_list]
        
        # Check that path is displayed with file icon
        assert any("📁 path: /home/user/file.txt" in call for call in calls)
        # Check that content is displayed with text icon
        assert any("📝 content: Hello world" in call for call in calls)
    
    @patch('click.echo')
    def test_format_command_arguments(self, mock_echo, repl_session):
        """Test formatting of command-related arguments."""
        repl_session._format_tool_arguments("exec_tool", {
            "command": ["ls", "-la", "/tmp"],
            "timeout": 30
        })
        
        calls = [call.args[0] for call in mock_echo.call_args_list]
        
        # Check command formatting
        assert any("🏃 command: [ls, ...2 more]" in call for call in calls)
        # Check additional args
        assert any("📋 Additional args: timeout=30" in call for call in calls)
    
    @patch('click.echo')
    def test_format_empty_arguments(self, mock_echo, repl_session):
        """Test formatting when no arguments are provided."""
        repl_session._format_tool_arguments("test_tool", {})
        
        # Should not echo anything for empty arguments
        assert mock_echo.call_count == 0
    
    def test_format_argument_value_truncation(self, repl_session):
        """Test that argument values can be truncated when max_length is specified."""
        long_text = "a" * 100
        
        # Test with max_length specified
        result = repl_session._format_argument_value("content", long_text, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")
        
        # Test with max_length=None (no truncation)
        result = repl_session._format_argument_value("content", long_text, max_length=None)
        assert len(result) == 100
        assert result == long_text
        assert not result.endswith("...")
    
    def test_format_argument_value_types(self, repl_session):
        """Test formatting of different argument value types."""
        # Test boolean
        assert repl_session._format_argument_value("flag", True) == "✓"
        assert repl_session._format_argument_value("flag", False) == "✗"
        
        # Test numbers
        assert repl_session._format_argument_value("count", 42) == "42"
        assert repl_session._format_argument_value("rate", 3.14) == "3.14"
        
        # Test list
        assert repl_session._format_argument_value("items", []) == "[]"
        assert repl_session._format_argument_value("items", ["one"]) == "[one]"
        assert repl_session._format_argument_value("items", ["one", "two", "three"]) == "[one, ...2 more]"
        
        # Test dict
        assert repl_session._format_argument_value("config", {}) == "{}"
        assert repl_session._format_argument_value("config", {"key": "value"}) == "{key}"
    
    def test_get_argument_icon(self, repl_session):
        """Test that appropriate icons are returned for argument keys."""
        # File/path icons
        assert repl_session._get_argument_icon("path") == "📁"
        assert repl_session._get_argument_icon("file_path") == "📁"
        assert repl_session._get_argument_icon("directory") == "📁"
        
        # Command icons
        assert repl_session._get_argument_icon("command") == "🏃"
        assert repl_session._get_argument_icon("cmd") == "🏃"
        assert repl_session._get_argument_icon("script") == "🏃"
        
        # Search icons
        assert repl_session._get_argument_icon("query") == "🔍"
        assert repl_session._get_argument_icon("search_term") == "🔍"
        
        # Content icons
        assert repl_session._get_argument_icon("content") == "📝"
        assert repl_session._get_argument_icon("text") == "📝"
        
        # Network icons
        assert repl_session._get_argument_icon("url") == "🌐"
        assert repl_session._get_argument_icon("endpoint") == "🌐"
        
        # ID icons
        assert repl_session._get_argument_icon("id") == "🏷️"
        assert repl_session._get_argument_icon("name") == "🏷️"
        
        # Default icon
        assert repl_session._get_argument_icon("unknown_arg") == "⚙️"
    
    @patch('click.echo')
    def test_format_full_arguments_no_truncation(self, mock_echo, repl_session):
        """Test that tool arguments are displayed in full without truncation."""
        long_explanation = "This is a very long explanation that would normally be truncated but should now be shown in full without any truncation"
        
        repl_session._format_tool_arguments("test_tool", {
            "path": "/home/user/very/long/path/to/some/file/that/is/quite/lengthy.txt",
            "explanation": long_explanation,
            "timeout": 300
        })
        
        calls = [call.args[0] for call in mock_echo.call_args_list]
        
        # Check that long path is displayed in full
        assert any("/home/user/very/long/path/to/some/file/that/is/quite/lengthy.txt" in call for call in calls)
        
        # Check that long explanation is displayed in full in additional args
        assert any(long_explanation in call for call in calls)
        assert not any("..." in call for call in calls if "explanation=" in call) 