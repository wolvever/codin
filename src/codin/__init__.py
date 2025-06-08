"""Codin Platform core package.

This package exposes high-level helpers for building agents, tools and runtimes
that interoperate via the A2A protocol.
"""

from .config import get_api_key, get_config, load_config
from .utils.message import (
    extract_text_from_message,
    format_history_for_prompt,
    format_tool_results_for_conversation,
)

__all__: list[str] = [
    'get_api_key',
    'get_config',
    'load_config',
    'extract_text_from_message',
    'format_history_for_prompt',
    'format_tool_results_for_conversation',
    'version',
]

version: str = '0.1.0'
