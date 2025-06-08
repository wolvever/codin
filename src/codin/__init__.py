"""Codin Platform core package.

This package exposes high-level helpers for building agents, tools and runtimes
that interoperate via the A2A protocol.
"""

import sys as _sys

import codin.agent.types as _agent_types

from .config import get_api_key, get_config, load_config


def extract_text_from_message(*args, **kwargs):
    from .utils.message import extract_text_from_message as _f

    return _f(*args, **kwargs)


def format_history_for_prompt(*args, **kwargs):
    from .utils.message import format_history_for_prompt as _f

    return _f(*args, **kwargs)


def format_tool_results_for_conversation(*args, **kwargs):
    from .utils.message import format_tool_results_for_conversation as _f

    return _f(*args, **kwargs)

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

# Provide backward-compatible imports for `src.codin` paths used in tests

_sys.modules.setdefault('src', _sys.modules[__name__])
_sys.modules.setdefault('src.codin', _sys.modules[__name__])
_sys.modules.setdefault('src.codin.agent.types', _agent_types)
