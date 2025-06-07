"""Codin Platform core package.

This package exposes high-level helpers for building agents, tools and runtimes
that interoperate via the A2A protocol.
"""

from .config import get_api_key, get_config, load_config


__all__: list[str] = [
    'get_api_key',
    'get_config',
    'load_config',
    'version',
]

version: str = '0.1.0'
