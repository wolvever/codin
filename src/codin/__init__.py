"""Codin Platform core package.

This package exposes high-level helpers for building agents, tools and runtimes
that interoperate via the A2A protocol.
"""

from .config import get_config, load_config, get_api_key

__all__: list[str] = [
    "version",
    "get_config",
    "load_config", 
    "get_api_key",
]

version: str = "0.1.0" 