"""MCP type conversion utilities for codin agents.

This module provides utilities for converting between MCP protocol types and
codin internal types for seamless integration.

Utilities to convert generic MCP JSON responses into *codin* protocol types are
currently implemented in a heuristic manner and therefore kept very small. Once
real-world MCP responses are available we can iteratively enhance these
mappings.
"""

from __future__ import annotations

import typing as _t

from a2a.types import DataPart, FilePart, TextPart

__all__ = [
    'convert_mcp_to_protocol_types',
]


def _is_text_part(data: dict[str, _t.Any]) -> bool:
    return data.get('type') == 'text' and 'text' in data


def _is_data_part(data: dict[str, _t.Any]) -> bool:
    return data.get('type') == 'data' and 'data' in data and isinstance(data['data'], dict)


def _is_file_part(data: dict[str, _t.Any]) -> bool:
    return data.get('type') == 'file' and 'file' in data and isinstance(data['file'], dict)


def _convert_single(obj: _t.Any) -> _t.Any:
    """Recursively convert *obj* into protocol types when recognised."""
    # Primitive types are returned unchanged.
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj

    # Lists are converted element-wise.
    if isinstance(obj, list):
        return [_convert_single(item) for item in obj]

    # Dictionaries may represent protocol parts.
    if isinstance(obj, dict):
        if _is_text_part(obj):
            return TextPart(text=_t.cast(str, obj['text']))
        if _is_data_part(obj):
            return DataPart(data=_t.cast(dict[str, _t.Any], obj['data']), mime_type='application/json')
        if _is_file_part(obj):
            file_dict = _t.cast(dict[str, _t.Any], obj['file'])
            return FilePart(
                file_id=file_dict.get('uri', file_dict.get('name', 'unknown')),
                mime_type=file_dict.get('mimeType'),
                name=file_dict.get('name'),
            )
        # Fallback – return a recursively converted dictionary.
        return {key: _convert_single(val) for key, val in obj.items()}

    # Any other type is left untouched.
    return obj


def convert_mcp_to_protocol_types(payload: dict[str, _t.Any]) -> _t.Any:
    """Convert *payload* recursively, returning protocol types when matched.

    The function is intentionally *very* permissive and only looks for keys
    that clearly identify one of the known A2A part variants.  Everything else
    is forwarded unchanged.

    Examples:
    --------
    >>> convert_mcp_to_protocol_types({'type': 'text', 'text': 'hello'})
    TextPart(text='hello', metadata=None)
    """
    return _convert_single(payload)
