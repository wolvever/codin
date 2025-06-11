from __future__ import annotations

import logging
import typing as _t

import pydantic as _pyd
import requests

from .base import Tool, ToolContext

__all__ = ['FetchTool', 'FetchInput']

logger = logging.getLogger(__name__)

class FetchInput(_pyd.BaseModel):
    url: str = _pyd.Field(..., description='The URL to fetch.')

class FetchTool(Tool):
    def __init__(self):
        super().__init__(
            name='fetch',
            description='Fetches the content of a URL.',
            input_schema=FetchInput,
        )

    async def run(self, args: FetchInput, tool_context: ToolContext) -> str:
        try:
            response = requests.get(args.url)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f'Error fetching URL {args.url}: {e}')
            return f'Error: Could not fetch URL. {e}'
