"""Core tool definitions built on top of sandbox methods."""

from __future__ import annotations

import logging

# pydantic and requests related imports are no longer needed if FetchTool is the only user
# from bs4 import BeautifulSoup # No longer needed if FetchTool is removed
from typing import Any

import httpx
import pydantic as _pyd

from .base import Tool, ToolContext

__all__ = ["FetchTool"]

logger = logging.getLogger(__name__)


# Simple HTTP fetch tool used in tests


class FetchInput(_pyd.BaseModel):
    url: str
    max_length: int = 1000
    raw: bool = False
    start_index: int = 0


class FetchTool(Tool):
    """Fetch the content of a URL for tests."""

    def __init__(self) -> None:
        super().__init__("fetch", "Fetch URL content", input_schema=FetchInput)

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any:
        params = self.validate_input(args)
        async with httpx.AsyncClient() as client:
            response = await client.get(params["url"])
        content = response.text
        if not params["raw"]:
            start = params["start_index"]
            content = content[start : start + params["max_length"]]
        return {"url": params["url"], "status_code": response.status_code, "content": content}


# If there are other tools in this file, they would remain here.
# Based on the provided content, FetchTool was the only one.
# If this file becomes empty of tools, it might be a candidate for deletion
# or refactoring, but the task is only to remove FetchTool.
