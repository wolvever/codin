"""Core tool definitions built on top of sandbox methods."""

from __future__ import annotations

import logging
import typing as _t

import pydantic as _pyd
import requests
from bs4 import BeautifulSoup

from .base import Tool, ToolContext

__all__ = ["FetchTool"]

logger = logging.getLogger(__name__)


class FetchInput(_pyd.BaseModel):
    """Input schema for fetching URLs."""

    url: str = _pyd.Field(..., description="URL to fetch")
    max_length: int | None = _pyd.Field(5000, description="Maximum number of characters to return")
    raw: bool | None = _pyd.Field(False, description="Get the actual HTML content without simplification")
    start_index: int | None = _pyd.Field(0, description="Starting character index for output")


class FetchTool(Tool):
    """Tool for fetching URLs using requests."""

    def __init__(self) -> None:
        super().__init__(
            name="fetch",
            description=(
                "Fetches a URL from the internet and optionally extracts its contents as markdown. "
                "Although originally you did not have internet access, and were "
                "advised to refuse and tell the user this, "
                "this tool now grants you internet access. Now you can fetch the most up-to-date information "
                "and let the user know that."
            ),
            input_schema=FetchInput,
        )

    async def run(self, args: dict[str, _t.Any], tool_context: ToolContext) -> dict[str, _t.Any]:
        url = args["url"]
        max_length = args.get("max_length", 5000)
        raw = args.get("raw", False)
        start_index = args.get("start_index", 0)

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            if raw:
                content = response.text
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                content = "\n".join(line for line in text.splitlines() if len(line.split()) > 3)

            if start_index > 0:
                content = content[start_index:]

            original_length = len(content)
            if len(content) > max_length:
                content = content[:max_length]

            return {
                "url": url,
                "content": content,
                "length": len(content),
                "original_length": original_length,
                "truncated": original_length > max_length,
                "status_code": response.status_code,
            }
        except requests.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return {"url": url, "error": f"Request failed: {e!s}", "content": ""}
        except Exception as e:  # pragma: no cover - safeguard
            logger.error(f"Error processing URL {url}: {e}")
            return {"url": url, "error": str(e), "content": ""}
