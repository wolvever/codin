"""Core tool definitions built on top of sandbox methods."""

from __future__ import annotations

import logging
import typing as _t

# pydantic and requests related imports are no longer needed if FetchTool is the only user
# from bs4 import BeautifulSoup # No longer needed if FetchTool is removed

from .base import Tool, ToolContext # Tool and ToolContext might be used by other tools if any remain

__all__ = [] # FetchTool removed from __all__

logger = logging.getLogger(__name__)


# FetchInput class removed
# FetchTool class removed

# If there are other tools in this file, they would remain here.
# Based on the provided content, FetchTool was the only one.
# If this file becomes empty of tools, it might be a candidate for deletion
# or refactoring, but the task is only to remove FetchTool.
