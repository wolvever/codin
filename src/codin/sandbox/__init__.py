"""Sandbox abstraction layer.

This package provides a unified interface for interacting with sandboxed execution
environments. Multiple backends are supported including local subprocess execution,
E2B cloud sandboxes, Daytona Runner API, and Codex CLI sandboxes.
"""

from .base import ExecResult, Sandbox
from .local import LocalSandbox
from .e2b import E2BSandbox
from .daytona import DaytonaSandbox
from .codex import CodexSandbox
from .factory import create_sandbox

__all__ = [
    "ExecResult",
    "Sandbox",
    "LocalSandbox",
    "E2BSandbox", 
    "DaytonaSandbox",
    "CodexSandbox",
    "create_sandbox",
] 