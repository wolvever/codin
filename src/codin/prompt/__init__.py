"""Wrapper module exposing the :mod:`prompti` package under ``codin.prompt``.

This repository's original sources expected the ``codin.prompt`` package to be
provided via a local checkout of the `prompti` project. In this trimmed
repository we simply re-export the installed :mod:`prompti` package so that the
rest of the code and tests can import ``codin.prompt`` transparently.
"""

from __future__ import annotations

import importlib
import sys

import prompti as _prompti
from prompti import *  # type: ignore F401,F403

# ruff: noqa: F405 - allow re-exported names below
# Local minimal implementations used in tests
from . import base as _codin_base
from . import registry as _codin_registry
from . import run as _codin_run

# Re-export commonly used submodules so imports like
# ``from codin.prompt.engine import PromptEngine`` continue to work.
for _name in (
    "engine",
    "loader",
    "message",
    "model_client",
    "experiment",
):
    try:
        _mod = importlib.import_module(f"prompti.{_name}")
        sys.modules[f"{__name__}.{_name}"] = _mod
    except Exception:  # pragma: no cover - optional extras
        pass

# Override with lightweight codin implementations
sys.modules[f"{__name__}.base"] = _codin_base
sys.modules[f"{__name__}.registry"] = _codin_registry
sys.modules[f"{__name__}.run"] = _codin_run

# Export key names at the package level for convenience
PromptTemplate = _codin_base.PromptTemplate
PromptVariant = _codin_base.PromptVariant
RenderedPrompt = _codin_base.RenderedPrompt
ToolDefinition = _codin_base.ToolDefinition

PromptRegistry = _codin_registry.PromptRegistry
get_registry = _codin_registry.get_registry

PromptEngine = _codin_run.PromptEngine
prompt_run = _codin_run.prompt_run
prompt_render = _codin_run.prompt_render

# Re-export everything defined by prompti at the package level and add local helpers
__all__ = list(getattr(_prompti, "__all__", []))
__all__ += [
    "PromptTemplate",
    "PromptVariant",
    "RenderedPrompt",
    "ToolDefinition",
    "PromptRegistry",
    "get_registry",
    "PromptEngine",
    "prompt_run",
    "prompt_render",
]
