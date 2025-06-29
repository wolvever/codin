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

# Re-export commonly used submodules so imports like
# ``from codin.prompt.engine import PromptEngine`` continue to work.
for _name in (
    "base",
    "engine",
    "registry",
    "run",
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

# Re-export everything defined by prompti at the package level
__all__ = getattr(_prompti, "__all__", [])
