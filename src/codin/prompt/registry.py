from __future__ import annotations

import builtins
import os
from pathlib import Path
from typing import Optional

import httpx

from .base import PromptTemplate

__all__ = ["PromptRegistry", "get_registry"]


class PromptRegistry:
    _instance: Optional[PromptRegistry] = None

    def __init__(self, endpoint: str | None = None) -> None:
        self._registry: dict[tuple[str, str], PromptTemplate] = {}
        self._in_memory_templates: dict[tuple[str, str], PromptTemplate] = {}
        self.endpoint = endpoint
        self.run_mode = os.environ.get("PROMPT_RUN_MODE", "local")
        self.template_dir = os.environ.get("PROMPT_TEMPLATE_DIR", "tests/fixtures/prompts")
        self.remote_base_url = os.environ.get("PROMPT_REMOTE_BASE_URL")

    # Singleton helpers
    @classmethod
    def get_instance(cls) -> PromptRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # Class helpers used in tests
    @classmethod
    def register_template(cls, tmpl: PromptTemplate) -> None:
        cls.get_instance().register(tmpl)

    @classmethod
    def get_template(cls, name: str, version: str = "latest") -> PromptTemplate:
        return cls.get_instance().get(name, version)

    @classmethod
    def list_templates(cls, name: str | None = None) -> builtins.list[PromptTemplate]:
        return cls.get_instance().list(name)

    @classmethod
    def set_mode(cls, mode: str) -> None:
        cls.get_instance().set_run_mode(mode)

    @classmethod
    def get_mode(cls) -> str:
        return cls.get_instance().run_mode

    # Instance methods
    def register(self, template: PromptTemplate) -> None:
        key = (template.name, template.version)
        self._registry[key] = template
        self._in_memory_templates[key] = template

    def list(self, name: str | None = None) -> builtins.list[PromptTemplate]:
        if name is None:
            return list(self._registry.values())
        return [t for (n, _), t in self._registry.items() if n == name]

    def set_run_mode(self, mode: str) -> None:
        if mode not in {"local", "remote"}:
            raise ValueError("Invalid mode")
        self.run_mode = mode

    def _load_local(self, name: str, version: str) -> PromptTemplate | None:
        dir_path = Path(self.template_dir)
        if version == "latest":
            path = dir_path / f"{name}.jinja2"
        else:
            path = dir_path / f"{name}.{version}.jinja2"
        if path.exists():
            text = path.read_text()
            tmpl = PromptTemplate(name=name, version=version, text=text)
            self.register(tmpl)
            return tmpl
        return None

    def _load_remote(self, name: str, version: str) -> PromptTemplate | None:
        if not self.remote_base_url:
            return None
        url = f"{self.remote_base_url.rstrip('/')}/{name}"
        if version != "latest":
            url += f"?version={version}"
        try:
            response = httpx.get(url)
            if response.status_code != 200:
                return None
            data = response.json()
            text = data.get("text")
            if not text and data.get("variants"):
                text = data["variants"][0]["text"]
            tmpl = PromptTemplate(
                name=data.get("name", name),
                version=data.get("version", version),
                text=text or "",
            )
            self.register(tmpl)
            return tmpl
        except Exception:
            return None

    def get(self, name: str, version: str = "latest") -> PromptTemplate:
        key = (name, version)
        if key in self._registry:
            return self._registry[key]
        tmpl = None
        if self.run_mode == "local":
            tmpl = self._load_local(name, version)
        else:
            tmpl = self._load_remote(name, version)
        if tmpl is None:
            raise KeyError(f"Template '{name}' version '{version}' not found")
        return tmpl


def get_registry() -> PromptRegistry:
    return PromptRegistry.get_instance()
