from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from jinja2 import Environment, StrictUndefined

__all__ = [
    "PromptTemplate",
    "PromptVariant",
    "RenderedPrompt",
    "ToolDefinition",
]

_env = Environment(undefined=StrictUndefined)


@dataclass
class PromptVariant:
    text: str
    conditions: Optional[dict[str, Any]] = None


@dataclass
class RenderedPrompt:
    text: str


@dataclass
class PromptTemplate:
    name: str
    text: str = ""
    version: str = "latest"
    variants: list[PromptVariant] = field(default_factory=list)

    def add_variant(self, variant: PromptVariant) -> None:
        self.variants.append(variant)

    def _select_text(self, conditions: Optional[dict[str, Any]] = None) -> str:
        if not self.variants:
            return self.text
        if not conditions:
            return self.variants[0].text
        for variant in self.variants:
            if not variant.conditions:
                continue
            if all(conditions.get(k) == v for k, v in variant.conditions.items()):
                return variant.text
        return self.variants[0].text

    def render(
        self,
        *,
        variables: Optional[dict[str, Any]] = None,
        conditions: Optional[dict[str, Any]] = None,
    ) -> RenderedPrompt:
        variables = variables or {}
        text = self._select_text(conditions) or self.text
        template = _env.from_string(text)
        return RenderedPrompt(text=template.render(**variables))


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
