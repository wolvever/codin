from __future__ import annotations

from typing import Any, Optional

from ..model.mock_llm import MockLLM
from .base import RenderedPrompt
from .registry import get_registry

__all__ = ["PromptEngine", "prompt_run", "prompt_render"]


class PromptEngine:
    def __init__(self, llm: MockLLM | str, endpoint: str | None = None) -> None:
        if isinstance(llm, str):
            self.llm = MockLLM(llm)
        else:
            self.llm = llm
        self.endpoint = endpoint

    async def run(
        self,
        template_name: str,
        *,
        version: str = "latest",
        stream: bool = False,
        tools: list[dict] | None = None,
        variables: Optional[dict[str, Any]] = None,
        conditions: Optional[dict[str, Any]] = None,
    ) -> RenderedPrompt:
        registry = get_registry()
        template = registry.get(template_name, version)
        rendered = template.render(variables=variables, conditions=conditions)
        if tools:
            result = await self.llm.generate_with_tools(rendered.text, tools, stream=stream)
        else:
            result = await self.llm.generate(rendered.text, stream=stream)
        return RenderedPrompt(text=result if isinstance(result, str) else str(result))

    async def render_only(
        self,
        template_name: str,
        *,
        version: str = "latest",
        variables: Optional[dict[str, Any]] = None,
        conditions: Optional[dict[str, Any]] = None,
    ) -> RenderedPrompt:
        registry = get_registry()
        template = registry.get(template_name, version)
        return template.render(variables=variables, conditions=conditions)


_engine: Optional[PromptEngine] = None


async def _get_engine() -> PromptEngine:
    global _engine
    if _engine is None:
        _engine = PromptEngine(MockLLM("mock-llm"))
    return _engine


async def prompt_run(
    template_name: str,
    *,
    version: str = "latest",
    variables: Optional[dict[str, Any]] = None,
    conditions: Optional[dict[str, Any]] = None,
    stream: bool = False,
    tools: list[dict] | None = None,
) -> str:
    engine = await _get_engine()
    result = await engine.run(
        template_name,
        version=version,
        stream=stream,
        tools=tools,
        variables=variables,
        conditions=conditions,
    )
    return result.text


async def prompt_render(
    template_name: str,
    *,
    version: str = "latest",
    variables: Optional[dict[str, Any]] = None,
    conditions: Optional[dict[str, Any]] = None,
) -> str:
    engine = await _get_engine()
    rendered = await engine.render_only(
        template_name,
        version=version,
        variables=variables,
        conditions=conditions,
    )
    return rendered.text
