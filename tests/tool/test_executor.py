"""Tests for the ToolExecutor with async generator handling."""

import asyncio
import time
import pytest
import typing as _t

from codin.tool.base import Tool, ToolContext
from codin.tool.executor import ToolExecutor
from codin.tool.registry import ToolRegistry
from codin.agent.types import TextPart


class AsyncGeneratorTestTool(Tool):
    """Test tool that returns an async generator."""

    def __init__(self):
        super().__init__(name="async_generator_test", description="Test tool that returns an async generator")

    async def run(self, args: dict[str, _t.Any], context: ToolContext) -> _t.AsyncGenerator[str, None]:
        """Return an async generator."""
        for i in range(3):
            yield f"item_{i}"


class RegularTestTool(Tool):
    """Test tool that returns a regular result."""

    def __init__(self):
        super().__init__(name="regular_test", description="Test tool that returns a regular result")

    async def run(self, args: dict[str, _t.Any], context: ToolContext) -> str:
        """Return a regular string result."""
        return "regular_result"


@pytest.fixture
def mock_context():
    """Create a mock ToolContext for testing."""
    return ToolContext(
        session_id="test_session",
        tool_call_id="test_request",
    )


@pytest.fixture
def registry():
    """Create a tool registry with test tools."""
    registry = ToolRegistry()
    registry.register_tool(AsyncGeneratorTestTool())
    registry.register_tool(RegularTestTool())
    return registry


@pytest.fixture
def executor(registry):
    """Create a tool executor with the test registry."""
    return ToolExecutor(registry)


@pytest.mark.asyncio
async def test_executor_with_async_generator(executor, mock_context):
    """Test that executor properly handles tools that return async generators."""
    result = await executor.execute("async_generator_test", {}, mock_context)

    # Should return a list of items from the async generator
    assert isinstance(result, list)
    assert len(result) == 3
    assert result == ["item_0", "item_1", "item_2"]


@pytest.mark.asyncio
async def test_executor_with_regular_tool(executor, mock_context):
    """Test that executor properly handles tools that return regular results."""
    result = await executor.execute("regular_test", {}, mock_context)

    # Should return the string result converted to TextPart
    assert isinstance(result, TextPart)
    assert result.text == "regular_result"


@pytest.mark.asyncio
async def test_executor_with_single_item_async_generator(executor, mock_context):
    """Test executor with async generator that yields single item."""

    class SingleItemAsyncGeneratorTool(Tool):
        def __init__(self):
            super().__init__(name="single_item_test", description="Test tool that returns single item async generator")

        async def run(self, args: dict[str, _t.Any], context: ToolContext) -> _t.AsyncGenerator[str, None]:
            yield "single_item"

    # Register the tool
    executor.registry.register_tool(SingleItemAsyncGeneratorTool())

    result = await executor.execute("single_item_test", {}, mock_context)

    # Should return the single item converted to TextPart
    assert isinstance(result, TextPart)
    assert result.text == "single_item"


class SleepTool(Tool):
    """Tool that sleeps briefly to test concurrency."""

    def __init__(self, name: str = "sleep_tool"):
        super().__init__(name=name, description="Sleep tool")

    async def run(self, args: dict[str, _t.Any], context: ToolContext) -> str:
        await asyncio.sleep(0.5)
        return context.tool_call_id or "done"


@pytest.mark.asyncio
async def test_executor_parallel_execution():
    """Tools run concurrently when concurrency > 1."""
    registry = ToolRegistry()
    registry.register_tool(SleepTool())
    executor = ToolExecutor(registry, max_concurrency=2)

    ctx1 = ToolContext(session_id="s", tool_call_id="one")
    ctx2 = ToolContext(session_id="s", tool_call_id="two")

    start = time.monotonic()
    r1, r2 = await asyncio.gather(
        executor.execute("sleep_tool", {}, ctx1),
        executor.execute("sleep_tool", {}, ctx2),
    )
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert isinstance(r1, TextPart) and r1.text == "one"
    assert isinstance(r2, TextPart) and r2.text == "two"


@pytest.mark.asyncio
async def test_executor_concurrency_limit():
    """Concurrency limit of 1 forces sequential execution."""
    registry = ToolRegistry()
    registry.register_tool(SleepTool())
    executor = ToolExecutor(registry, max_concurrency=1)

    ctx1 = ToolContext(session_id="s", tool_call_id="one")
    ctx2 = ToolContext(session_id="s", tool_call_id="two")

    start = time.monotonic()
    t1 = asyncio.create_task(executor.execute("sleep_tool", {}, ctx1))
    await asyncio.sleep(0.1)
    t2 = asyncio.create_task(executor.execute("sleep_tool", {}, ctx2))

    r1, r2 = await asyncio.gather(t1, t2)
    elapsed = time.monotonic() - start

    assert elapsed >= 0.9
    assert isinstance(r1, TextPart) and r1.text == "one"
    assert isinstance(r2, TextPart) and r2.text == "two"
