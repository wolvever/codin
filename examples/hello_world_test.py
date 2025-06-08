#!/usr/bin/env python3
"""
Hello World Test Example - Mock LLM Version

This example demonstrates the BaseAgent + CodePlanner loop logic
using a mock LLM that doesn't require API keys.

Usage:
    python examples/hello_world_test.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codin.agent.types import Role, TextPart

from codin.agent.base_agent import BaseAgent
from codin.agent.code_planner import CodePlanner, CodePlannerConfig
from codin.agent.types import AgentRunInput, RunConfig, ToolCall, Message
from codin.memory.base import MemMemoryService
from codin.tool.registry import ToolRegistry
from codin.tool import SandboxToolset # Changed from codin.tool.sandbox
from codin.sandbox.local import LocalSandbox
from codin.model.base import BaseLLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLLM(BaseLLM):
    """Mock LLM for testing that doesn't require API keys."""

    def __init__(self):
        super().__init__("mock-llm")
        self.turn_count = 0

    @classmethod
    def supported_models(cls) -> list[str]:
        """Return supported model patterns."""
        return ["mock-*"]

    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Generate mock responses that simulate the hello world task."""
        self.turn_count += 1

        # Mock structured responses for different turns
        if self.turn_count == 1:
            # First turn: Create the hello world script
            response_content = """```json
{
    "thinking": "I need to create a Python hello world script and then execute it to check the output.",
    "task_list": {
        "completed": [],
        "pending": ["Create hello.py script", "Execute script", "Verify output"]
    },
    "tool_calls": [
        {
            "name": "edit_file",
            "arguments": {
                "target_file": "hello.py",
                "instructions": "Create a simple hello world Python script",
                "code_edit": "#!/usr/bin/env python3\\nprint('Hello, World!')"
            }
        }
    ],
    "message": "I'll create a simple Python hello world script for you.",
    "should_continue": true
}```"""
        elif self.turn_count == 2:
            # Second turn: Execute the script
            response_content = """```json
{
    "thinking": "Now I need to execute the hello world script to verify it works correctly.",
    "task_list": {
        "completed": ["Create hello.py script"],
        "pending": ["Execute script", "Verify output"]
    },
    "tool_calls": [
        {
            "name": "run_shell",
            "arguments": {
                "command": "python hello.py",
                "explanation": "Execute the hello world script to check output"
            }
        }
    ],
    "message": "Now let me execute the script to verify it works.",
    "should_continue": true
}```"""
        else:
            # Final turn: Task complete
            response_content = """```json
{
    "thinking": "The hello world script has been created and executed successfully. The task is complete.",
    "task_list": {
        "completed": ["Create hello.py script", "Execute script", "Verify output"],
        "pending": []
    },
    "tool_calls": [],
    "message": (
        "Perfect! I've successfully created and executed the Python hello world script. "
        "The output 'Hello, World!' confirms it's working correctly."
    ),
    "should_continue": false
}```"""

        return response_content

    async def generate_with_tools(
        self,
        prompt: str | list[dict[str, str]],
        tools: list[dict],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Generate with tools - delegate to generate for simplicity."""
        content = await self.generate(prompt, stream=stream, temperature=temperature, max_tokens=max_tokens)
        return {"content": content, "tool_calls": []}


async def create_test_agent():
    """Create BaseAgent with CodePlanner using mock LLM."""

    print("🔧 Setting up test environment...")

    # 1. Initialize sandbox
    sandbox = LocalSandbox()
    await sandbox.up()
    logger.info("✓ Mock sandbox initialized")

    # 2. Setup tool registry
    tool_registry = ToolRegistry()
    sandbox_toolset = SandboxToolset(sandbox)
    await sandbox_toolset.up()
    tool_registry.register_toolset(sandbox_toolset)
    logger.info(f"✓ Registered {len(tool_registry.get_tools())} tools")

    # 3. Create mock LLM
    mock_llm = MockLLM()
    logger.info("✓ Mock LLM created")

    # 4. Configure planner
    planner_config = CodePlannerConfig(
        model="mock-llm",
        max_tokens=2000,
        temperature=0.7,
        max_tool_calls_per_turn=3,
        thinking_enabled=True,
        streaming_enabled=False,
        rules="Test environment - simulate creating and executing hello world script",
    )
    planner = CodePlanner(config=planner_config, llm=mock_llm, tool_registry=tool_registry, debug=True)
    logger.info("✓ CodePlanner created with mock LLM")

    # 5. Initialize memory
    memory = MemMemoryService()
    logger.info("✓ Memory service initialized")

    # 6. Create agent
    run_config = RunConfig(turn_budget=10, time_budget_seconds=60, token_budget=5000, cost_budget=0.10)

    agent = BaseAgent(
        agent_id="test-hello-world-agent",
        name="TestHelloWorldAgent",
        description="Test agent for hello world task using mock LLM",
        planner=planner,
        memory=memory,
        tools=tool_registry.get_tools(),
        llm=mock_llm,
        default_config=run_config,
        debug=True,
    )
    logger.info("✓ BaseAgent created")

    return agent, sandbox


async def run_test():
    """Run the test hello world task."""

    print("🚀 Starting Hello World Test (Mock LLM)")
    print("=" * 50)

    agent, sandbox = await create_test_agent()

    try:
        # Create test task message
        task_message = Message(
            messageId="test-task-hello-world",
            role=Role.user,
            parts=[TextPart(text="Write a python hello world script and check output result")],
            contextId="test-session",
            kind="message",
        )

        # Create agent input
        agent_input = AgentRunInput(session_id="test-session", message=task_message, options={})

        print(f"📝 Task: {task_message.parts[0].root.text}")
        print()
        print("🤖 Agent executing task with mock LLM...")
        print()

        # Track execution
        step_count = 0
        tool_calls_made = 0

        # Run the agent
        async for output in agent.run(agent_input):
            step_count += 1

            if hasattr(output, "result") and output.result:
                # Extract and display content
                if hasattr(output.result, "parts"):
                    for part in output.result.parts:
                        text = ""
                        if hasattr(part, "text"):
                            text = part.text
                        elif hasattr(part, "root") and hasattr(part.root, "text"):
                            text = part.root.text

                        if text and len(text.strip()) > 0:
                            # Truncate very long outputs for readability
                            if len(text) > 200:
                                text = text[:200] + "..."
                            print(f"🔹 {text}")

                # Show metadata
                if hasattr(output, "metadata") and output.metadata:
                    step_type = output.metadata.get("step_type", "unknown")
                    if step_type == "tool_call":
                        tool_calls_made += 1
                        tool_name = output.metadata.get("tool_name", "unknown")
                        success = output.metadata.get("success", False)
                        status = "✅" if success else "❌"
                        print(f"  🔧 Tool: {tool_name} {status}")
                    elif step_type == "finish":
                        reason = output.metadata.get("reason", "completed")
                        print(f"  ✨ Finished: {reason}")
                        break

        print()
        print("📊 Execution Summary:")
        print(f"   Steps executed: {step_count}")
        print(f"   Tool calls made: {tool_calls_made}")
        print()
        print("✅ Test completed successfully!")
        print()
        print("🎯 This demonstrates that BaseAgent + CodePlanner implements")
        print("   the same iterative loop logic as CodeAgent and Codex.rs:")
        print("   1. Planner generates steps (Message, ToolCall, Finish)")
        print("   2. Agent executes each step")
        print("   3. Tool results feed back to planner")
        print("   4. Loop continues until FinishStep")

    except Exception as e:
        logger.error(f"Error during test execution: {e}", exc_info=True)
        print(f"❌ Test failed: {e}")

    finally:
        # Cleanup
        try:
            await agent.cleanup()
            await sandbox.down()
            logger.info("✓ Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


async def main():
    """Main entry point."""
    try:
        await run_test()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected test error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
