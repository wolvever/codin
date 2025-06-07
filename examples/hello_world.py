#!/usr/bin/env python3
"""
Hello World Example for BaseAgent with CodePlanner

This example demonstrates using BaseAgent with CodePlanner to complete a task:
"Write a python hello world script and check output result"

Usage:
    python examples/hello_world.py

Environment Variables:
    OPENAI_API_KEY - Required for OpenAI models
    ANTHROPIC_API_KEY - Required for Anthropic models
    GOOGLE_API_KEY - Required for Google models
    MODEL_PROVIDER - Optional, defaults to "openai"
    MODEL_NAME - Optional, defaults to "gpt-4"
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from a2a.types import Role, TextPart

from codin.agent.base_agent import BaseAgent
from codin.agent.code_planner import CodePlanner, CodePlannerConfig
from codin.agent.types import AgentRunInput, RunConfig, Message
from codin.memory.base import MemMemoryService
from codin.tool.registry import ToolRegistry
from codin.tool.sandbox import SandboxToolset
from codin.sandbox.local import LocalSandbox
from codin.model.factory import LLMFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_agent_and_planner():
    """Create and configure BaseAgent with CodePlanner and tools."""

    # 1. Initialize sandbox for code execution
    sandbox = LocalSandbox()
    await sandbox.up()
    logger.info("✓ Sandbox initialized")

    # 2. Setup tool registry with sandbox tools
    tool_registry = ToolRegistry()
    sandbox_toolset = SandboxToolset(sandbox)
    await sandbox_toolset.up()
    tool_registry.register_toolset(sandbox_toolset)
    logger.info(f"✓ Registered {len(tool_registry.get_tools())} tools")

    # 3. Create LLM instance
    model_provider = os.getenv("MODEL_PROVIDER", "openai")
    model_name = os.getenv("MODEL_NAME", "gpt-4")
    llm = LLMFactory.create_llm(model=model_name)
    logger.info(f"✓ LLM created: {model_name}")

    # 4. Configure and create planner
    planner_config = CodePlannerConfig(
        model=model_name,
        max_tokens=4000,
        temperature=0.7,
        max_tool_calls_per_turn=5,
        thinking_enabled=True,
        streaming_enabled=False,
        rules="Always write and execute code step by step. Verify output after execution.",
    )
    planner = CodePlanner(config=planner_config, llm=llm, tool_registry=tool_registry, debug=True)
    logger.info("✓ CodePlanner created")

    # 5. Initialize memory service
    memory = MemMemoryService()
    logger.info("✓ Memory service initialized")

    # 6. Create agent with run configuration
    run_config = RunConfig(
        turn_budget=20,
        time_budget_seconds=300,  # 5 minutes
        token_budget=10000,
        cost_budget=1.0,
    )

    agent = BaseAgent(
        agent_id="hello-world-agent",
        name="HelloWorldAgent",
        description="Agent for writing and executing Python hello world script",
        planner=planner,
        memory=memory,
        tools=tool_registry.get_tools(),
        llm=llm,
        default_config=run_config,
        debug=True,
    )
    logger.info("✓ BaseAgent created")

    return agent, sandbox


async def run_hello_world_task():
    """Execute the hello world task using BaseAgent and CodePlanner."""

    print("🚀 Starting Hello World Example")
    print("=" * 50)

    # Create agent and planner
    agent, sandbox = await create_agent_and_planner()

    try:
        # Create task message
        task_message = Message(
            messageId="task-hello-world",
            role=Role.user,
            parts=[TextPart(text="Write a python hello world script and check output result")],
            contextId="hello-world-session",
            kind="message",
        )

        # Create agent run input
        agent_input = AgentRunInput(session_id="hello-world-session", message=task_message, options={})

        print(f"📝 Task: {task_message.parts[0].root.text}")
        print()

        # Run the agent
        print("🤖 Agent executing task...")
        async for output in agent.run(agent_input):
            if hasattr(output, "result") and output.result:
                # Extract text from result message
                if hasattr(output.result, "parts"):
                    for part in output.result.parts:
                        if hasattr(part, "text"):
                            print(f"🔹 {part.text}")
                        elif hasattr(part, "root") and hasattr(part.root, "text"):
                            print(f"🔹 {part.root.text}")

                # Show metadata if available
                if hasattr(output, "metadata") and output.metadata:
                    step_type = output.metadata.get("step_type", "unknown")
                    if step_type == "tool_call":
                        tool_name = output.metadata.get("tool_name", "unknown")
                        success = output.metadata.get("success", False)
                        status = "✅" if success else "❌"
                        print(f"  🔧 Tool: {tool_name} {status}")
                    elif step_type == "finish":
                        reason = output.metadata.get("reason", "completed")
                        print(f"  ✨ Finished: {reason}")

        print()
        print("✅ Task completed successfully!")

    except Exception as e:
        logger.error(f"Error during task execution: {e}", exc_info=True)
        print(f"❌ Task failed: {e}")

    finally:
        # Cleanup
        try:
            await agent.cleanup()
            await sandbox.down()
            logger.info("✓ Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


def check_environment():
    """Check if required environment variables are set."""
    required_vars = []

    # Check for at least one LLM API key
    llm_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    if not any(os.getenv(key) for key in llm_keys):
        required_vars.extend(llm_keys)

    if required_vars:
        print("❌ Missing required environment variables:")
        for var in required_vars:
            print(f"   {var}")
        print("\nPlease set at least one LLM API key and try again.")
        print("Example:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   export MODEL_PROVIDER='openai'")
        print("   export MODEL_NAME='gpt-4'")
        return False

    return True


async def main():
    """Main entry point."""
    if not check_environment():
        sys.exit(1)

    try:
        await run_hello_world_task()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
