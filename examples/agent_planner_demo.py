#!/usr/bin/env python3
"""
Demo of the new planner-based agent architecture.

This demonstrates the separation of concerns between:
- Agent (base.py): Framework-agnostic wrapper (unchanged)
- Planner: Generates execution steps from state
- BaseAgent: Orchestrates Session, State, Planner and execution loop
- Session: Data-oriented session management like Codex-rs
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Demo the new agent architecture."""
    try:
        from a2a.types import Message, Role, TextPart
        from src.codin.agent import (
            BaseAgent, 
            CodePlanner, 
            CodePlannerConfig,
            SessionManager,
            AgentRunInput
        )
        from src.codin.tool.registry import ToolRegistry
        from src.codin.tool.core_tools import CoreToolset
        
        # Create components
        logger.info("🔧 Setting up agent components...")
        
        # 1. Create tool registry with basic tools
        tool_registry = ToolRegistry()
        core_toolset = CoreToolset()
        tool_registry.register_toolset(core_toolset)
        
        # 2. Create planner configuration
        planner_config = CodePlannerConfig(
            model="gpt-4",
            thinking_enabled=True,
            streaming_enabled=True,
            rules="Be helpful and concise. Always think through problems step by step."
        )
        
        # 3. Create planner
        planner = CodePlanner(
            config=planner_config,
            tool_registry=tool_registry,
            debug=True
        )
        
        # 4. Create session manager
        session_manager = SessionManager()
        
        # 5. Create the BaseAgent
        agent = BaseAgent(
            name="DemoAgent",
            description="Demo agent using the new planner architecture",
            planner=planner,
            session_manager=session_manager,
            tool_registry=tool_registry,
            debug=True
        )
        
        # 6. Add event callback for monitoring
        async def event_monitor(event):
            event_type = event["event_type"]
            timestamp = event["timestamp"].strftime("%H:%M:%S")
            data = event["data"]
            
            if event_type == "task_start":
                logger.info(f"📋 [{timestamp}] Task started: {data.get('input_message', '')[:50]}...")
            elif event_type == "turn_start":
                logger.info(f"🔄 [{timestamp}] Turn {data['turn']} started")
            elif event_type == "agent_thinking":
                logger.info(f"🧠 [{timestamp}] Agent thinking: {data['thinking'][:50]}...")
            elif event_type == "tool_call_start":
                logger.info(f"🔧 [{timestamp}] Tool call: {data['tool_name']}")
            elif event_type == "tool_call_end":
                status = "✅" if data['success'] else "❌"
                logger.info(f"{status} [{timestamp}] Tool completed: {data['tool_name']}")
            elif event_type == "agent_message":
                logger.info(f"💬 [{timestamp}] Agent message sent")
            elif event_type == "task_complete":
                logger.info(f"🎉 [{timestamp}] Task completed!")
        
        agent.add_event_callback(event_monitor)
        
        # 7. Create a test message
        test_message = Message(
            messageId="demo-001",
            role=Role.user,
            parts=[TextPart(text="List the files in the current directory.")],
            contextId="demo-session",
            kind="message"
        )
        
        # 8. Create agent input
        agent_input = AgentRunInput(
            message=test_message,
            session_id="demo-session-001"
        )
        
        logger.info("🚀 Running agent with new architecture...")
        
        # 9. Run the agent
        result = await agent.run(agent_input)
        
        # 10. Display results
        logger.info("✅ Agent execution completed!")
        logger.info(f"Session ID: {result.metadata.get('session_id')}")
        logger.info(f"Turns: {result.metadata.get('turns')}")
        logger.info(f"Execution time: {result.metadata.get('execution_time', 0):.2f}s")
        
        # Extract and display the final message
        if hasattr(result.result, 'parts'):
            response_text = ""
            for part in result.result.parts:
                if hasattr(part, 'text'):
                    response_text += part.text
                elif hasattr(part, 'root') and hasattr(part.root, 'text'):
                    response_text += part.root.text
            
            logger.info(f"📝 Final response: {response_text[:200]}...")
        
        # 11. Show session metrics
        session_metrics = result.metadata.get('session_metrics', {})
        logger.info("📊 Session Metrics:")
        for key, value in session_metrics.items():
            logger.info(f"  {key}: {value}")
        
        # 12. Cleanup
        await agent.cleanup()
        logger.info("🧹 Cleanup completed")
        
    except ImportError as e:
        logger.error(f"Import error - make sure the project is properly installed: {e}")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    print("🎯 Agent Planner Architecture Demo")
    print("=" * 50)
    print()
    print("This demo showcases the new architecture where:")
    print("1. 📋 Agent orchestrates the execution loop")
    print("2. 🧠 Planner generates execution steps from state")
    print("3. 💾 Session manages conversation data (like Codex-rs)")
    print("4. 🔄 Clean separation of planning vs execution")
    print()
    
    asyncio.run(main()) 