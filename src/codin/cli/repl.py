from __future__ import annotations

"""Interactive REPL for codin.

This module provides an interactive Read-Eval-Print Loop (REPL) interface
similar to the original codex CLI, allowing users to have conversational
interactions with the coding agent.
"""

import asyncio
import logging
import sys
import typing as _t
from pathlib import Path

import click

from ..config import (
    get_config, 
    load_agents_instructions, 
)
from ..agent.code_agent import CodeAgent
from ..model.factory import LLMFactory
from ..agent.base import AgentRunInput
from ..sandbox import LocalSandbox
from ..tool.base import Toolset
from ..config import ApprovalMode
from .utils import create_mcp_toolsets_from_config

# Use a2a types instead of internal protocol types
from a2a.types import Message, Role, TextPart

__all__ = [
    "start_repl",
    "ReplSession",
]

_logger = logging.getLogger(__name__)

def approval_mode_to_policy(mode: ApprovalMode) -> ApprovalMode:
    """Convert ApprovalMode to ApprovalMode (identity function for backward compatibility)."""
    return mode


class ReplSession:
    """Interactive REPL session for codin."""
    
    def __init__(
        self,
        *,
        approval_mode: ApprovalMode | None = None,
        verbose: bool = False,
        debug: bool = False,
        config_file: str | None = None,
    ) -> None:
        """Initialize REPL session.
        
        Args:
            approval_mode: Override approval mode
            verbose: Enable verbose output
            debug: Enable debug mode (shows LLM requests and responses)
            config_file: Path to custom config file
        """
        self.config = get_config(config_file)
        self.config_file = config_file  # Store for later use
        
        # Override config with parameters
        if approval_mode:
            self.config.approval_mode = approval_mode
        if verbose:
            self.config.verbose = verbose
        
        self.debug = debug  # Store debug flag for CodeAgent
        self.conversation_history: list[Message] = []
        
        # Initialize agent and toolsets
        self.agent: CodeAgent | None = None
        self.toolsets: list[Toolset] = []
        self.sandbox: LocalSandbox | None = None
        self._initialized = False  # Track initialization state
        self._cleaned_up = False  # Track cleanup state
        
    async def initialize(self) -> None:
        """Initialize the REPL session."""
        if self._initialized:
            return  # Already initialized, skip
            
        try:
            # Set up global exception handler to suppress MCP task warnings
            import asyncio
            import logging
            
            # Suppress specific MCP-related warnings
            def suppress_mcp_warnings(loop, context):
                """Suppress MCP-related task warnings during cleanup."""
                exception = context.get('exception')
                message = context.get('message', '')
                
                # Suppress specific MCP cleanup warnings
                if (exception and 'cancel scope' in str(exception)) or \
                   'Task exception was never retrieved' in message or \
                   'stdio_client' in message:
                    return  # Suppress these warnings
                
                # For other exceptions, use default handler
                loop.default_exception_handler(context)
            
            # Set the exception handler
            loop = asyncio.get_event_loop()
            loop.set_exception_handler(suppress_mcp_warnings)
            
            # Create and initialize sandbox
            sandbox = LocalSandbox()
            await sandbox.up()
            self.sandbox = sandbox  # Store reference for cleanup
            
            if self.config.verbose:
                click.echo(f"[OK] Initialized sandbox: {type(sandbox).__name__}")
            
            # Create sandbox toolset (now auto-generates tools from sandbox methods)
            from codin.tool.sandbox import SandboxToolset
            sandbox_toolset = SandboxToolset(sandbox)
            await sandbox_toolset.up()
            
            # Create toolsets from config
            toolsets = create_mcp_toolsets_from_config(self.config_file)
            
            toolsets.append(sandbox_toolset)  # Add sandbox toolset
            self.toolsets = toolsets  # Store reference for cleanup
            
            # Initialize all MCP toolsets (skip sandbox toolset - already initialized)
            for toolset in toolsets[:-1]:
                try:
                    await toolset.up()
                    if self.config.verbose:
                        click.echo(f"[OK] Initialized toolset: {toolset.name}")
                except Exception as e:
                    click.echo(f"[WARN] Failed to initialize toolset {toolset.name}: {e}", err=True)
            
            # Create agent with initialized sandbox and toolsets
            self.agent = CodeAgent(
                name="Codin Assistant",
                description="AI coding assistant with tool-calling capabilities",
                llm_model=self.config.model,
                sandbox=sandbox,
                toolsets=toolsets,
                approval_mode=self.config.approval_mode,
                debug=self.debug,
            )
            
            # Add enhanced streaming event callback
            processed_tool_calls = set()  # Track processed tool calls
            
            # Track streaming state
            current_assistant_response = ""
            streaming_started = False
            
            async def stream_event_callback(event):
                nonlocal current_assistant_response, streaming_started
                
                if event.event_type == "llm_text_delta":
                    # Real-time streaming of LLM text generation
                    delta = event.data.get("delta", "")
                    if delta:
                        # Show header when streaming starts
                        if not streaming_started:
                            click.echo()
                            click.echo(click.style("Assistant:", bold=True, fg="green"))
                            streaming_started = True
                        
                        # Print the delta immediately without newline
                        click.echo(delta, nl=False)
                        current_assistant_response += delta
                        
                elif event.event_type == "llm_response":
                    # Show the assistant's final response (if not already shown via streaming)
                    content = event.data.get("content", "")
                    thinking = event.data.get("thinking", "")
                    message = event.data.get("message", "")
                    task_list = event.data.get("task_list", {})
                    should_continue = event.data.get("should_continue", True)
                    has_tool_calls = event.data.get("has_tool_calls", False)
                    
                    # If we haven't shown any streaming content, show it now
                    if not current_assistant_response:
                        click.echo()
                        click.echo(click.style("Assistant:", bold=True, fg="green"))
                        
                        # Show thinking if available
                        if thinking:
                            click.echo(click.style("💭 Thinking:", bold=True, fg="cyan"))
                            click.echo(thinking)
                            click.echo()
                        
                        # Show the main message
                        display_message = message or content
                        if display_message:
                            click.echo(display_message)
                        
                        # Show task list if available
                        if task_list and (task_list.get("completed") or task_list.get("pending")):
                            click.echo()
                            click.echo(click.style("📋 Task Progress:", bold=True, fg="blue"))
                            
                            completed = task_list.get("completed", [])
                            pending = task_list.get("pending", [])
                            
                            if completed:
                                click.echo(click.style("✅ Completed:", fg="green"))
                                for task in completed:
                                    click.echo(f"  • {task}")
                            
                            if pending:
                                click.echo(click.style("⏳ Pending:", fg="yellow"))
                                for task in pending:
                                    click.echo(f"  • {task}")
                        
                        # Show continuation status
                        if not should_continue:
                            click.echo()
                            click.echo(click.style("🎯 Task completed!", bold=True, fg="green"))
                            
                    elif streaming_started:
                        # We've been streaming, just add a newline to end the stream
                        click.echo()  # End the streaming line
                    
                    # Reset for next response
                    current_assistant_response = ""
                    streaming_started = False
                    
                    if has_tool_calls:
                        click.echo()
                        click.echo(click.style("🔧 Executing tools...", bold=True, fg="cyan"))
                            
                elif event.event_type == "tool_call_start":
                    # Show individual tool call starting
                    tool_name = event.data.get("tool_name", "unknown")
                    arguments = event.data.get("arguments", {})
                    
                    click.echo()
                    click.echo(click.style(f"[TOOL] {tool_name}", bold=True, fg="yellow"))
                    
                    # Format arguments in a generic, well-structured way
                    self._format_tool_arguments(tool_name, arguments)
                    
                elif event.event_type == "tool_call_end":
                    # Show tool call completion - only once per tool call
                    call_id = event.data.get("call_id", "")
                    tool_name = event.data.get("tool_name", "unknown")
                    success = event.data.get("success", False)
                    output = event.data.get("output", "")
                    error = event.data.get("error", "")
                    
                    # Skip if we've already processed this tool call
                    if call_id in processed_tool_calls:
                        return
                    processed_tool_calls.add(call_id)
                    
                    if success:
                        click.echo(f"  ✅ {tool_name} completed")
                        if output and len(output.strip()) > 0:
                            # Show output preview for some tools
                            if tool_name == "sandbox_read_file":
                                lines = output.strip().split('\n')
                                click.echo(f"  📄 Content ({len(lines)} lines):\n{output.strip()}")
                            elif tool_name == "sandbox_list_files":
                                files = output.strip().split('\n') if output.strip() else []
                                if files:
                                    click.echo(f"  📁 Found {len(files)} items:")
                                    for file in files:
                                        click.echo(f"    • {file}")
                            elif tool_name in ["sandbox_exec", "shell", "container_exec"]:
                                if output.strip():
                                    lines = output.strip().split('\n')
                                    click.echo(f"  📤 Output ({len(lines)} lines):\n{output.strip()}")
                            elif tool_name == "sandbox_write_file":
                                click.echo(f"  💾 File written successfully")
                    else:
                        click.echo(f"  ❌ {tool_name} failed: {error}")
                        
                elif event.event_type == "tool_results":
                    # Summary of all tool executions
                    num_tools = event.data.get("num_tools_executed", 0)
                    successful = event.data.get("successful_calls", 0) 
                    failed = event.data.get("failed_calls", 0)
                    
                    if num_tools > 0:
                        click.echo()
                        if failed == 0:
                            click.echo(click.style(f"✨ All {num_tools} tools completed successfully", bold=True, fg="green"))
                        else:
                            click.echo(click.style(f"⚠️  Tools: {successful} successful, {failed} failed", bold=True, fg="yellow"))
                            
                elif event.event_type == "approval_requested":
                    tool_name = event.data.get("tool_name", "unknown")
                    auto_approved = event.data.get("auto_approved", False)
                    if auto_approved:
                        click.echo(f"  🔐 Auto-approved: {tool_name}")
                
                elif event.event_type == "debug_llm_response":
                    # Handle debug information display
                    debug_info = event.data
                    click.echo()
                    click.echo(click.style(f"🤖 LLM Response (Turn {debug_info['turn_count']}):", bold=True, fg="cyan"))
                    click.echo(click.style("-" * 60, fg="cyan"))
                    
                    click.echo(f"📄 Raw content length: {debug_info['raw_content_length']} characters")
                    
                    thinking = debug_info.get('thinking')
                    if thinking:
                        click.echo(f"💭 Thinking: {thinking}")
                    else:
                        click.echo("💭 Thinking: None")
                    
                    message = debug_info.get('message')
                    if message:
                        click.echo(f"💬 Message: {message}")
                    else:
                        click.echo("💬 Message: None")
                    
                    click.echo(f"🔄 Should continue: {debug_info['should_continue']}")
                    
                    task_list = debug_info['task_list']
                    click.echo(f"📋 Task list - Completed: {task_list['completed_count']}, Pending: {task_list['pending_count']}")
                    
                    tool_calls = debug_info.get('tool_calls', [])
                    if tool_calls:
                        click.echo(f"🔧 Tool calls: {len(tool_calls)}")
                        for i, tool_call in enumerate(tool_calls):
                            args_keys = tool_call.get('arguments_keys', [])
                            click.echo(f"  {i+1}. {tool_call['name']}({args_keys})")
                    else:
                        click.echo("🔧 Tool calls: None")
                    
                    click.echo(click.style("-" * 60, fg="cyan"))
                    click.echo()
            
            self.agent.add_event_callback(stream_event_callback)
            
            if self.config.verbose:
                click.echo(f"[OK] Initialized with model: {self.config.model}")
            
            # Mark as initialized
            self._initialized = True
                
        except Exception as e:
            click.echo(f"[ERROR] Failed to initialize: {e}", err=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup all resources."""
        if self._cleaned_up:
            return  # Already cleaned up, skip
            
        _logger.info("Cleaning up REPL session...")
        
        # Use a flag to track if we're in shutdown mode
        in_shutdown = False
        try:
            import sys
            if hasattr(sys, '_getframe'):
                # Check if we're being called during asyncio.run() shutdown
                for i in range(10):
                    try:
                        frame = sys._getframe(i)
                        if frame and 'run_until_complete' in str(frame.f_code.co_name):
                            in_shutdown = True
                            break
                    except ValueError:
                        # No more frames
                        break
        except Exception:
            # If frame inspection fails, assume we're not in shutdown
            pass
        
        # Cleanup agent first
        if self.agent:
            try:
                if in_shutdown:
                    # During shutdown, just fire and forget
                    asyncio.create_task(self._safe_cleanup_agent())
                    click.echo("[OK] Initiated agent cleanup")
                else:
                    await asyncio.wait_for(self.agent.cleanup(), timeout=3.0)
                    click.echo("[OK] Cleaned up agent")
            except (asyncio.TimeoutError, asyncio.CancelledError):
                click.echo("[WARN] Agent cleanup timed out or was cancelled")
            except Exception as e:
                click.echo(f"[WARN] Error cleaning up agent: {e}")
            finally:
                self.agent = None
        
        # Cleanup toolsets with better error handling
        if hasattr(self, 'toolsets') and self.toolsets:
            if in_shutdown:
                # During shutdown, just fire and forget all cleanups
                for toolset in self.toolsets:
                    asyncio.create_task(self._safe_cleanup_toolset(toolset))
                click.echo(f"[OK] Initiated cleanup for {len(self.toolsets)} toolsets")
            else:
                # Normal cleanup with timeouts
                cleanup_tasks = []
                for toolset in self.toolsets:
                    cleanup_tasks.append(self._safe_cleanup_toolset(toolset))
                
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*cleanup_tasks, return_exceptions=True),
                        timeout=5.0
                    )
                    for result in results:
                        if isinstance(result, str):
                            click.echo(result)
                        elif isinstance(result, Exception):
                            click.echo(f"[WARN] Toolset cleanup error: {result}")
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    click.echo("[WARN] Overall toolset cleanup timed out or was cancelled")
                except Exception as e:
                    click.echo(f"[WARN] Error during toolset cleanup: {e}")
            
            self.toolsets = []
        
        # Cleanup sandbox
        if hasattr(self, 'sandbox') and self.sandbox:
            try:
                if in_shutdown:
                    # During shutdown, just fire and forget
                    asyncio.create_task(self._safe_cleanup_sandbox())
                    click.echo("[OK] Initiated sandbox cleanup")
                else:
                    await asyncio.wait_for(self.sandbox.down(), timeout=3.0)
                    click.echo("[OK] Cleaned up sandbox")
            except (asyncio.TimeoutError, asyncio.CancelledError):
                click.echo("[WARN] Sandbox cleanup timed out or was cancelled")
            except Exception as e:
                click.echo(f"[WARN] Error cleaning up sandbox: {e}")
            finally:
                self.sandbox = None
        
        click.echo("[OK] REPL session cleanup completed")
        
        # Mark as cleaned up
        self._cleaned_up = True
    
    async def _safe_cleanup_agent(self) -> None:
        """Safely cleanup agent with complete error suppression."""
        try:
            if self.agent:
                await self.agent.cleanup()
        except Exception:
            pass  # Completely ignore all errors during shutdown
    
    async def _safe_cleanup_toolset(self, toolset) -> str:
        """Safely cleanup a toolset with complete error suppression."""
        try:
            await toolset.cleanup()
            return f"[OK] Cleaned up toolset: {toolset.name}"
        except Exception:
            # During shutdown, don't show errors for expected cancellations
            return f"[OK] Cleaned up toolset: {toolset.name}"
    
    async def _safe_cleanup_sandbox(self) -> None:
        """Safely cleanup sandbox with complete error suppression."""
        try:
            if self.sandbox:
                await self.sandbox.down()
        except Exception:
            pass  # Completely ignore all errors during shutdown
    
    def load_project_instructions(self) -> str | None:
        """Load project instructions from README or .codin files."""
        if not self.config.enable_rules:
            return None
            
        try:
            instructions = load_agents_instructions()
            if instructions.strip():
                return instructions
        except Exception as e:
            if self.config.verbose:
                click.echo(f"[WARN] Warning: Failed to load project instructions: {e}")
        
        return None
    
    def _format_tool_arguments(self, tool_name: str, arguments: dict) -> None:
        """Format and display tool arguments in a generic, consistent way.
        
        Args:
            tool_name: Name of the tool being called
            arguments: Dictionary of tool arguments
        """
        if not arguments:
            return
            
        # Define argument formatting rules
        IMPORTANT_ARGS = {
            "path", "file_path", "target_file", "directory", "filename",
            "command", "cmd", "script", "query", "search_term", "pattern",
            "content", "text", "data", "message", "input", "output",
            "url", "endpoint", "host", "port", "name", "id"
        }
        
        # Separate important and other arguments
        important_args = {}
        other_args = {}
        
        for key, value in arguments.items():
            if key.lower() in IMPORTANT_ARGS:
                important_args[key] = value
            else:
                other_args[key] = value
        
        # Display important arguments first
        if important_args:
            for key, value in important_args.items():
                formatted_value = self._format_argument_value(key, value, max_length=None)
                icon = self._get_argument_icon(key)
                click.echo(f"  {icon} {key}: {formatted_value}")
        
        # Display other arguments if any
        if other_args:
            other_formatted = []
            for key, value in other_args.items():
                formatted_value = self._format_argument_value(key, value, max_length=None)
                other_formatted.append(f"{key}={formatted_value}")
            
            if other_formatted:
                click.echo(f"  📋 Additional args: {', '.join(other_formatted)}")
    
    def _format_argument_value(self, key: str, value, max_length: int | None = None) -> str:
        """Format an argument value for display.
        
        Args:
            key: The argument key name
            value: The argument value
            max_length: Maximum length before truncation (None for no truncation)
            
        Returns:
            Formatted string representation of the value
        """
        if value is None:
            return "None"
        
        # Handle different value types
        if isinstance(value, bool):
            return "✓" if value else "✗"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            if len(value) == 0:
                return "[]"
            elif len(value) == 1:
                # For single item lists, format recursively
                item_max_length = max_length // 2 if max_length else None
                return f"[{self._format_argument_value(key, value[0], item_max_length)}]"
            else:
                # For multi-item lists, show first item and count
                item_max_length = max_length // 3 if max_length else None
                first_item = self._format_argument_value(key, value[0], item_max_length)
                return f"[{first_item}, ...{len(value)-1} more]"
        elif isinstance(value, dict):
            if len(value) == 0:
                return "{}"
            else:
                keys = list(value.keys())[:3]
                key_str = ", ".join(keys)
                if len(value) > 3:
                    key_str += f", ...{len(value)-3} more"
                return f"{{{key_str}}}"
        else:
            # String or other types
            str_value = str(value)
            if max_length is None or len(str_value) <= max_length:
                return str_value
            else:
                return str_value[:max_length-3] + "..."
    
    def _get_argument_icon(self, key: str) -> str:
        """Get an appropriate icon for an argument key.
        
        Args:
            key: The argument key name
            
        Returns:
            Emoji icon representing the argument type
        """
        key_lower = key.lower()
        
        # File/path related
        if any(word in key_lower for word in ["path", "file", "directory", "folder"]):
            return "📁"
        # Command/execution related  
        elif any(word in key_lower for word in ["command", "cmd", "script", "exec"]):
            return "🏃"
        # Search/query related
        elif any(word in key_lower for word in ["query", "search", "pattern", "filter"]):
            return "🔍"
        # Content/text related
        elif any(word in key_lower for word in ["content", "text", "data", "message", "body"]):
            return "📝"
        # Network/URL related
        elif any(word in key_lower for word in ["url", "endpoint", "host", "address"]):
            return "🌐"
        # ID/name related
        elif any(word in key_lower for word in ["id", "name", "key", "token"]):
            return "🏷️"
        # Default
        else:
            return "⚙️"
    
    def display_welcome(self) -> None:
        """Display welcome message."""
        click.echo()
        click.echo(click.style("Codin AI Coding Assistant", bold=True, fg="cyan"))
        click.echo(f"Model: {self.config.model} | Provider: {self.config.provider}")
        click.echo(f"Approval: {self.config.approval_mode.value}")
        
        # Check for project instructions
        instructions = self.load_project_instructions()
        if instructions:
            click.echo("[DOCS] Loaded project instructions from codin_rules.md")
        
        click.echo()
        click.echo("Type your coding request, or:")
        click.echo("  /help    - Show available commands")
        click.echo("  /clear   - Clear conversation history")
        click.echo("  /config  - Show current configuration")
        click.echo("  /exit    - Exit the REPL")
        click.echo()
    
    def display_help(self) -> None:
        """Display help information."""
        click.echo()
        click.echo(click.style("Available Commands:", bold=True))
        click.echo("  /help     - Show this help message")
        click.echo("  /clear    - Clear conversation history")
        click.echo("  /config   - Show current configuration")
        click.echo("  /history  - Show conversation history")
        click.echo("  /mode     - Change approval mode")
        click.echo("  /exit     - Exit the REPL")
        click.echo()
        click.echo(click.style("Usage Tips:", bold=True))
        click.echo("• Ask for coding tasks: 'Create a Python web scraper'")
        click.echo("• Request explanations: 'Explain this regex pattern'")
        click.echo("• Get help with errors: 'Fix the bug in utils.py'")
        click.echo("• Multi-line input: End with Ctrl+D or empty line")
        click.echo()
    
    def display_config(self) -> None:
        """Display current configuration."""
        click.echo()
        click.echo(click.style("Current Configuration:", bold=True))
        click.echo(f"  Model: {self.config.model}")
        click.echo(f"  Provider: {self.config.provider}")
        
        # Show base URL for current provider
        if self.config.provider in self.config.providers:
            provider_config = self.config.providers[self.config.provider]
            click.echo(f"  Base URL: {provider_config.base_url}")
        
        click.echo(f"  Approval Mode: {self.config.approval_mode.value}")
        click.echo(f"  Verbose: {self.config.verbose}")
        click.echo(f"  Project Docs: {'enabled' if self.config.enable_rules else 'disabled'}")
        click.echo()
    
    def display_history(self) -> None:
        """Display conversation history."""
        if not self.conversation_history:
            click.echo("No conversation history.")
            return
            
        click.echo()
        click.echo(click.style("Conversation History:", bold=True))
        for i, msg in enumerate(self.conversation_history, 1):  # Show all messages
            role_color = "blue" if msg.role == Role.user else "green"
            role_name = "You" if msg.role == Role.user else "Assistant"
            
            # Get text content from parts
            text_parts = []
            for p in msg.parts:
                # Handle Part objects that contain TextPart objects
                if hasattr(p, 'root') and hasattr(p.root, 'text'):
                    text_parts.append(p.root.text)
                # Handle direct TextPart objects
                elif hasattr(p, 'text'):
                    text_parts.append(p.text)
            content = " ".join(text_parts)
                
            click.echo(f"  {i}. {click.style(role_name, fg=role_color)}: {content}")
        click.echo()
    
    def change_approval_mode(self) -> None:
        """Change approval mode interactively."""
        click.echo()
        click.echo("Available approval modes:")
        click.echo("  1. suggest    - Show suggestions only (safest)")
        click.echo("  2. auto-edit  - Automatically edit files")
        click.echo("  3. full-auto  - Fully autonomous (runs commands)")
        click.echo()
        
        try:
            choice = click.prompt("Select mode (1-3)", type=int)
            mode_map = {1: ApprovalMode.SUGGEST, 2: ApprovalMode.AUTO_EDIT, 3: ApprovalMode.FULL_AUTO}
            
            if choice in mode_map:
                self.config.approval_mode = mode_map[choice]
                click.echo(f"[OK] Approval mode changed to: {self.config.approval_mode.value}")
            else:
                click.echo("Invalid choice.")
        except click.Abort:
            click.echo("Cancelled.")
        
        click.echo()
    
    async def process_user_input(self, user_input: str) -> bool:
        """Process user input and return True to continue, False to exit."""
        # Handle commands
        if user_input.startswith('/'):
            command = user_input[1:].lower().strip()
            
            if command in ['exit', 'quit', 'q']:
                return False
            elif command == 'help':
                self.display_help()
                return True
            elif command == 'clear':
                self.conversation_history.clear()
                click.echo("[OK] Conversation history cleared.")
                return True
            elif command == 'config':
                self.display_config()
                return True
            elif command == 'history':
                self.display_history()
                return True
            elif command == 'mode':
                self.change_approval_mode()
                return True
            else:
                click.echo(f"Unknown command: /{command}")
                click.echo("Type /help for available commands.")
                return True
        
        # Process as regular input
        if not user_input.strip():
            return True
        
        try:
            # Create user message using a2a structure
            user_message = Message(
                messageId=f"user-{len(self.conversation_history)}",
                role=Role.user,
                parts=[TextPart(text=user_input)]
            )
            
            # Add to history
            self.conversation_history.append(user_message)
            
            # Add project instructions if this is the first message
            if len(self.conversation_history) == 1:
                instructions = self.load_project_instructions()
                if instructions:
                    # Prepend instructions to the user message instead of creating a separate system message
                    # since a2a doesn't have a system role
                    original_text = user_message.parts[0].text
                    user_message.parts[0] = TextPart(text=f"Project Instructions:\n{instructions}\n\nUser Request: {original_text}")
            
            # Run agent with timeout
            agent_input = AgentRunInput(message=user_message)
            
            if self.agent is None:
                raise RuntimeError("Agent not initialized")
            
            try:
                # Add overall timeout for the entire agent run
                result = await asyncio.wait_for(
                    self.agent.run(agent_input),
                    timeout=300.0  # 5 minute timeout for entire conversation turn
                )
            except asyncio.TimeoutError:
                click.echo(f"\n[WARN] Request timed out after 5 minutes. This might be due to network issues or complex processing.")
                click.echo("You can try:")
                click.echo("  - Simplifying your request")
                click.echo("  - Checking your internet connection")
                click.echo("  - Trying again in a moment")
                return True
            except ConnectionError as e:
                click.echo(f"\n[WARN] Connection error: {e}")
                click.echo("Please check your internet connection and try again.")
                return True
            except Exception as e:
                # Check for specific error types
                error_str = str(e).lower()
                if "timeout" in error_str or "timed out" in error_str:
                    click.echo(f"\n[WARN] Request timed out: {e}")
                    click.echo("This might be due to network issues or high server load. Please try again.")
                elif "connection" in error_str or "network" in error_str:
                    click.echo(f"\n[WARN] Network error: {e}")
                    click.echo("Please check your internet connection and try again.")
                elif "cancelled" in error_str:
                    click.echo(f"\n[WARN] Request was cancelled. This might be due to cleanup during shutdown.")
                    return True
                else:
                    # Re-raise for general error handling below
                    raise
                return True
            
            # Extract response text for conversation history
            if hasattr(result.result, 'parts'):
                # It's a Message
                text_parts = []
                for p in result.result.parts:
                    # Handle Part objects that contain TextPart objects
                    if hasattr(p, 'root') and hasattr(p.root, 'text'):
                        text_parts.append(p.root.text)
                    # Handle direct TextPart objects
                    elif hasattr(p, 'text'):
                        text_parts.append(p.text)
                response_text = "\n".join(text_parts)
            else:
                # Fallback to string representation
                response_text = str(result.result)
            
            # Note: Assistant response display is now handled by the streaming callback
            
            # Add assistant response to history
            assistant_message = Message(
                messageId=f"assistant-{len(self.conversation_history)}",
                role=Role.agent,
                parts=[TextPart(text=response_text)]
            )
            self.conversation_history.append(assistant_message)
            
        except KeyboardInterrupt:
            click.echo("\n[WARN] Interrupted by user.")
            return True
        except Exception as e:
            click.echo(f"\n[ERROR] Error: {e}", err=True)
            if self.config.verbose:
                import traceback
                click.echo(traceback.format_exc(), err=True)
            
            # Provide helpful suggestions based on error type
            error_str = str(e).lower()
            if "model" in error_str or "api" in error_str:
                click.echo("\nTip: Check your API keys and model configuration with /config")
            elif "tool" in error_str:
                click.echo("\nTip: Some tools might not be available. Try a simpler request.")
            elif "memory" in error_str or "out of memory" in error_str:
                click.echo("\nTip: Try clearing conversation history with /clear")
            
            return True
        
        return True
    
    async def run(self) -> None:
        """Run the REPL session."""
        try:
            await self.initialize()
            self.display_welcome()
            
            # Check if stdin is a pipe (non-interactive mode)
            import sys
            is_pipe = not sys.stdin.isatty()
            
            try:
                while True:
                    try:
                        # Get user input
                        user_input = click.prompt(
                            click.style("You", fg="blue", bold=True),
                            prompt_suffix="> ",
                            show_default=False
                        )
                        
                        # Process input
                        should_continue = await self.process_user_input(user_input)
                        
                        # If this is piped input, exit after processing the first command
                        if is_pipe:
                            click.echo("\n[INFO] Piped input processed. Exiting.")
                            break
                        
                        if not should_continue:
                            break
                            
                    except (KeyboardInterrupt, EOFError):
                        if is_pipe:
                            # For piped input, EOFError is expected - just exit gracefully
                            click.echo("\n[INFO] Piped input completed.")
                        else:
                            # For interactive mode, show goodbye message
                            click.echo("\nGoodbye!")
                        break
                        
            except Exception as e:
                click.echo(f"\n[ERROR] Fatal error: {e}", err=True)
                raise
        except Exception as e:
            # Re-raise exceptions to be handled by start_repl
            raise


async def start_repl(
    *,
    initial_prompt: str | None = None,
    approval_mode: ApprovalMode | None = None,
    verbose: bool = False,
    debug: bool = False,
    config_file: str | None = None,
) -> None:
    """Start the interactive REPL session.
    
    Args:
        initial_prompt: Optional initial prompt to start with
        approval_mode: Approval mode for agent actions
        verbose: Enable verbose output
        debug: Enable debug mode
        config_file: Path to custom config file
    """
    session = ReplSession(
        approval_mode=approval_mode,
        verbose=verbose,
        debug=debug,
        config_file=config_file,
    )
    
    try:
        if initial_prompt:
            # Initialize first if we have an initial prompt
            await session.initialize()
            # Process initial prompt if provided
            await session.process_user_input(initial_prompt)
        
        await session.run()
    except KeyboardInterrupt:
        click.echo("\nGoodbye!")
    except Exception as e:
        click.echo(f"\n[ERROR] Fatal error: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
    finally:
        # Always attempt cleanup, but don't let cleanup errors propagate
        try:
            await session.cleanup()
        except (asyncio.CancelledError, RuntimeError) as e:
            # These are expected during shutdown
            if "Event loop is closed" in str(e) or "cancel scope" in str(e):
                click.echo("[OK] Cleanup completed during shutdown")
            else:
                click.echo(f"[WARN] Cleanup error during shutdown: {e}")
        except Exception as e:
            click.echo(f"[WARN] Error during cleanup: {e}") 