"""CLI commands for codin.

This module defines the command-line interface for codin, providing codex-compatible
commands including interactive REPL mode, quiet mode, and comprehensive provider support.
"""

from __future__ import annotations

import asyncio
import os
import sys
import typing as _t
from pathlib import Path

import click
from dotenv import load_dotenv

from ..config import (
    get_config, 
    load_agents_instructions, 
    find_config_files,
    get_api_key,
    get_default_providers,
)
from ..agent.code_agent import CodeAgent
from ..model.factory import LLMFactory
from a2a.types import Message, Role, TextPart
from ..agent.base import AgentRunInput
from ..sandbox import LocalSandbox
from ..tool.base import ApprovalMode, Toolset
from ..tool.mcp import MCPToolset, StdioServerParams, SseServerParams, HttpServerParams
from ..tool.sandbox import SandboxToolset
from ..tool.core_tools import RunShellTool
from ..tool.registry import ToolRegistry
from .repl import start_repl
from .utils import create_mcp_toolsets_from_config

__all__ = [
    "cli",
    "main",
]

# Load environment variables
load_dotenv()


def validate_provider(provider: str) -> str:
    """Validate and return provider name."""
    providers = get_default_providers()
    if provider not in providers:
        available = ", ".join(providers.keys())
        raise click.BadParameter(f"Unknown provider '{provider}'. Available: {available}")
    return provider


def validate_approval_mode(mode: str) -> ApprovalMode:
    """Validate and return approval mode."""
    try:
        return ApprovalMode(mode)
    except ValueError:
        available = ", ".join([m.value for m in ApprovalMode])
        raise click.BadParameter(f"Unknown approval mode '{mode}'. Available: {available}")


async def run_quiet_mode(
    prompt: str,
    *,
    approval_mode: ApprovalMode | None = None,
    verbose: bool = False,
    debug: bool = False,
    config_file: str | None = None,
) -> None:
    """Run in quiet mode (non-interactive)."""
    config = get_config(config_file)
    
    # Override with parameters
    if approval_mode:
        config.approval_mode = approval_mode
    
    try:
        # Create and initialize sandbox
        sandbox = LocalSandbox()
        await sandbox.up()
        
        if verbose:
            click.echo(f"[OK] Initialized sandbox: {type(sandbox).__name__}")
        
        toolsets = []
        # Create sandbox toolset (auto-generates tools from sandbox methods)
        from codin.tool.sandbox import SandboxToolset
        sandbox_toolset = SandboxToolset(sandbox)
        toolsets.append(sandbox_toolset)
        
        # Create toolsets from config
        toolsets.extend(create_mcp_toolsets_from_config(config_file))
        
        # Initialize all MCP toolsets (skip sandbox toolset - already initialized)
        for toolset in toolsets[:-1]:
            try:
                await toolset.up()
                if verbose:
                    click.echo(f"[OK] Initialized toolset: {toolset.name}")
            except Exception as e:
                click.echo(f"[WARN] Failed to initialize toolset {toolset.name}: {e}", err=True)
        
        # Create agent with initialized sandbox and toolsets
        agent = CodeAgent(
            name="Codin Assistant",
            description="AI coding assistant with tool-calling capabilities",
            llm_model=config.model,
            sandbox=sandbox,
            toolsets=toolsets,
            approval_mode=approval_mode or ApprovalMode.ALWAYS,
            debug=debug,
        )
        
        # Add debug event callback for quiet mode if debug is enabled
        if debug:
            async def debug_event_callback(event):
                if event.event_type == "debug_llm_response":
                    debug_info = event.data
                    print(f"🤖 LLM Response (Turn {debug_info['turn_count']}):")
                    print("-" * 60)
                    print(f"📄 Raw content length: {debug_info['raw_content_length']} characters")
                    
                    thinking = debug_info.get('thinking')
                    if thinking:
                        print(f"💭 Thinking: {thinking}")
                    else:
                        print("💭 Thinking: None")
                    
                    message = debug_info.get('message')
                    if message:
                        print(f"💬 Message: {message}")
                    else:
                        print("💬 Message: None")
                    
                    print(f"🔄 Should continue: {debug_info['should_continue']}")
                    
                    task_list = debug_info['task_list']
                    print(f"📋 Task list - Completed: {task_list['completed_count']}, Pending: {task_list['pending_count']}")
                    
                    tool_calls = debug_info.get('tool_calls', [])
                    if tool_calls:
                        print(f"🔧 Tool calls: {len(tool_calls)}")
                        for i, tool_call in enumerate(tool_calls):
                            args_keys = tool_call.get('arguments_keys', [])
                            print(f"  {i+1}. {tool_call['name']}({args_keys})")
                    else:
                        print("🔧 Tool calls: None")
                    
                    print("-" * 60 + "\n")
            
            agent.add_event_callback(debug_event_callback)
        
        if verbose:
            click.echo(f"Using model: {config.model} (provider: {config.provider})")
        
        # Load project instructions
        instructions = load_agents_instructions()
        if instructions and verbose:
            click.echo("[DOCS] Loaded project instructions from codin_rules.md")
        
        # Create user message using a2a structure
        user_message_text = prompt
        if instructions:
            # Prepend instructions to the user message since a2a doesn't have a system role
            user_message_text = f"Project Instructions:\n{instructions}\n\nUser Request: {prompt}"
        
        user_message = Message(
            messageId="user-input",
            role=Role.user,
            parts=[TextPart(text=user_message_text)]
        )
        
        # Run agent
        agent_input = AgentRunInput(message=user_message)
        
        if not config.quiet_mode:
            click.echo("Processing...")
        
        result = await agent.run(agent_input)
        
        # Extract and display response
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
        
        if not config.quiet_mode:
            click.echo("\n" + click.style("Result:", bold=True, fg="green"))
        
        click.echo(response_text)
        
    except Exception as e:
        click.echo(f"[ERROR] Error: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


def format_tool_signature(tool) -> str:
    """Format a tool's signature like a Python function with arguments and datatypes."""
    try:
        tool_def = tool.to_tool_definition()
        if not hasattr(tool_def, 'parameters') or not tool_def.parameters:
            return f"{tool.name}()"
        
        properties = tool_def.parameters.get('properties', {})
        required = tool_def.parameters.get('required', [])
        
        if not properties:
            return f"{tool.name}()"
        
        params = []
        for param_name, param_info in properties.items():
            # Handle different schema formats
            python_type = _extract_python_type(param_info)
            
            # Check if parameter is optional
            if param_name in required:
                param_str = f"{param_name}: {python_type}"
            else:
                param_str = f"{param_name}: {python_type} = None"
            
            params.append(param_str)
        
        return f"{tool.name}({', '.join(params)})"
    
    except Exception:
        # Fallback if we can't parse the schema
        return f"{tool.name}(...)"


def _extract_python_type(param_info: dict) -> str:
    """Extract Python type from JSON Schema parameter info."""
    # Handle anyOf (union types, often used for Optional)
    if 'anyOf' in param_info:
        types = []
        for any_type in param_info['anyOf']:
            if any_type.get('type') == 'null':
                continue  # Skip null type, we'll handle optionality separately
            types.append(_extract_python_type(any_type))
        if len(types) == 1:
            return types[0]
        elif len(types) > 1:
            return f"Union[{', '.join(types)}]"
    
    # Handle allOf (intersection types)
    if 'allOf' in param_info:
        # For allOf, we'll just take the first non-null type
        for all_type in param_info['allOf']:
            if all_type.get('type') != 'null':
                return _extract_python_type(all_type)
    
    # Get the basic type
    param_type = param_info.get('type', 'Any')
    
    # Map JSON Schema types to Python types
    type_mapping = {
        'string': 'str',
        'integer': 'int', 
        'number': 'float',
        'boolean': 'bool',
        'array': 'list',
        'object': 'dict',
        'null': 'None'
    }
    
    python_type = type_mapping.get(param_type, param_type)
    
    # Handle array types with items
    if param_type == 'array' and 'items' in param_info:
        item_type = _extract_python_type(param_info['items'])
        python_type = f"list[{item_type}]"
    
    # Handle object types with additional properties
    elif param_type == 'object':
        if 'additionalProperties' in param_info:
            if param_info['additionalProperties'] is True:
                python_type = "dict[str, Any]"
            elif isinstance(param_info['additionalProperties'], dict):
                value_type = _extract_python_type(param_info['additionalProperties'])
                python_type = f"dict[str, {value_type}]"
        elif 'properties' in param_info:
            # This is a structured object, but for simplicity we'll call it dict
            python_type = "dict"
    
    return python_type


# --- CLI Command Groups ---

@click.command()
@click.option("--version", is_flag=True, help="Show version and exit")
@click.option(
    "--approval-mode",
    type=click.Choice([mode.value for mode in ApprovalMode]),
    envvar="CODIN_APPROVAL_MODE",
    help="Approval mode for agent actions"
)
@click.option("-q", "--quiet", is_flag=True, help="Run in quiet mode (single execution, no REPL)")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("--debug", is_flag=True, help="Enable debug mode (shows LLM requests and responses)")
@click.option("--config", "show_config", is_flag=True, help="Show configuration and exit")
@click.option("--config-file", type=click.Path(exists=True, path_type=Path), help="Specify custom config file path")
@click.option("--tools", "show_tools", is_flag=True, help="Show available tools and exit")
@click.option("--providers", "show_providers", is_flag=True, help="Show available providers and exit")
@click.argument("prompt", required=False)
def cli(
    version: bool,
    approval_mode: str | None,
    quiet: bool,
    verbose: bool,
    debug: bool,
    show_config: bool,
    config_file: Path | None,
    show_tools: bool,
    show_providers: bool,
    prompt: str | None,
) -> None:
    """Codin - AI-powered coding assistant with tool execution capabilities.

    \b
    USAGE MODES:
      Interactive:  codin                           # Start REPL session
      Interactive:  codin "Create a web scraper"    # Start REPL with initial prompt
      Quiet:        codin -q "Fix this bug"         # Single execution, no REPL

    \b
    UTILITY OPTIONS:
      --config      Show current configuration
      --tools       Show available tools
      --providers   Show available providers

    \b
    COMMON OPTIONS:
      --approval-mode MODE  Set approval policy (suggest, auto_edit, full_auto)
      -v, --verbose         Enable detailed output
      -q, --quiet           Single execution mode (no REPL)
      --config-file FILE    Use custom config file instead of defaults
      --debug               Enable debug mode (shows LLM requests and responses)

    \b
    ENVIRONMENT VARIABLES:
      Model & Provider:
        LLM_MODEL                        Default model name
        LLM_PROVIDER                     Default provider
        CODIN_APPROVAL_MODE              Default approval mode

      Behavior:
        CODIN_QUIET_MODE                 Enable quiet mode by default
        CODIN_RULE                       Enable codin_rules.md loading

    \b
    EXAMPLES:
      codin                                         # Interactive mode
      codin "Write a Python script to parse CSV"   # Interactive with initial prompt
      codin -q "create a function"                  # Quiet mode
      codin -v "help me debug this"                 # Verbose interactive
      codin --config                                # Show configuration
      codin --config-file ./my-config.yaml         # Use custom config file
      codin --tools                                 # Show available tools
    """
    # Handle utility flags first
    if version:
        from codin import version as codin_version
        click.echo(f"codin {codin_version}")
        return
    
    if show_config:
        # Import and call the config function logic
        show_config_info(config_file=config_file)
        return
    
    if show_tools:
        # Show available tools
        asyncio.run(show_tools_info(verbose=verbose, config_file=config_file))
        return
    
    if show_providers:
        show_providers_info()
        return
    
    # Parse approval mode
    approval_enum = None
    if approval_mode:
        approval_enum = validate_approval_mode(approval_mode)
    
    if quiet:
        # Quiet mode - single execution
        if not prompt:
            click.echo("Error: Prompt required in quiet mode. Use 'codin -q \"your prompt\"'")
            return
        
        asyncio.run(run_quiet_mode(
            prompt,
            approval_mode=approval_enum,
            verbose=verbose,
            debug=debug,
            config_file=config_file.as_posix() if config_file else None,
        ))
    else:
        # Interactive mode (REPL)
        asyncio.run(start_repl(
            initial_prompt=prompt,
            approval_mode=approval_enum,
            verbose=verbose,
            debug=debug,
            config_file=config_file.as_posix() if config_file else None,
        ))


def show_config_info(config_file: Path | None = None) -> None:
    """Show current configuration and settings."""
    config_obj = get_config(config_file)
    
    click.echo()
    click.echo(click.style("Codin Configuration", bold=True, fg="cyan"))
    click.echo("=" * 50)
    
    # Current settings
    click.echo(click.style("Current Settings:", bold=True))
    click.echo(f"  Model: {config_obj.model}")
    click.echo(f"  Provider: {config_obj.provider}")
    click.echo(f"  Approval Mode: {config_obj.approval_mode.value}")
    click.echo(f"  Project Docs: {'enabled' if config_obj.enable_rules else 'disabled'}")
    click.echo(f"  Debug: {config_obj.debug}")
    click.echo()
    
    # Provider information
    if config_file and config_file.stem in config_obj.providers:
        provider_config = config_obj.providers[config_file.stem]
        click.echo(click.style(f"Provider: {provider_config.name}", bold=True))
        click.echo(f"  Base URL: {provider_config.base_url}")
        click.echo(f"  Environment Variable: {provider_config.env_key}")
        api_key_status = "🟢 SET" if get_api_key(config_file.stem) else "🔴 NOT SET"
        click.echo(f"  API Key: {api_key_status}")
        if provider_config.models:
            click.echo(f"  Supported Models: {', '.join(provider_config.models[:5])}")
            if len(provider_config.models) > 5:
                click.echo(f"    ...and {len(provider_config.models) - 5} more")
    elif not config_file:
        # All providers
        click.echo(click.style("Available Providers:", bold=True))
        for name, provider_config in config_obj.providers.items():
            status = "[SET]" if get_api_key(name) else "[NOT SET]"
            status_icon = "🟢" if get_api_key(name) else "🔴"
            click.echo(f"  {status_icon} {name:<12} {provider_config.name}")
    
    click.echo()
    
    # MCP servers information
    if config_obj.mcp_servers:
        click.echo(click.style("MCP Servers:", bold=True))
        for name, mcp_config in config_obj.mcp_servers.items():
            if mcp_config.url:
                # SSE server
                click.echo(f"  {name:<12} [SSE] {mcp_config.description}")
                click.echo(f"    URL: {mcp_config.url}")
            else:
                # Stdio server
                click.echo(f"  {name:<12} [STDIO] {mcp_config.description}")
                click.echo(f"    Command: {mcp_config.command}")
                if mcp_config.args:
                    click.echo(f"    Args: {' '.join(mcp_config.args)}")
                if mcp_config.env:
                    click.echo(f"    Environment: {', '.join(f'{k}={v}' for k, v in mcp_config.env.items())}")
        click.echo()
    else:
        click.echo(click.style("MCP Servers:", bold=True))
        click.echo("  No MCP servers configured")
        click.echo()
    
    # Configuration files
    config_dir = Path.home() / ".codin"
    click.echo(click.style("Configuration:", bold=True))
    click.echo(f"  Config directory: {config_dir}")
    
    # Show which config files exist (using the same logic as config loading)
    config_files = find_config_files(config_file)
    
    if config_files:
        click.echo(f"  Config files: {', '.join(str(f) for f in config_files)}")
    else:
        click.echo(f"  Config files: None found")
    
    # codin_rules.md files
    agents_instructions = load_agents_instructions()
    if agents_instructions:
        click.echo(f"  codin_rules.md: Found and loaded")
    else:
        click.echo(f"  codin_rules.md: Not found")
    
    click.echo()


def show_providers_info() -> None:
    """List all available LLM providers and their configuration status."""
    config_obj = get_config()
    
    click.echo()
    click.echo(click.style("Available Providers", bold=True, fg="cyan"))
    click.echo("=" * 50)
    
    for name, provider_config in config_obj.providers.items():
        api_key = get_api_key(name)
        status = click.style("🟢 CONFIGURED", fg="green") if api_key else click.style("🔴 NO API KEY", fg="red")
        
        click.echo()
        click.echo(click.style(f"{provider_config.name} ({name})", bold=True))
        click.echo(f"  Status: {status}")
        click.echo(f"  Base URL: {provider_config.base_url}")
        click.echo(f"  Environment Variable: {provider_config.env_key}")
        
        if provider_config.models:
            models_display = ", ".join(provider_config.models[:3])
            if len(provider_config.models) > 3:
                models_display += f" (+{len(provider_config.models) - 3} more)"
            click.echo(f"  Example Models: {models_display}")
    
    click.echo()


async def show_tools_info(verbose: bool = False, config_file: Path | None = None) -> None:
    """List available tools and their status."""
    from codin.agent.code_agent import CodeAgent
    from codin.sandbox import LocalSandbox
    from .utils import create_mcp_toolsets_from_config
    
    toolsets = []
    try:
        # Create and initialize sandbox
        sandbox = LocalSandbox()
        await sandbox.up()
        
        if verbose:
            click.echo(f"[OK] Initialized sandbox: {type(sandbox).__name__}")
        
        # Create sandbox toolset (auto-generates tools from sandbox methods)
        from codin.tool.sandbox import SandboxToolset
        sandbox_toolset = SandboxToolset(sandbox)
        await sandbox_toolset.up()
        
        # Create toolsets from config
        try:
            mcp_toolsets = create_mcp_toolsets_from_config(config_file.as_posix() if config_file else None)
        except Exception as e:
            click.echo(f"[ERROR] Failed to create toolsets: {e}", err=True)
            return
        
        toolsets.append(sandbox_toolset)
        toolsets.extend(mcp_toolsets)
        
        # Initialize all MCP toolsets
        for toolset in mcp_toolsets:
            try:
                await toolset.up()
                if verbose:
                    click.echo(f"[OK] Initialized toolset: {toolset.name}")
            except Exception as e:
                if verbose:
                    click.echo(f"[WARN] Failed to initialize toolset {toolset.name}: {e}", err=True)
        
        # Display tools
        click.echo()
        click.echo(click.style("Available Tools", bold=True, fg="cyan"))
        click.echo("=" * 50)
        
        for toolset in toolsets:
            click.echo()
            click.echo(click.style(f"Toolset: {toolset.name}", bold=True))
            
            if hasattr(toolset, 'tools') and toolset.tools:
                for tool in toolset.tools:
                    if verbose:
                        signature = format_tool_signature(tool)
                        click.echo(f"  🔧 {signature}")
                        if hasattr(tool, 'description') and tool.description:
                            click.echo(f"     {tool.description}")
                    else:
                        click.echo(f"  🔧 {tool.name}")
            else:
                click.echo("  No tools available")
        
        click.echo()
        
    except Exception as e:
        click.echo(f"[ERROR] Failed to initialize tools: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
    finally:
        # Cleanup
        if toolsets:
            for toolset in toolsets:
                try:
                    await toolset.down()
                except Exception:
                    pass  # Ignore cleanup errors
        
        if 'sandbox' in locals():
            try:
                await sandbox.down()
            except Exception:
                pass  # Ignore cleanup errors


def main() -> None:
    """Main entry point for the CLI."""
    import sys
    import os
    import warnings
    import atexit
    
    # Suppress specific asyncio warnings that occur during subprocess cleanup on Windows
    warnings.filterwarnings("ignore", category=ResourceWarning, module="asyncio")
    warnings.filterwarnings("ignore", message=".*unclosed transport.*")
    warnings.filterwarnings("ignore", message=".*I/O operation on closed pipe.*")
    
    # Override the warning function to suppress subprocess cleanup warnings
    original_warn = warnings._warn_unawaited_coroutine if hasattr(warnings, '_warn_unawaited_coroutine') else None
    
    def suppress_subprocess_warnings(*args, **kwargs):
        """Suppress subprocess-related warnings."""
        pass
    
    # Set up aggressive cleanup suppression for the entire process
    def suppress_cleanup_warnings():
        """Suppress all warnings during process cleanup."""
        try:
            # Redirect stderr to devnull to suppress "Exception ignored" messages
            sys.stderr = open(os.devnull, 'w')
            warnings.filterwarnings("ignore")
            # Override the _warn function used by asyncio
            if hasattr(warnings, '_warn_unawaited_coroutine'):
                warnings._warn_unawaited_coroutine = suppress_subprocess_warnings
        except:
            pass
    
    # Register cleanup suppression to run at exit
    atexit.register(suppress_cleanup_warnings)
    
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n[INTERRUPTED] Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        click.echo(f"[FATAL ERROR] Fatal error: {e}", err=True)
        sys.exit(1)
    finally:
        # Force garbage collection and suppress any warnings
        try:
            import gc
            warnings.filterwarnings("ignore")
            gc.collect()
            # Give a moment for cleanup
            import time
            time.sleep(0.05)
        except:
            pass


if __name__ == "__main__":
    main() 