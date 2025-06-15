"""Approval extension for tool execution."""

from __future__ import annotations

import asyncio
from typing import Callable, Awaitable

from ...config import ApprovalMode
from .base import Extension, ExtensionContext

__all__ = ['ApprovalExtension', 'ApprovalHandler']


ApprovalHandler = Callable[[str, str, dict], Awaitable[bool]]


class ApprovalExtension(Extension):
    """Extension that handles user approval for tool execution."""
    
    def __init__(
        self, 
        approval_mode: ApprovalMode = ApprovalMode.NEVER,
        approval_handler: ApprovalHandler | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.approval_mode = approval_mode
        self.approval_handler = approval_handler or self._default_approval_handler
        self._dangerous_patterns = {
            'delete', 'remove', 'rm', 'kill', 'terminate', 'destroy', 'format',
            'drop', 'truncate', 'clear', 'reset', 'wipe', 'purge'
        }
    
    @property
    def name(self) -> str:
        return "approval"
    
    async def before_execute(self, ctx: ExtensionContext) -> None:
        """Check if approval is needed before execution."""
        if self.approval_mode == ApprovalMode.NEVER:
            return
        
        needs_approval = False
        
        if self.approval_mode == ApprovalMode.ALWAYS:
            needs_approval = True
        elif self.approval_mode == ApprovalMode.UNSAFE_ONLY:
            needs_approval = (
                ctx.spec.metadata.is_dangerous or
                ctx.spec.metadata.requires_approval or
                self._is_potentially_dangerous(ctx.spec.name, ctx.args)
            )
        
        if needs_approval:
            approved = await self._request_approval(ctx)
            if not approved:
                raise PermissionError(f"Tool execution not approved: {ctx.spec.name}")
    
    async def _request_approval(self, ctx: ExtensionContext) -> bool:
        """Request approval from user."""
        tool_name = ctx.spec.name
        description = ctx.spec.description
        args = ctx.args
        
        return await self.approval_handler(tool_name, description, args)
    
    async def _default_approval_handler(
        self, 
        tool_name: str, 
        description: str, 
        args: dict
    ) -> bool:
        """Default approval handler that prompts via stdin/stdout."""
        print(f"\n🔒 Approval Required")
        print(f"Tool: {tool_name}")
        print(f"Description: {description}")
        print(f"Arguments: {args}")
        
        while True:
            response = input("Approve execution? (y/n): ").strip().lower()
            if response in ('y', 'yes'):
                return True
            elif response in ('n', 'no'):
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def _is_potentially_dangerous(self, tool_name: str, args: dict) -> bool:
        """Check if tool or arguments suggest dangerous operation."""
        # Check tool name
        tool_name_lower = tool_name.lower()
        if any(pattern in tool_name_lower for pattern in self._dangerous_patterns):
            return True
        
        # Check arguments
        args_str = str(args).lower()
        if any(pattern in args_str for pattern in self._dangerous_patterns):
            return True
        
        # Check for file system operations on system paths
        dangerous_paths = {'/etc', '/usr', '/bin', '/sbin', '/boot', '/sys', '/proc'}
        for value in args.values():
            if isinstance(value, str):
                for path in dangerous_paths:
                    if value.startswith(path):
                        return True
        
        return False