"""Authentication extension for tool execution."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Awaitable

from .base import Extension, ExtensionContext

__all__ = ['AuthExtension', 'AuthHandler', 'AuthPolicy']


AuthHandler = Callable[[str, str, dict], Awaitable[bool]]


class AuthPolicy:
    """Authentication policy for tools."""
    
    def __init__(
        self,
        required_roles: list[str] | None = None,
        required_permissions: list[str] | None = None,
        allowed_users: list[str] | None = None,
        denied_users: list[str] | None = None,
        time_restrictions: dict[str, Any] | None = None,
    ):
        self.required_roles = required_roles or []
        self.required_permissions = required_permissions or []
        self.allowed_users = allowed_users
        self.denied_users = denied_users or []
        self.time_restrictions = time_restrictions or {}
    
    def check_user(self, user_id: str) -> bool:
        """Check if user is allowed."""
        if user_id in self.denied_users:
            return False
        
        if self.allowed_users is not None:
            return user_id in self.allowed_users
        
        return True
    
    def check_time(self) -> bool:
        """Check if current time is within allowed restrictions."""
        if not self.time_restrictions:
            return True
        
        current_hour = time.localtime().tm_hour
        
        start_hour = self.time_restrictions.get('start_hour')
        end_hour = self.time_restrictions.get('end_hour')
        
        if start_hour is not None and end_hour is not None:
            if start_hour <= end_hour:
                return start_hour <= current_hour <= end_hour
            else:  # Crosses midnight
                return current_hour >= start_hour or current_hour <= end_hour
        
        return True


class AuthExtension(Extension):
    """Extension that handles authentication and authorization for tool execution."""
    
    def __init__(
        self,
        auth_handler: AuthHandler | None = None,
        default_policy: AuthPolicy | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.auth_handler = auth_handler or self._default_auth_handler
        self.default_policy = default_policy or AuthPolicy()
        self._tool_policies: dict[str, AuthPolicy] = {}
        self._user_cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes
    
    @property
    def name(self) -> str:
        return "auth"
    
    def set_tool_policy(self, tool_name: str, policy: AuthPolicy) -> None:
        """Set authentication policy for a specific tool."""
        self._tool_policies[tool_name] = policy
    
    def set_user_context(self, user_id: str, context: dict[str, Any]) -> None:
        """Set user context (roles, permissions, etc.)."""
        self._user_cache[user_id] = {
            'context': context,
            'timestamp': time.time()
        }
    
    async def before_execute(self, ctx: ExtensionContext) -> None:
        """Check authentication before tool execution."""
        user_id = ctx.tool_context.metadata.get('user_id')
        if not user_id:
            # No user context, skip auth (could be system call)
            return
        
        # Get policy for this tool
        policy = self._tool_policies.get(ctx.spec.name, self.default_policy)
        
        # Check basic user access
        if not policy.check_user(user_id):
            raise PermissionError(f"User {user_id} is not authorized to use tool {ctx.spec.name}")
        
        # Check time restrictions
        if not policy.check_time():
            raise PermissionError(f"Tool {ctx.spec.name} is not available at this time")
        
        # Get user context
        user_context = await self._get_user_context(user_id)
        
        # Check roles
        if policy.required_roles:
            user_roles = user_context.get('roles', [])
            if not any(role in user_roles for role in policy.required_roles):
                raise PermissionError(
                    f"Tool {ctx.spec.name} requires one of roles: {policy.required_roles}"
                )
        
        # Check permissions
        if policy.required_permissions:
            user_permissions = user_context.get('permissions', [])
            missing_perms = set(policy.required_permissions) - set(user_permissions)
            if missing_perms:
                raise PermissionError(
                    f"Tool {ctx.spec.name} requires permissions: {list(missing_perms)}"
                )
        
        # Custom auth handler
        if not await self.auth_handler(user_id, ctx.spec.name, ctx.args):
            raise PermissionError(f"Custom authentication failed for tool {ctx.spec.name}")
        
        # Record authorization in metadata
        ctx.set_metadata('auth_user_id', user_id)
        ctx.set_metadata('auth_timestamp', time.time())
        ctx.set_metadata('auth_policy_hash', self._hash_policy(policy))
    
    async def _get_user_context(self, user_id: str) -> dict[str, Any]:
        """Get user context from cache or fetch it."""
        cached = self._user_cache.get(user_id)
        
        if cached and (time.time() - cached['timestamp']) < self._cache_ttl:
            return cached['context']
        
        # Cache expired or not found, use empty context
        # In real implementation, this would fetch from user service
        return {}
    
    async def _default_auth_handler(
        self, 
        user_id: str, 
        tool_name: str, 
        args: dict
    ) -> bool:
        """Default auth handler that always allows."""
        return True
    
    def _hash_policy(self, policy: AuthPolicy) -> str:
        """Create hash of policy for audit trails."""
        policy_str = f"{policy.required_roles}{policy.required_permissions}{policy.allowed_users}"
        return hashlib.md5(policy_str.encode()).hexdigest()[:8]