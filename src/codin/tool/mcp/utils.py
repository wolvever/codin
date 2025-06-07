"""Utility functions for MCP integration.

This module provides helper functions and utilities for working with
MCP servers and handling connection management.
"""

import functools
import logging

__all__ = [
    'retry_on_closed_resource',
]

_logger = logging.getLogger(__name__)


def retry_on_closed_resource(async_reinit_func_name: str):
    """Decorator to retry an operation when a resource is closed.

    When an MCP session is closed (e.g., by timeout, server restart, etc.),
    this decorator will automatically attempt to reinitialize the session
    and retry the operation.

    Parameters
    ----------
    async_reinit_func_name:
        Name of the method that will reinitialize the session. This method
        must be an async method of the same class as the decorated method.

    Returns:
    -------
    callable:
        Decorated function that will retry once after reinitialization.

    Example:
    -------
    ```python
    class MCPTool:
        async def _reinitialize_session(self):
            self._session = await self._session_manager.create_session()

        @retry_on_closed_resource('_reinitialize_session')
        async def run(self, args, context):
            # Use the session, which might be closed
            ...
    ```
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                # Check for common "closed resource" errors from different libraries
                is_closed_error = (
                    'ClosedResourceError' in e.__class__.__name__
                    or 'ConnectionClosed' in e.__class__.__name__
                    or 'not connected' in str(e).lower()
                    or 'connection closed' in str(e).lower()
                )

                if not is_closed_error:
                    # Not a connection closure error, re-raise
                    raise

                _logger.info('Detected closed connection. Attempting to reinitialize session.')

                # Try to reinitialize the session
                try:
                    if hasattr(self, async_reinit_func_name) and callable(getattr(self, async_reinit_func_name)):
                        reinit_func = getattr(self, async_reinit_func_name)
                        await reinit_func()
                    else:
                        raise ValueError(
                            f'Function {async_reinit_func_name} not found in class '
                            f'{self.__class__.__name__}. Check the function name '
                            'used with retry_on_closed_resource decorator.'
                        )
                except Exception as reinit_err:
                    _logger.error(f'Failed to reinitialize session: {reinit_err}')
                    raise RuntimeError(f'Failed to reinitialize after connection closed: {reinit_err}') from e

                # Retry the operation once
                _logger.info('Session reinitialized. Retrying operation.')
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator
