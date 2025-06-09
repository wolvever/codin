"""Codin Platform core package.

This package exposes high-level helpers for building agents, tools and runtimes
that interoperate via the A2A protocol. It also provides core modules for
actor systems, agent definitions, and A2A specific types.
"""

import sys as _sys

# Import submodules or specific types to make them available at the 'codin' level
from . import config
from . import agent
from . import actor
from . import a2a # Added new a2a submodule
# from . import tool # Example if tool was a submodule
# from . import memory # Example if memory was a submodule

# Expose specific functions or classes directly from codin
# For example, if these were defined in config.py:
# from .config import get_api_key, get_config, load_config

# Utility functions - keep them as they are if they are meant to be top-level
# Consider moving them into a 'utils' submodule if not already there and re-exporting.
# For now, assuming their current definition style is intentional.
def extract_text_from_message(*args, **kwargs):
    from .utils.message import extract_text_from_message as _f
    return _f(*args, **kwargs)


def format_history_for_prompt(*args, **kwargs):
    from .utils.message import format_history_for_prompt as _f
    return _f(*args, **kwargs)


def format_tool_results_for_conversation(*args, **kwargs):
    from .utils.message import format_tool_results_for_conversation as _f
    return _f(*args, **kwargs)

# Version of the codin package
version: str = '0.1.0'

__all__: list[str] = [
    # Config functions (assuming they are still top-level or re-exported)
    'get_api_key', # Requires from .config import get_api_key
    'get_config',  # Requires from .config import get_config
    'load_config', # Requires from .config import load_config
    # Utility functions
    'extract_text_from_message',
    'format_history_for_prompt',
    'format_tool_results_for_conversation',
    # Submodules
    'agent',
    'actor',
    'a2a',       # Added a2a
    'config',    # Added config (if it's meant to be accessed as codin.config)
    # Package version
    'version',
]

# For backward compatibility or specific testing setups, these sys.modules manipulations exist.
# They might need review in a broader context but are kept for now.
# It's generally better if modules are directly importable via PYTHONPATH.
if 'src.codin.agent.types' not in _sys.modules: # Check before setting default
    import codin.agent.types as _agent_types_module
    _sys.modules.setdefault('src.codin.agent.types', _agent_types_module)

if 'src.codin' not in _sys.modules:
     _sys.modules.setdefault('src.codin', _sys.modules[__name__])

if 'src' not in _sys.modules: # This makes `import src.codin` work if only `codin` is in PYTHONPATH
    # This can be tricky. If 'codin' is a top-level package in PYTHONPATH,
    # then `import codin` works. `import src.codin` implies 'src' is also in PYTHONPATH.
    # This line seems to try to make `src.codin` an alias for `codin` if `codin` is imported.
    # It might be better to ensure consistent PYTHONPATH setup.
    # For now, preserving the logic but with a note.
    _sys.modules.setdefault('src', _sys.modules[__name__].__path__) # type: ignore


# To make get_api_key etc. work if they are in config.py
# This should ideally be done more explicitly if these are public APIs of the package.
# For now, these are not directly imported at the top, so they would fail if not in __init__
# or if config is not imported as `from . import config`.
# The __all__ list implies they should be available. Let's add the import for config's contents.
from .config import get_api_key, get_config, load_config
