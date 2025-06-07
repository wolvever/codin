#!/usr/bin/env python3
"""Main entry point for codin CLI."""

import sys
from pathlib import Path

# Add the parent directory to sys.path if running directly
if __name__ == '__main__':
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    sys.path.append(str(project_root))

from .commands import main

if __name__ == '__main__':
    main()
