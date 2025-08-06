#!/usr/bin/env python3
"""
LitKG CLI wrapper script.

This script ensures the litkg package can be found and runs the CLI.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now import and run the CLI
try:
    from litkg.cli import main
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"Error importing litkg: {e}", file=sys.stderr)
    print(f"Make sure you're running from the project root: {project_root}", file=sys.stderr)
    sys.exit(1)