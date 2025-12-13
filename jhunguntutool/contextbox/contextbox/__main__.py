"""
ContextBox CLI - Main Entry Point

This module provides the command-line interface for ContextBox.
Usage: python -m contextbox.cli capture
"""

import sys
import os

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

if __name__ == '__main__':
    # Import and run CLI
    from contextbox.cli import main
    main()
