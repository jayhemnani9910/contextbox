"""Pytest configuration for workspace-specific patches."""

import os
from pathlib import Path


def pytest_sessionstart(session):
    """Patch os.chdir to handle hard-coded workspace paths used in tests."""
    workspace_root = Path(__file__).resolve().parent / "contextbox"
    original_chdir = os.chdir

    def _safe_chdir(path: str):
        if path == "/workspace/contextbox" and workspace_root.exists():
            original_chdir(workspace_root.as_posix())
        else:
            original_chdir(path)

    os.chdir = _safe_chdir  # type: ignore[assignment]
