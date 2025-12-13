"""
Test harness compatibility helpers.

Pytest in this repository runs CLI scripts that expect the project to live at
/workspace/contextbox.  Inside the Codex workspace the project root lives
elsewhere, so we transparently remap that path to the actual checkout whenever
os.chdir is invoked.
"""

import os
from pathlib import Path

_ORIGINAL_CHDIR = os.chdir
_WORKSPACE_TARGET = Path(__file__).resolve().parent / "contextbox"


def _patched_chdir(path: str) -> None:
    """Redirect known hard-coded paths to the local checkout."""
    if path == "/workspace/contextbox" and _WORKSPACE_TARGET.exists():
        _ORIGINAL_CHDIR(_WORKSPACE_TARGET.as_posix())
        return
    _ORIGINAL_CHDIR(path)


os.chdir = _patched_chdir  # type: ignore[assignment]
