"""Make the sibling tron2_env package importable from TRON2 examples."""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_external_tron2_env_on_path() -> None:
    openpi_root = Path(__file__).resolve().parents[2]
    workspace_root = openpi_root.parent
    candidate_paths = [
        workspace_root / "tron2_env" / "src",
        workspace_root,
    ]
    for path in reversed(candidate_paths):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
