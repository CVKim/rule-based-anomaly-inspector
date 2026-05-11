"""CLI wrapper to inspect images using a saved reference.

Usage:
    python scripts/inspect.py --reference ref.npz --input <file_or_folder> \
                              --output <dir> [--config <yaml>]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_inspector.cli import inspect_app  # noqa: E402


if __name__ == "__main__":
    inspect_app()
