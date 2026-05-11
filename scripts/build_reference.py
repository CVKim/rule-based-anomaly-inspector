"""CLI wrapper to build a reference from a folder of known-good images.

Usage:
    python scripts/build_reference.py --input <folder> --output <out.npz> [--config <yaml>]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the script in-place without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_inspector.cli import build_app  # noqa: E402


if __name__ == "__main__":
    build_app()
