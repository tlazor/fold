"""Absolute paths anchored to the project root.

Import from this module instead of using bare Path(".") strings so that
scripts run correctly regardless of the working directory.
"""

from pathlib import Path

# src/paths.py → src/ → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
