"""Absolute paths anchored to the project root.

Import from this module instead of using bare Path(".") strings so that
scripts run correctly regardless of the working directory.
"""

from pathlib import Path

import torch

# src/paths.py → src/ → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
