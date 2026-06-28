import sys
from pathlib import Path
import numpy as np
import pytest

_SRC = str(Path(__file__).parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture
def freq_2d():
    rng = np.random.default_rng(42)
    return np.abs(rng.random((3, 8))) + 0.1


@pytest.fixture
def freq_3d():
    rng = np.random.default_rng(42)
    return np.abs(rng.random((3, 8, 4))) + 0.1


@pytest.fixture
def signal_list_2d():
    rng = np.random.default_rng(7)
    return [rng.random((3, 16)) for _ in range(4)]


@pytest.fixture
def psd_list_2d():
    rng = np.random.default_rng(13)
    return [np.abs(rng.random((3, 8))) + 0.01 for _ in range(4)]


@pytest.fixture
def psd_list_3d():
    rng = np.random.default_rng(17)
    return [np.abs(rng.random((3, 4, 8))) + 0.01 for _ in range(4)]
