import math
import pytest
from wals_similarity import get_wals_similarity, WALS_LANG_IDS


@pytest.fixture(scope="module")
def sim():
    return get_wals_similarity()


def test_shape(sim):
    n = len(WALS_LANG_IDS)
    assert sim.shape == (n, n)


def test_index_labels(sim):
    expected = set(WALS_LANG_IDS.keys())
    assert set(sim.index) == expected
    assert set(sim.columns) == expected


def test_diagonal_is_one(sim):
    for lang in sim.index:
        assert sim.loc[lang, lang] == pytest.approx(1.0)


def test_symmetry(sim):
    diff = (sim - sim.T).abs()
    assert diff.max().max() < 1e-10


def test_hindi_urdu_high(sim):
    # Hindi and Urdu share grammar almost entirely; similarity should be very high.
    assert sim.loc["hi", "ur"] > 0.9


def test_thai_vietnamese_high(sim):
    # Both isolating, Mainland Southeast Asian, strong typological overlap.
    assert sim.loc["th", "vi"] > 0.6


def test_arabic_swahili_low_or_nan(sim):
    val = sim.loc["ar", "sw"]
    assert math.isnan(val) or val < 0.5


def test_values_in_range(sim):
    for lang_a in sim.index:
        for lang_b in sim.columns:
            v = sim.loc[lang_a, lang_b]
            if not math.isnan(v):
                assert 0.0 <= v <= 1.0
