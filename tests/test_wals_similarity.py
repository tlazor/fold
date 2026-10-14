import math
import pytest
from wals_similarity import get_wals_ids, get_wals_similarity, WALS_LANG_IDS


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


def test_wals_ids_match_curated_mapping():
    assert get_wals_ids(list(WALS_LANG_IDS)) == WALS_LANG_IDS


def test_wals_ids_use_wals_codes_not_iso():
    # ISO 639-3 codes either aren't WALS IDs (fra vs fre) or collide with
    # unrelated WALS languages (deu is Deuri, dan is Dan, slv is Slavey).
    assert get_wals_ids(["cs", "da", "de", "fr", "nl", "ro", "sl"]) == {
        "cs": "cze",
        "da": "dsh",
        "de": "ger",
        "fr": "fre",
        "nl": "dut",
        "ro": "rom",
        "sl": "slo",
    }


def test_wals_ids_prefer_best_covered_dialect():
    # e.g. Hebrew (Modern) over Hebrew (Modern Ashkenazic).
    assert get_wals_ids(["he", "it", "sv"]) == {"he": "heb", "it": "ita", "sv": "swe"}


def test_wals_ids_manual_overrides():
    assert get_wals_ids(["ne", "sr"]) == {"ne": "nep", "sr": "scr"}


def test_wals_ids_omit_languages_missing_from_wals():
    # Corsican has no WALS entry.
    assert get_wals_ids(["en", "co"]) == {"en": "eng"}
