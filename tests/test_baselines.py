import numpy as np
import pandas as pd
import pytest

import baselines
import languages


@pytest.fixture(
    scope="module",
    params=[
        "mutual_intelligibility",
        "lexical_distance",
        "phonetic_distance",
        "wals_similarity",
    ],
)
def baseline(request):
    return getattr(baselines, request.param)()


# ===========================================================================
# Shared invariants
# ===========================================================================


def test_square_and_keyed_by_registry_codes(baseline):
    m = baseline.matrix
    assert list(m.index) == list(m.columns)
    assert set(m.index) <= set(languages.REGISTRY.index)


def test_diagonal_is_nan(baseline):
    assert np.isnan(np.diag(baseline.matrix.to_numpy())).all()


def test_kind_declared(baseline):
    assert baseline.kind in {"distance", "similarity"}


def test_symmetric_baselines_are_symmetric(baseline):
    if baseline.symmetric:
        pd.testing.assert_frame_equal(baseline.matrix, baseline.matrix.T)


def test_pairwise_names_unique():
    names = [b.name for b in baselines.pairwise()]
    assert len(names) == len(set(names))


# ===========================================================================
# Individual baselines
# ===========================================================================


def test_mutual_intelligibility_orientation():
    # Rows are the listener's language: Portuguese listeners understand
    # Spanish far better than Spanish listeners understand Portuguese.
    m = baselines.mutual_intelligibility().matrix
    assert m.loc["pt", "es"] == 77.4
    assert m.loc["es", "pt"] == 37.2


def test_mutual_intelligibility_includes_dutch():
    # The Germanic table used to label Dutch "du".
    assert "nl" in baselines.mutual_intelligibility().matrix.index


def test_lexical_distance_includes_greek_and_norwegian():
    # The spreadsheet abbreviates them "grk" and "nob".
    assert {"el", "no"} <= set(baselines.lexical_distance().matrix.index)


def test_lexical_distance_is_a_distance():
    m = baselines.lexical_distance().matrix
    assert m.loc["es", "pt"] < m.loc["es", "it"]


def test_phonetic_distance_includes_macrolanguages():
    # ASJP tags these by individual-language codes (arb, swh, cmn).
    assert {"ar", "sw", "zh"} <= set(baselines.phonetic_distance().matrix.index)


def test_fsi_is_relative_to_english():
    fsi = baselines.fsi()
    assert "en" not in fsi.index
    assert fsi["ja"] > fsi["es"]
