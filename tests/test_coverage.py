"""Pin which languages each baseline resolves for each dataset.

Mismatched identifiers between sources used to drop languages silently, which
only showed up as a smaller num_points in the output tables. Any change to
these numbers should be deliberate: update the table below in the same commit
and say why.
"""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

import baselines
from analysis import get_langs, get_overlap

# Languages each dataset analyses, by model.
EXPECTED_LANG_COUNTS = {
    ("xnli", True): 15,
    ("xnli", False): 15,
    ("un6", True): 6,
    ("un6", False): 6,
    ("bible", True): 37,
    ("bible", False): 37,
}

# (dataset, use_bert, baseline): (usable pairs, languages with no usable pair).
# Pairs are what the correlation sees: unordered for symmetric baselines,
# ordered for mut_int. Missing languages are absent from the source itself
# (e.g. mut_int only has Germanic, Romance and Slavic tables; the lexical
# chart only links nearby European languages), or too sparse to score
# (WALS has 8 features for Haitian Creole).
EXPECTED_COVERAGE = {
    ("xnli", True, "mut_int"): (
        4,
        {"ar", "bg", "el", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"},
    ),
    ("xnli", True, "lex_dist"): (
        5,
        {"ar", "es", "hi", "sw", "th", "tr", "ur", "vi", "zh"},
    ),
    ("xnli", True, "pho_dist"): (105, set()),
    ("xnli", True, "wals_sim"): (102, set()),
    ("un6", True, "mut_int"): (2, {"ar", "en", "ru", "zh"}),
    ("un6", True, "lex_dist"): (1, {"ar", "es", "ru", "zh"}),
    ("un6", True, "pho_dist"): (15, set()),
    ("un6", True, "wals_sim"): (15, set()),
    ("bible", True, "mut_int"): (
        60,
        {
            "ar", "el", "fa", "fi", "he", "hi", "ht", "hu", "id", "ja", "ko",
            "lt", "my", "ne", "no", "ru", "sq", "sr", "th", "tr", "vi", "zh",
        },
    ),
    ("bible", True, "lex_dist"): (
        43,
        {
            "ar", "fa", "he", "hi", "ht", "id", "ja", "ko", "my", "ne", "th",
            "tr", "vi", "zh",
        },
    ),
    ("bible", True, "pho_dist"): (666, set()),
    ("bible", True, "wals_sim"): (607, {"ht"}),
    ("bible", False, "mut_int"): (
        60,
        {
            "am", "ar", "el", "fa", "fi", "he", "hi", "hu", "id", "ja", "ko",
            "lt", "my", "ne", "no", "ru", "sq", "sr", "th", "tr", "vi", "zh",
        },
    ),
    ("bible", False, "lex_dist"): (
        43,
        {
            "am", "ar", "fa", "he", "hi", "id", "ja", "ko", "my", "ne", "th",
            "tr", "vi", "zh",
        },
    ),
    ("bible", False, "pho_dist"): (666, set()),
    ("bible", False, "wals_sim"): (642, set()),
}  # fmt: skip


def _langs(dataset, use_bert):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_langs(dataset, use_bert)


def _usable_pairs(langs, baseline):
    """Language pairs that reach the correlation, via analyze_output's path."""
    fold = np.ones((len(langs), len(langs)))
    np.fill_diagonal(fold, np.nan)
    fold_df = pd.DataFrame(fold, index=langs, columns=langs)
    overlap = get_overlap(
        fold_df, baseline.matrix, baseline.name, symmetrical=baseline.symmetric
    )
    return list(overlap.dropna().index)


@pytest.mark.parametrize(("dataset", "use_bert"), list(EXPECTED_LANG_COUNTS))
def test_dataset_language_count(dataset, use_bert):
    assert len(_langs(dataset, use_bert)) == EXPECTED_LANG_COUNTS[(dataset, use_bert)]


@pytest.mark.parametrize(("dataset", "use_bert", "name"), list(EXPECTED_COVERAGE))
def test_baseline_coverage(dataset, use_bert, name):
    langs = _langs(dataset, use_bert)
    baseline = next(b for b in baselines.pairwise() if b.name == name)
    pairs = _usable_pairs(langs, baseline)
    missing = set(langs) - {lang for pair in pairs for lang in pair}
    assert (len(pairs), missing) == EXPECTED_COVERAGE[(dataset, use_bert, name)]


def test_every_baseline_has_coverage_entries():
    names = {b.name for b in baselines.pairwise()}
    assert {name for _, _, name in EXPECTED_COVERAGE} == names


@pytest.mark.parametrize(("dataset", "use_bert"), list(EXPECTED_LANG_COUNTS))
def test_every_language_has_fsi(dataset, use_bert):
    langs = set(_langs(dataset, use_bert)) - {"en"}
    assert langs <= set(baselines.fsi().index)
