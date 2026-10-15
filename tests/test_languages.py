import pandas as pd
import pytest

import constants
import languages
from paths import DATA_DIR

_WALS_DIR = DATA_DIR / "wals-v2020.4" / "cldf-datasets-wals-0f5cd82" / "cldf"
_ASJP_DIR = DATA_DIR / "asjp-v20" / "lexibank-asjp-f0f1d0d" / "cldf"


# ===========================================================================
# Registry contents
# ===========================================================================


def test_codes_unique():
    assert languages.REGISTRY.index.is_unique


def test_covers_every_language_the_pipeline_can_use():
    # Every analysed language needs an FSI rating (English is the reference).
    assert set(languages.REGISTRY.index) == set(constants.FSI_SCALE) | {"en"}
    assert set(constants.XNLI_LANGUAGES) <= set(languages.REGISTRY.index)
    assert set(constants.UN6_LANGS) <= set(languages.REGISTRY.index)


# Each source's full set of identifiers, loaded only by the test that needs it.
_SOURCE_VALUES = {
    "bible_file": lambda: {
        p.stem for p in (DATA_DIR / "aligned" / "1-b.GEN").glob("*.txt")
    },
    "mbert_name": lambda: set(constants.BERT_MULTILINGUAL_LANGS),
    "xlmr_name": lambda: set(constants.XLMR_LANGS),
    "wals_id": lambda: set(
        pd.read_csv(_WALS_DIR / "languages.csv", usecols=["ID"])["ID"]
    ),
    "asjp_id": lambda: set(
        pd.read_csv(_ASJP_DIR / "languages.csv", usecols=["ID"])["ID"]
    ),
    "lexdist_abbr": lambda: set(
        pd.read_excel(DATA_DIR / "lexical-distance-matrix.xlsx", skiprows=2)[
            "Abreviation"
        ]
    ),
}


@pytest.mark.parametrize("source", list(_SOURCE_VALUES))
def test_source_ids_exist_in_source(source):
    valid = _SOURCE_VALUES[source]()
    ids = languages.source_ids(source)
    assert {code: i for code, i in ids.items() if i not in valid} == {}


@pytest.mark.parametrize("source", ["bible_file", "wals_id", "asjp_id", "lexdist_abbr"])
def test_source_ids_unique(source):
    ids = list(languages.source_ids(source).values())
    assert len(ids) == len(set(ids))


# ===========================================================================
# source_ids / from_source_ids
# ===========================================================================


def test_source_ids_omit_uncovered_languages():
    # WALS has no entry for Kurmanji Kurdish.
    assert languages.source_ids("wals_id", ["en", "ku"]) == {"en": "eng"}


def test_source_ids_unknown_code_raises():
    with pytest.raises(KeyError):
        languages.source_ids("wals_id", ["xx"])


def test_from_source_ids_inverts():
    assert languages.from_source_ids("lexdist_abbr")["grk"] == "el"
    assert languages.from_source_ids("lexdist_abbr")["nob"] == "no"


def test_wals_ids_differ_from_iso_codes():
    # ISO 639-3 codes either aren't WALS IDs (fra vs fre) or collide with
    # unrelated WALS languages (deu is Deuri, dan is Dan, slv is Slavey).
    assert languages.source_ids(
        "wals_id", ["cs", "da", "de", "fr", "nl", "ro", "sl"]
    ) == {
        "cs": "cze",
        "da": "dsh",
        "de": "ger",
        "fr": "fre",
        "nl": "dut",
        "ro": "rom",
        "sl": "slo",
    }


def test_bible_files_differ_from_display_names():
    assert languages.source_ids("bible_file", ["fa", "my", "ht", "sl"]) == {
        "fa": "Farsi",
        "my": "Myanmar",
        "ht": "Creole",
        "sl": "Slovene",
    }
