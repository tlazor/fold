"""Registry of the languages the pipeline can analyse.

Each language is identified by its pipeline code: ISO 639-1 where one exists,
otherwise ISO 639-3 (e.g. "yue"). data/constants/languages.csv maps that code
to the identifier every external source uses for it (Bible filename, model
language-list name, WALS ID, ASJP doculect, lexical-distance abbreviation), so
sources are matched by explicit lookup rather than by converting codes on the
fly.
"""

from collections.abc import Iterable

import pandas as pd

from paths import DATA_DIR

REGISTRY = pd.read_csv(
    DATA_DIR / "constants" / "languages.csv",
    index_col="code",
    dtype=str,
    keep_default_na=False,
)


def source_ids(source: str, langs: Iterable[str] | None = None) -> dict[str, str]:
    """Map pipeline codes to the identifiers `source` uses for them.

    `source` is a registry column such as "wals_id" or "bible_file". Languages
    the source doesn't cover are omitted; codes missing from the registry raise
    KeyError.
    """
    column = REGISTRY[source]
    if langs is not None:
        column = column.loc[list(langs)]
    return {code: value for code, value in column.items() if value}


def from_source_ids(source: str) -> dict[str, str]:
    """Map `source`'s identifiers back to pipeline codes."""
    return {value: code for code, value in source_ids(source).items()}
