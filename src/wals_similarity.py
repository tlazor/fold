import langcodes
import numpy as np
import pandas as pd

from paths import DATA_DIR

_WALS_DIR = DATA_DIR / "wals-v2020.4" / "cldf-datasets-wals-0f5cd82" / "cldf"

WALS_LANG_IDS = {
    "ar": "ams",  # Arabic (Modern Standard)
    "bg": "bul",  # Bulgarian
    "de": "ger",  # German
    "el": "grk",  # Greek (Modern)
    "en": "eng",  # English
    "es": "spa",  # Spanish
    "fr": "fre",  # French
    "hi": "hin",  # Hindi
    "ru": "rus",  # Russian
    "sw": "swa",  # Swahili
    "th": "tha",  # Thai
    "tr": "tur",  # Turkish
    "ur": "urd",  # Urdu
    "vi": "vie",  # Vietnamese
    "zh": "mnd",  # Mandarin
}

# Languages outside WALS_LANG_IDS that can't be found by ISO 639-3 code:
# langcodes gives the macrolanguage "nep" where WALS lists Nepali as "npi",
# and WALS merges Serbian into Serbian-Croatian ("hbs").
_EXTRA_WALS_IDS = {
    "ne": "nep",  # Nepali
    "sr": "scr",  # Serbian-Croatian
}

_CORE_AREAS = [
    "Phonology",
    "Morphology",
    "Nominal Categories",
    "Nominal Syntax",
    "Verbal Categories",
    "Word Order",
    "Simple Clauses",
    "Complex Sentences",
    "Lexicon",
]

# Minimum shared features for an area to count toward a pair's score.
_MIN_FEATS_PER_AREA = 2
# Minimum number of qualifying areas required to produce a non-NaN score.
_MIN_AREAS = 3


def get_wals_ids(langs: list[str]) -> dict[str, str]:
    """Map pipeline language codes to WALS language IDs.

    WALS uses its own IDs (German is "ger", not ISO 639-3 "deu"), and some ISO
    codes collide with unrelated WALS IDs ("deu" is Deuri), so match on the
    ISO639P3code column instead. Where several WALS entries share an ISO code
    (dialect samples such as "German (Berlin)"), take the one with the most
    coded features. WALS_LANG_IDS and _EXTRA_WALS_IDS take precedence.
    Languages with no WALS entry are omitted.
    """
    languages = pd.read_csv(_WALS_DIR / "languages.csv", usecols=["ID", "ISO639P3code"])
    values = pd.read_csv(_WALS_DIR / "values.csv", usecols=["Language_ID"])
    languages["n_feats"] = (
        languages["ID"].map(values["Language_ID"].value_counts()).fillna(0)
    )
    best_by_iso = (
        languages.dropna(subset=["ISO639P3code"])
        .sort_values("n_feats", ascending=False, kind="stable")
        .drop_duplicates("ISO639P3code")
        .set_index("ISO639P3code")["ID"]
    )

    curated = {**WALS_LANG_IDS, **_EXTRA_WALS_IDS}
    wals_ids = {}
    for lang in langs:
        if lang in curated:
            wals_ids[lang] = curated[lang]
            continue
        iso3 = langcodes.Language.get(lang).to_alpha3()
        if iso3 in best_by_iso.index:
            wals_ids[lang] = best_by_iso[iso3]
    return wals_ids


def _load_param_to_area() -> dict[str, str]:
    params = pd.read_csv(_WALS_DIR / "parameters.csv")
    chapters = pd.read_csv(_WALS_DIR / "chapters.csv")
    areas = pd.read_csv(_WALS_DIR / "areas.csv")

    # chapters.ID mixes "s1" (intro) with plain integers; drop the non-numeric row.
    chapters["_id_int"] = pd.to_numeric(chapters["ID"], errors="coerce")
    chapters = chapters.dropna(subset=["_id_int"])
    chapters["_id_int"] = chapters["_id_int"].astype(int)

    merged = params.merge(
        chapters[["_id_int", "Area_ID"]],
        left_on="Chapter_ID",
        right_on="_id_int",
    ).merge(
        areas[["ID", "Name"]],
        left_on="Area_ID",
        right_on="ID",
        suffixes=("_p", "_a"),
    )
    return dict(zip(merged["ID_p"], merged["Name_a"]))


def _load_ic_weights() -> dict[str, dict[float, float]]:
    """IC(f, v) = -log2(count(v, f) / N_observed(f)) over all WALS languages.

    Using global frequencies means the IC reflects genuine typological rarity,
    not just rarity within the small set of pipeline languages.
    """
    values = pd.read_csv(_WALS_DIR / "values.csv", usecols=["Parameter_ID", "Value"])
    ic: dict[str, dict[float, float]] = {}
    for param, group in values.groupby("Parameter_ID"):
        counts = group["Value"].value_counts()
        total = counts.sum()
        ic[str(param)] = {
            float(v): float(-np.log2(c / total)) for v, c in counts.items()
        }
    return ic


def get_wals_similarity(lang_ids: dict[str, str] | None = None) -> pd.DataFrame:
    """Area-balanced, IC-weighted pairwise WALS feature agreement.

    For each language pair (A, B) and each of the 9 WALS areas:
    - Collect features where both languages have values.
    - Weight each feature by IC(f, A[f]) + IC(f, B[f]): rare shared values
      count for more than common ones.
    - Area score = weighted fraction of agreeing features.
    Skip areas with fewer than _MIN_FEATS_PER_AREA shared features.
    The final score is the mean of qualifying area scores; NaN if fewer than
    _MIN_AREAS areas qualify.

    Returns a symmetric DataFrame indexed by ISO 639-1 2-letter codes.
    """
    if lang_ids is None:
        lang_ids = WALS_LANG_IDS

    iso2 = list(lang_ids.keys())
    wids = list(lang_ids.values())

    param_to_area = _load_param_to_area()
    ic = _load_ic_weights()

    values = pd.read_csv(_WALS_DIR / "values.csv")
    v = values[values["Language_ID"].isin(wids)][
        ["Language_ID", "Parameter_ID", "Value"]
    ].copy()
    v["Area"] = v["Parameter_ID"].map(param_to_area)
    v = v[v["Area"].isin(_CORE_AREAS)]

    feat_matrix = v.pivot(
        index="Language_ID", columns="Parameter_ID", values="Value"
    ).reindex(wids)

    area_feats: dict[str, list[str]] = {
        area: [f for f in feat_matrix.columns if param_to_area.get(f) == area]
        for area in _CORE_AREAS
    }

    n = len(wids)
    sim = np.full((n, n), np.nan)

    for i, la in enumerate(wids):
        for j, lb in enumerate(wids):
            area_scores: list[float] = []
            for area, feats in area_feats.items():
                if not feats:
                    continue
                a = feat_matrix.loc[la, feats].to_numpy(dtype=float)
                b = feat_matrix.loc[lb, feats].to_numpy(dtype=float)
                shared_mask = ~np.isnan(a) & ~np.isnan(b)
                if shared_mask.sum() < _MIN_FEATS_PER_AREA:
                    continue
                shared_feats = [feats[k] for k in np.where(shared_mask)[0]]
                av = a[shared_mask]
                bv = b[shared_mask]
                # Weight by the combined IC of both languages' observed values.
                # When they agree on rare value v: weight 2·IC(f,v) (high).
                # When they agree on common value v: weight 2·IC(f,v) (low).
                w = np.array(
                    [
                        ic.get(f, {}).get(float(va), 0.0)
                        + ic.get(f, {}).get(float(vb), 0.0)
                        for f, va, vb in zip(shared_feats, av, bv)
                    ]
                )
                total_w = w.sum()
                if total_w == 0:
                    continue
                agree = (av == bv).astype(float)
                area_scores.append(float(np.dot(agree, w) / total_w))

            if len(area_scores) >= _MIN_AREAS:
                sim[i, j] = float(np.mean(area_scores))

    return pd.DataFrame(sim, index=iso2, columns=iso2)
