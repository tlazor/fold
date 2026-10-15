import numpy as np
import pandas as pd

from paths import DATA_DIR

_WALS_DIR = DATA_DIR / "wals-v2020.4" / "cldf-datasets-wals-0f5cd82" / "cldf"

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


def get_wals_similarity(lang_ids: dict[str, str]) -> pd.DataFrame:
    """Area-balanced, IC-weighted pairwise WALS feature agreement.

    For each language pair (A, B) and each of the 9 WALS areas:
    - Collect features where both languages have values.
    - Weight each feature by IC(f, A[f]) + IC(f, B[f]): rare shared values
      count for more than common ones.
    - Area score = weighted fraction of agreeing features.
    Skip areas with fewer than _MIN_FEATS_PER_AREA shared features.
    The final score is the mean of qualifying area scores; NaN if fewer than
    _MIN_AREAS areas qualify.

    `lang_ids` maps pipeline codes to WALS IDs (see languages.source_ids).
    Returns a symmetric DataFrame indexed by the pipeline codes.
    """
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
    score_sum = np.zeros((n, n))
    n_areas = np.zeros((n, n), dtype=int)

    # Every language pair at once: arrays below are (lang_a, lang_b, feature).
    for feats in area_feats.values():
        if not feats:
            continue
        a = feat_matrix[feats].to_numpy(dtype=float)
        ic_a = np.column_stack(
            [
                feat_matrix[f].map(ic.get(f, {})).fillna(0.0).to_numpy(dtype=float)
                for f in feats
            ]
        )
        observed = ~np.isnan(a)
        shared = observed[:, None, :] & observed[None, :, :]
        # Weight by the combined IC of both languages' observed values.
        # When they agree on rare value v: weight 2·IC(f,v) (high).
        # When they agree on common value v: weight 2·IC(f,v) (low).
        w = np.where(shared, ic_a[:, None, :] + ic_a[None, :, :], 0.0)
        total_w = w.sum(axis=-1)
        agree = a[:, None, :] == a[None, :, :]
        qualifies = (shared.sum(axis=-1) >= _MIN_FEATS_PER_AREA) & (total_w > 0)
        agreed_w = np.where(agree, w, 0.0).sum(axis=-1)
        score_sum += np.divide(agreed_w, total_w, out=np.zeros((n, n)), where=qualifies)
        n_areas += qualifies

    sim = np.full((n, n), np.nan)
    np.divide(score_sum, n_areas, out=sim, where=n_areas >= _MIN_AREAS)
    return pd.DataFrame(sim, index=iso2, columns=iso2)
