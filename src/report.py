"""
Tables and figures for presenting FOLD results (used by notebooks/).

Colours follow a validated reference palette: one series hue and a one-hue
sequential ramp, each stepped for a light or a dark background.
"""

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr

import analysis
import baselines
import constants

FOLD_KIND = {
    "compute_overlaps": "similarity",
    "kl_divergence_matrix": "distance",
    "coherence_fun": "similarity",
}
METRIC_LABEL = {
    "compute_overlaps": "Spectral overlap",
    "kl_divergence_matrix": "KL divergence",
    "coherence_fun": "Coherence",
}
MEASURE_LABEL = {
    "mut_int": "Mutual intelligibility (%)",
    "lex_dist": "Lexical distance (0–100)",
    "pho_dist": "Phonetic distance (ASJP LDND)",
    "wals_sim": "WALS similarity (0–1)",
    "fsi": "FSI weeks of study from English",
}

_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


@cache
def measure_kind(name):
    """'distance' or 'similarity' for a ground-truth measure."""
    kinds = {b.name: b.kind for b in baselines.pairwise()}
    return kinds.get(name, "distance")  # fsi: weeks of study, a distance


def band_label(band):
    return f"{band[0]:.2f}–{band[1]:.2f}"


def correlation_table(results, metric):
    """
    Readable correlations of one FOLD metric with every measure.

    The agreement sign is the sign a correlation takes when FOLD and the
    measure agree: + when both are distances or both similarities, − otherwise.
    """
    rows = []
    for _, r in results.iterrows():
        kind = measure_kind(r["metric"])
        rows.append(
            {
                "measure": MEASURE_LABEL[r["metric"]],
                "is a": kind,
                "agreement sign": "+" if kind == FOLD_KIND[metric] else "−",
                "Pearson r": round(r["p_coef"], 3),
                "Pearson p": analysis.star_fmt(r["p_pval"]),
                "Spearman ρ": round(r["s_coef"], 3),
                "Spearman p": analysis.star_fmt(r["s_pval"]),
                "Kendall τ": round(r["kt_coef"], 3),
                "pairs": int(r["num_points"]),
            }
        )
    return pd.DataFrame(rows)


def coverage_table(langs):
    """How many language pairs each measure can score, and who is left out."""
    ones = np.ones((len(langs), len(langs)))
    np.fill_diagonal(ones, np.nan)
    pairs = analysis.baseline_pairs(pd.DataFrame(ones, index=langs, columns=langs))
    rows = []
    for name, df in pairs.items():
        usable = df.dropna()
        if name == "fsi":
            covered = set(usable.index) | {"en"}
        else:
            covered = {lang for pair in usable.index for lang in pair}
        rows.append(
            {
                "measure": MEASURE_LABEL[name],
                "pairs": len(usable),
                "languages without data": ", ".join(sorted(set(langs) - covered))
                or "–",
            }
        )
    return pd.DataFrame(rows)


def _style(dark):
    return {
        "surface": "#1a1a19" if dark else "#fcfcfb",
        "primary": "#ffffff" if dark else "#0b0b0b",
        "secondary": "#c3c2b7" if dark else "#52514e",
        "grid": "#2c2c2a" if dark else "#e1e0d9",
        "axis": "#383835" if dark else "#c3c2b7",
        "series": "#3987e5" if dark else "#2a78d6",
        # Low values recede toward the background in either theme.
        "ramp": _RAMP[::-1] if dark else _RAMP,
    }


def _chrome(fig, ax, style):
    fig.set_facecolor(style["surface"])
    ax.set_facecolor(style["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style["axis"])
    ax.tick_params(colors=style["axis"], labelcolor=style["secondary"], labelsize=8)


def scatter_figure(pairs, measure, metric, dark=False):
    """
    FOLD value against a measure, one dot per language pair.

    `pairs` is one entry of analysis.baseline_pairs(): columns "fold" and
    `measure`, indexed by (lang_a, lang_b), or by language for "fsi".
    """
    style = _style(dark)
    x, y = pairs[measure].to_numpy(float), pairs["fold"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), layout="constrained")
    _chrome(fig, ax, style)
    ax.grid(True, color=style["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.scatter(
        x, y, s=49, color=style["series"], edgecolor=style["surface"], linewidth=1.5
    )

    title = f"n = {len(x)}"
    if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        title = (
            f"Pearson r = {pearsonr(x, y)[0]:.2f}, "
            f"Spearman ρ = {spearmanr(x, y)[0]:.2f}, " + title
        )
        # Label up to three of the points furthest from the least-squares
        # line, skipping any that would sit on top of a label already placed.
        slope, intercept = np.polyfit(x, y, 1)
        xn, yn = (x - x.min()) / np.ptp(x), (y - y.min()) / np.ptp(y)
        placed = []
        for i in np.argsort(-np.abs(y - (slope * x + intercept))):
            if len(placed) == 3:
                break
            if any(np.hypot(xn[i] - xn[j], yn[i] - yn[j]) < 0.06 for j in placed):
                continue
            placed.append(i)
            label = pairs.index[i]
            label = "–".join(label) if isinstance(label, tuple) else f"en–{label}"
            ax.annotate(
                label,
                (x[i], y[i]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color=style["secondary"],
            )
    ax.set_title(title, loc="left", fontsize=10, color=style["primary"])
    ax.set_xlabel(
        f"{MEASURE_LABEL[measure]} — {measure_kind(measure)}",
        fontsize=9,
        color=style["secondary"],
    )
    ax.set_ylabel(
        f"{METRIC_LABEL[metric]} — {FOLD_KIND[metric]}",
        fontsize=9,
        color=style["secondary"],
    )
    return fig


def fold_heatmap(fold, metric, dark=False):
    """The median FOLD matrix (analysis.fold_matrix), languages grouped by family."""
    style = _style(dark)
    order = sorted(
        fold.index, key=lambda lang: (constants.lang2family.get(lang, "~"), lang)
    )
    fold = fold.loc[order, order]
    cmap = LinearSegmentedColormap.from_list("fold", style["ramp"])
    cmap.set_bad(style["surface"])  # the diagonal (self-pairs) is not data

    size = max(4.2, 0.17 * len(order) + 1.6)
    fig, ax = plt.subplots(figsize=(size + 1.0, size), layout="constrained")
    _chrome(fig, ax, style)
    image = ax.imshow(fold.to_numpy(float), cmap=cmap)
    ax.set_xticks(range(len(order)), order, rotation=90)
    ax.set_yticks(range(len(order)), order)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    # A thin background-coloured gap between language families.
    families = [constants.lang2family.get(lang, "~") for lang in order]
    for i in range(1, len(order)):
        if families[i] != families[i - 1]:
            ax.axhline(i - 0.5, color=style["surface"], linewidth=2)
            ax.axvline(i - 0.5, color=style["surface"], linewidth=2)

    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=style["axis"], labelcolor=style["secondary"], labelsize=8)
    bar.set_label(
        f"{METRIC_LABEL[metric]} ({FOLD_KIND[metric]})",
        fontsize=9,
        color=style["secondary"],
    )
    ax.set_title(
        "Median over sentences, grouped by family",
        loc="left",
        fontsize=10,
        color=style["primary"],
    )
    return fig
