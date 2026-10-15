import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import constants  # noqa: E402
import report  # noqa: E402
from analysis import baseline_pairs, fold_matrix  # noqa: E402


def test_coverage_table_counts_pairs_per_measure():
    table = report.coverage_table(list(constants.XNLI_LANGUAGES)).set_index("measure")
    pairs = {
        name: table.loc[label, "pairs"] for name, label in report.MEASURE_LABEL.items()
    }
    # Same counts as tests/test_coverage.py pins for XNLI.
    assert pairs == {
        "mut_int": 4,
        "lex_dist": 5,
        "pho_dist": 105,
        "wals_sim": 102,
        "fsi": 14,
    }


def _results(*names):
    return pd.DataFrame(
        [
            {
                "metric": name,
                "p_coef": 0.5,
                "p_pval": 0.01,
                "s_coef": 0.4,
                "s_pval": 0.2,
                "kt_coef": 0.3,
                "num_points": 10,
            }
            for name in names
        ]
    )


def test_agreement_sign_follows_both_directions():
    names = ("mut_int", "lex_dist", "fsi")
    overlap = report.correlation_table(_results(*names), "compute_overlaps")
    kl = report.correlation_table(_results(*names), "kl_divergence_matrix")
    # Overlap is a similarity, KL a distance; mut_int is a similarity, the rest distances.
    assert list(overlap["agreement sign"]) == ["+", "−", "−"]
    assert list(kl["agreement sign"]) == ["−", "+", "+"]
    assert overlap.loc[0, "Pearson p"] == "0.010*"


def test_figures_render():
    langs = ["de", "en", "es", "fr", "it"]
    output = np.random.default_rng(0).random((4, 5, 5))
    matrix = fold_matrix(output, langs)
    pairs = baseline_pairs(matrix)["fsi"].dropna()
    for dark in (False, True):
        scatter = report.scatter_figure(pairs, "fsi", "compute_overlaps", dark=dark)
        heatmap = report.fold_heatmap(matrix, "kl_divergence_matrix", dark=dark)
        assert scatter.axes and heatmap.axes
    plt.close("all")
