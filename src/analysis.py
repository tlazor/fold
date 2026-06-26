"""
Analysis helpers: correlation, mutual-intelligibility overlap, visualisation.

Deliberately free of torch / transformers imports so this module can be
imported in tests and notebooks without a GPU.
"""

from datetime import datetime
from itertools import combinations
from pathlib import Path
import warnings

import langcodes
from matplotlib.colors import TwoSlopeNorm
from paths import DATA_DIR
import matplotlib.transforms as mtransforms
from scipy.stats import pearsonr, spearmanr, rankdata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import constants


def show_heatmap(overlap_matrix, lang_labels=None):
    plt.figure(figsize=(6, 5))
    if lang_labels is not None:
        ax = sns.heatmap(
            overlap_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=lang_labels,
            yticklabels=lang_labels,
            cmap="Blues",
        )
    else:
        ax = sns.heatmap(overlap_matrix, annot=True, fmt=".2f", cmap="Blues")
    ax.set_title("Overlap Matrix Heatmap")
    plt.tight_layout()
    plt.show()


def calculate_correlations_new(dataframes):
    all_correlations = []

    for df_name, df in dataframes.items():
        if df.dropna().shape[0] < 2:
            print(f"{df_name=} has less than 2 non-NA pairs")
            continue

        for i, row in enumerate(df.columns):
            for j in range(i + 1, len(df.columns)):
                col = df.columns[j]
                x = df[row]
                y = df[col]
                valid = x.notna() & y.notna()
                x = pd.to_numeric(x[valid], errors="coerce")
                y = pd.to_numeric(y[valid], errors="coerce")

                if len(x) < 2:
                    continue

                p_coef, p_pval = pearsonr(x, y)
                x_mean, y_mean = x.mean(), y.mean()
                pointwise_pearson = ((x - x_mean) * (y - y_mean)) / np.sqrt(
                    ((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum()
                )

                s_coef, s_pval = spearmanr(x, y)

                rx = rankdata(x)
                ry = rankdata(y)
                num_concordant = 0
                num_discordant = 0
                for a, b in combinations(range(len(rx)), 2):
                    concordant = (rx[a] - rx[b]) * (ry[a] - ry[b]) > 0
                    discordant = (rx[a] - rx[b]) * (ry[a] - ry[b]) < 0
                    num_concordant += concordant
                    num_discordant += discordant

                all_correlations.append(
                    {
                        "metric": f"{col}",
                        "p_coef": p_coef,
                        "p_pval": p_pval,
                        "s_coef": s_coef,
                        "s_pval": s_pval,
                        "num_points": len(x),
                        "pearson_contrib": pointwise_pearson,
                        "num_concordant": num_concordant,
                        "num_discordant": num_discordant,
                    }
                )

    return pd.DataFrame(all_correlations)


def get_full_mut_int():
    all_labels = (
        constants.GERMANIC_INTELLIGABILITY.index.union(
            constants.ROMANCE_INTELLIGABILITY.index
        )
        .union(constants.SLAVIC_INTELLIGABILITY.index)
        .drop(["Mean", "Total"], errors="ignore")
    )

    df1_re = constants.GERMANIC_INTELLIGABILITY.reindex(
        index=all_labels, columns=all_labels
    )
    df2_re = constants.ROMANCE_INTELLIGABILITY.reindex(
        index=all_labels, columns=all_labels
    )
    df3_re = constants.SLAVIC_INTELLIGABILITY.reindex(
        index=all_labels, columns=all_labels
    )

    return df1_re.combine_first(df2_re).combine_first(df3_re)


def star_fmt(p: float) -> str:
    """Return p-value string with significance stars."""
    if p < 0.001:
        return f"{p:.3f}***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"


formatters = {c: star_fmt for c in ["p_pval", "s_pval"]}


def analyze_output(output, langs, f=None, model_name=None, flag_analyze_pearson_contrib=False):
    if "en" not in langs:
        raise ValueError(
            f"English ('en') must be present in langs for FSI correlation; got {langs}"
        )
    en_index = langs.index("en")

    median_matrix = np.nanmedian(output, axis=0)
    xnli_df = pd.DataFrame(median_matrix, index=langs, columns=langs)
    for i in range(len(langs)):
        xnli_df.iloc[i, i] = np.nan

    en_source_distances = median_matrix[en_index]
    df_fsi = pd.DataFrame(columns=["lang", "fsi", "fold"])

    for lang, fold_distance in zip(langs, en_source_distances):
        if lang == "en":
            continue
        df_fsi.loc[len(df_fsi)] = [lang, constants.FSI_SCALE[lang], fold_distance]

    df_fsi.set_index("lang", inplace=True)

    df_mut_int = get_overlap(xnli_df, get_full_mut_int(), "mut_int", symmetrical=False)
    df_lex_sim = get_overlap(xnli_df, constants.LEXICAL_SIMILARITY, "lex_sim")
    df_pho_sim = get_overlap(xnli_df, constants.PHONETIC_SIMILARITY, "pho_sim")

    dataframes = {
        "mut_int": df_mut_int,
        "lex_sim": df_lex_sim,
        "fsi": df_fsi,
        "pho_sim": df_pho_sim,
    }
    results_long = calculate_correlations_new(dataframes)
    print(
        results_long.to_markdown(index=False, floatfmt=".3f", tablefmt="github"),
        file=f,
        flush=True,
    )

    if flag_analyze_pearson_contrib:
        analyze_pearson_contrib(results_long, model_name)


def analyze_pearson_contrib(results_long, model_name):
    if "pearson_contrib" not in results_long.columns:
        print("analyze_pearson_contrib: no correlations computed (results_long is empty); skipping")
        return
    df = results_long["pearson_contrib"]

    for index, metric_pearson_contrib in df.items():
        metric_name = results_long.loc[index, "metric"]

        langs = set()

        if isinstance(metric_pearson_contrib.index[0], tuple):
            for index_langs in metric_pearson_contrib.index:
                langs.update(index_langs)

            langs = sorted(langs)
            matrix = pd.DataFrame(np.nan, index=langs, columns=langs)

            for index_langs in metric_pearson_contrib.index:
                matrix.loc[index_langs] = metric_pearson_contrib[index_langs]

            plot_pearson_contrib(matrix, metric_name, results_long, index)
            analyze_wikisize(matrix, metric_name, index, model_name)
        else:
            langs = sorted(metric_pearson_contrib.index)
            en_row = pd.DataFrame(np.nan, index=langs, columns=langs)
            en_row.loc["en"] = metric_pearson_contrib[langs]

            plot_pearson_contrib(en_row, f"{metric_name}_en_row", results_long, index)
            analyze_wikisize(en_row, f"{metric_name}_en_row", index, model_name)


def plot_pearson_contrib(matrix, metric_name, results_long, index):
    if metric_name == "fold_en_row":
        print(f"{matrix.loc['en']=}")
        return

    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix = matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if matrix.empty or matrix.notna().sum().sum() == 0:
        warnings.warn("Nothing to plot – every row/column lacked numerical data.")
        return

    def order_by_family(lang_list):
        blocks = {}
        for lang in lang_list:
            fam = constants.lang2family.get(lang, f"Unknown:{lang}")
            blocks.setdefault(fam, []).append(lang)
        ordered_families = sorted(blocks)
        ordered_langs = [lng for fam in ordered_families for lng in blocks[fam]]
        return ordered_langs, ordered_families, blocks

    ordered_rows, row_families, row_blocks = order_by_family(list(matrix.index))
    ordered_cols, col_families, col_blocks = order_by_family(list(matrix.columns))

    matrix = matrix.loc[ordered_rows, ordered_cols]

    vmin, vmax = np.nanmin(matrix.values), np.nanmax(matrix.values)
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    rows = len(ordered_rows)
    fig_height = np.clip(rows * 0.40 + 1.5, 2, 12)
    fig_width = 10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, cmap="coolwarm", norm=norm, interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, label="Score")

    ax.set_xticks(np.arange(len(ordered_cols)))
    ax.set_xticklabels(ordered_cols, rotation=90)
    ax.set_yticks(np.arange(len(ordered_rows)))
    ax.set_yticklabels(ordered_rows)

    pos = 0
    for fam in row_families[:-1]:
        pos += len(row_blocks[fam])
        ax.axhline(pos - 0.5, lw=0.6, ls="--", alpha=0.5, color="black")

    pos = 0
    for fam in col_families[:-1]:
        pos += len(col_blocks[fam])
        ax.axvline(pos - 0.5, lw=0.6, ls="--", alpha=0.5, color="black")

    row_centres = []
    counter = 0
    for fam in row_families:
        n = len(row_blocks[fam])
        row_centres.append(counter + n / 2 - 0.5)
        counter += n

    trans_left = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    left_texts = [
        ax.text(-0.05, y, fam, transform=trans_left,
                ha="right", va="center", fontsize=10, weight="bold", clip_on=False)
        for y, fam in zip(row_centres, row_families)
    ]

    col_centres = []
    counter = 0
    for fam in col_families:
        n = len(col_blocks[fam])
        col_centres.append(counter + n / 2 - 0.5)
        counter += n

    base_offset = -0.05
    extra_offset = max(0, (6 - fig_height)) * -0.15
    bottom_offset = base_offset + extra_offset

    trans_bottom = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    bottom_texts = [
        ax.text(x, bottom_offset, fam,
                transform=trans_bottom,
                ha="right", va="top",
                rotation=45, rotation_mode="anchor",
                fontsize=10, weight="bold", clip_on=False)
        for x, fam in zip(col_centres, col_families)
    ]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    all_bboxes = [
        t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        for t in (left_texts + bottom_texts)
    ]
    union = mtransforms.Bbox.union(all_bboxes)

    fig.subplots_adjust(
        left=max(fig.subplotpars.left, max(0, -union.x0) + 0.01),
        bottom=max(fig.subplotpars.bottom, max(0, -union.y0) + 0.01),
    )

    ax.set_title(
        f"{metric_name} Pearson Contrib "
        f"(coef={results_long.loc[index, 'p_coef']:.3f}, "
        f"pval={results_long.loc[index, 'p_pval']:.3f})"
    )
    fig.tight_layout()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]
    fig.savefig(f"{metric_name}_pearson_contrib_{current_time}.png")
    plt.close(fig)


def analyze_wikisize(matrix, metric_name, index, model_name):
    if model_name == "bert-base-multilingual-cased":
        size_map = constants.language_wikisize
    elif model_name == "FacebookAI/xlm-roberta-base":
        size_map = constants.XLMR_SIZE_LOG
    else:
        raise ValueError(f"Model {model_name} not supported")

    rows = []
    for lang1 in matrix.index:
        for lang2 in matrix.columns:
            if lang1 == lang2:
                continue

            if lang1 not in size_map or lang2 not in size_map:
                print(f"Skipping ({lang1}, {lang2}): not in wikisize table")
                continue

            lang1_size = size_map[lang1]
            lang2_size = size_map[lang2]

            rows.append({
                "lang1_size": lang1_size,
                "lang2_size": lang2_size,
                "pearson_contrib": matrix.loc[lang1, lang2],
                "min_size": min(lang1_size, lang2_size),
                "max_size": max(lang1_size, lang2_size),
                "mean_size": (lang1_size + lang2_size) / 2,
                "diff_size": abs(lang1_size - lang2_size),
            })

    size_df = pd.DataFrame(rows, index=[index] * len(rows))

    pearson_and_spearman_df = calculate_coef_and_pval(size_df)
    print(
        f"{metric_name} ({len(size_df['pearson_contrib'].dropna())})\n"
        f"{pearson_and_spearman_df.to_markdown(floatfmt='.3f')}"
    )

    for col_name, xlabel in [("min_size", "Min Size"), ("max_size", "Max Size")]:
        plt.scatter(size_df[col_name], size_df["pearson_contrib"])
        plt.xlabel(xlabel)
        plt.ylabel("Pearson Contrib")
        plt.title(f"{metric_name} Pearson Contrib vs {xlabel}")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]
        plt.savefig(f"{metric_name}_pearson_contrib_vs_{col_name}_{current_time}.png")
        plt.close()

    # English-only analysis
    en_rows = []
    for lang1 in matrix.index:
        for lang2 in matrix.columns:
            if lang1 == lang2 or (lang1 != "en" and lang2 != "en"):
                continue

            if lang1 not in size_map or lang2 not in size_map:
                print(f"Skipping ({lang1}, {lang2}): not in wikisize table")
                continue

            lang1_size = size_map[lang1]
            lang2_size = size_map[lang2]

            en_rows.append({
                "lang1_size": lang1_size,
                "lang2_size": lang2_size,
                "pearson_contrib": matrix.loc[lang1, lang2],
                "min_size": min(lang1_size, lang2_size),
            })

    size_df = pd.DataFrame(en_rows, index=[index] * len(en_rows))

    pearson_and_spearman_df = calculate_coef_and_pval(size_df, cols=["min_size"])
    print(
        f"{metric_name} ({len(size_df['pearson_contrib'].dropna())})\n"
        f"{pearson_and_spearman_df.to_markdown(floatfmt='.3f')}"
    )

    plt.scatter(size_df["min_size"], size_df["pearson_contrib"])
    plt.xlabel("Min Size")
    plt.ylabel("Pearson Contrib")
    plt.title(f"{metric_name} Pearson Contrib vs Min Size")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]
    plt.savefig(f"{metric_name}_pearson_contrib_english{current_time}.png")
    plt.close()


def calculate_coef_and_pval(
    size_df, cols=["min_size", "mean_size", "max_size", "diff_size"]
):
    results = []
    for col in cols:
        x = size_df[col]
        y = size_df["pearson_contrib"]

        both_not_na = ~x.isna() & ~y.isna()
        x = x[both_not_na]
        y = y[both_not_na]

        if len(x) < 2:
            continue

        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)

        results.append(
            {
                "column": col,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )
    return pd.DataFrame(results).set_index("column")


def get_overlap(xnli_df, baseline, baseline_name, symmetrical=True):
    labels_a = xnli_df.index
    labels_b = baseline.index

    intersection = sorted(set(labels_a).intersection(set(labels_b)))

    df_a_sub = xnli_df.loc[intersection, intersection]
    df_b_sub = baseline.loc[intersection, intersection]

    if symmetrical:
        n = len(intersection)
        mask = np.triu(np.ones((n, n), dtype=bool), k=0)
        df_b_sub = df_b_sub.where(mask)

    series_a_sub = df_a_sub.values.flatten()
    row_labels = df_a_sub.index
    col_labels = df_a_sub.columns
    flattened_indices = [
        (row_labels[i // len(col_labels)], col_labels[i % len(col_labels)])
        for i in range(len(series_a_sub))
    ]
    series_b_sub = df_b_sub.values.flatten()
    return pd.DataFrame(
        {"fold": series_a_sub, baseline_name: series_b_sub}, index=flattened_indices
    )


def get_langs(dataset="xnli", use_bert=True):
    if dataset == "bible":
        langs = ["en"]
        for lang_2l in list(constants.FSI_SCALE.keys()):
            lang_name = (
                langcodes.Language.make(language=lang_2l).display_name()
                if lang_2l != "sl"
                else "Slovene"
            )
            first_book_lang_file = DATA_DIR / "aligned" / "1-b.GEN" / f"{lang_name}.txt"
            if first_book_lang_file.exists():
                langs.append(lang_2l)
        langs.sort()
    elif dataset == "un6":
        langs = constants.UN6_LANGS
    else:
        langs = constants.XNLI_LANGUAGES

    model_langs_2l = []
    print(f"{len(langs)=}, {langs=}")
    model_langs = (
        constants.BERT_MULTILINGUAL_LANGS if use_bert else constants.XLMR_LANGS
    )
    for lang in model_langs:
        try:
            if lang == "Norwegian (Bokmal)":
                lang_2l = "no"
            else:
                lang_2l = langcodes.Language.find_name("language", lang).language
            if lang_2l is not None:
                model_langs_2l.append(lang_2l)
            else:
                print(f"{'BERT' if use_bert else 'XLMR'} Language not found in langcodes: {lang}")
        except LookupError as e:
            print(f"{'BERT' if use_bert else 'XLMR'} Language not found in langcodes: {lang}-{e}")
            continue

    filtered_langs = [lang for lang in langs if lang in model_langs_2l]
    langs_removed = [lang for lang in langs if lang not in model_langs_2l]
    print(f"Langs: {len(filtered_langs)=}, {filtered_langs=}")
    print(f"Removed langs: {langs_removed}") if langs_removed else None
    return filtered_langs
