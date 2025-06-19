from datetime import datetime
from itertools import combinations
from pathlib import Path
from joblib import Memory
import langcodes
from matplotlib.colors import TwoSlopeNorm
from sklearn.pipeline import Pipeline
import torch
from scipy.stats import pearsonr, spearmanr
from rich.progress import track
from functools import partial
from transformers import AutoTokenizer
from scipy.stats import pearsonr, spearmanr, rankdata

import pandas as pd

from BandSelectTransformer import BandSelectTransformer
from BibleTransformer import BibleTransformer
from PsdEstimator import PsdEstimator
from PsdNormalizer import PsdNormalizer
from SampleTokens import SampleTokens
from SpectralTransformer import SpectralTransformer
from TokenTransform import TokenTransform
from TsvToDataFrame import TsvToDataFrame
from LikelihoodEstimator import LikelihoodEstimator
from MetricTransformer import (
    MetricTransformer,
    coherence_matrix,
    compute_overlaps,
    kl_divergence_matrix,
    mae_matrix,
)
from Un6Transformer import Un6Transformer

import fold_globals
import constants

from langcodes import Language

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def show_heatmap(overlap_matrix, lang_labels=None):
    """
    Display a heatmap of the overlap_matrix.

    Parameters
    ----------
    overlap_matrix : np.ndarray of shape (num_langs, num_langs)
        The pairwise overlap or comparison matrix.
    lang_labels : list of str, optional
        Labels for each language, to use on the axes.
    """
    # Create a figure and axis
    plt.figure(figsize=(6, 5))

    # If provided, set the x and y tick labels
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


def calculate_correlations(dataframes):
    all_correlations = []

    for df_name, df in dataframes.items():
        # Check if df has at least 2 non-NA pairs
        if df.dropna().shape[0] < 2:
            print(f"{df_name=} has less than 2 non-NA pairs")
            continue
        for corr_name, corr_func in [("pearson", pearsonr), ("spearman", spearmanr)]:
            corr = df.corr(method=lambda x, y: corr_func(x, y)[0])
            pvals = df.corr(method=lambda x, y: corr_func(x, y)[1]) - np.eye(
                len(df.columns)
            )

            # Flatten into long form
            for i, row in enumerate(corr.index):
                for j in range(i + 1, len(corr.columns)):
                    col = corr.columns[j]
                    # Skip diagonal or collect it as well
                    if row != col:
                        all_correlations.append(
                            {
                                # "df_name": df_name,
                                "corr_type": corr_name,
                                "metric": col if col != "fold" else row,
                                "coef": corr.loc[row, col],
                                "pval": pvals.loc[row, col],
                                "num_points": df.dropna().shape[0],
                            }
                        )

    # Convert to a single DataFrame
    results_long = pd.DataFrame(all_correlations)

    # Pivot the table so pearson/spearman become columns
    pivoted = results_long.pivot_table(
        index="metric", columns="corr_type", values=["coef", "pval"], aggfunc="first"
    )

    # Flatten the multi-index columns
    pivoted.columns = [
        f"{stat[:1]}_{ptype}"  # => p_coef, p_pval, s_coef, s_pval
        for ptype, stat in pivoted.columns
    ]

    # Bring in num_points (same for both rows of a given metric, so we can just pick .first())
    num_points = results_long.groupby("metric")["num_points"].first()

    # Merge them together and reset_index so metric is a column again
    out = pivoted.join(num_points).reset_index()

    # Re-order the columns if needed
    out = out[["metric", "p_coef", "p_pval", "s_coef", "s_pval", "num_points"]]
    return out


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

                # Pearson correlation
                p_coef, p_pval = pearsonr(x, y)
                x_mean, y_mean = x.mean(), y.mean()
                pointwise_pearson = ((x - x_mean) * (y - y_mean)) / np.sqrt(
                    ((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum()
                )

                # print(f"{pointwise_pearson=}")

                # Spearman correlation
                s_coef, s_pval = spearmanr(x, y)

                # Concordant and discordant pairs
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
        .drop(["Mean", "Total"])
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

    ful_mut_int = df1_re.combine_first(df2_re).combine_first(df3_re)
    # print(ful_mut_int)
    return ful_mut_int


def star_fmt(p: float) -> str:
    """Return a string with significance stars appended."""
    if p < 0.001:
        return f"{p:.3f}***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"


formatters = {c: star_fmt for c in ["p_pval", "s_pval"]}


def analyze_output(output, langs, f=None, model_name=None):
    readable_names = [Language.make(language=lang).display_name() for lang in langs]

    # show_heatmap(np.average(output, axis=0), readable_names)

    en_index = langs.index("en")
    show_plot = False

    if show_plot:
        en_distances = output[:, en_index, :]
        fig, axes = plt.subplots(5, 3, figsize=(15, 20))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Plot distributions
        for i in range(15):
            if i == en_index:
                continue
            axes[i].hist(en_distances[:, i], bins=100, alpha=0.7, edgecolor="black")
            axes[i].set_title(f"Distribution {langs[i]}")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")

        # Adjust layout
        plt.tight_layout()
        plt.show()

    xnli_df = pd.DataFrame(
        np.median(output, axis=0),
        index=langs,
        columns=langs,
    )
    # Diagonal values are 1, so the correlation is less useful when including them
    # Set all diagonals to NA
    for i in range(0, len(langs)):
        xnli_df.iloc[i, i] = np.nan
    # print(f"{xnli_df=}")

    en_source_distances = np.nanmedian(output, axis=0)[en_index]
    df_fsi = pd.DataFrame(columns=["lang", "fsi", "fold"])

    for lang, fold_distance in zip(langs, en_source_distances):
        if lang == "en":
            continue

        df_fsi.loc[len(df_fsi)] = [lang, constants.FSI_SCALE[lang], fold_distance]

    df_fsi.set_index("lang", inplace=True)

    ######################## Mutual Intelligibility ########################
    df_mut_int = get_overlap(xnli_df, get_full_mut_int(), "mut_int", symmetrical=False)
    ########################### Lexical Distance ###########################
    df_lex_sim = get_overlap(xnli_df, constants.LEXICAL_SIMILARITY, "lex_sim")
    ########################### Phonetic Distance ##########################
    df_pho_sim = get_overlap(xnli_df, constants.PHONETIC_SIMILARITY, "pho_sim")
    ########################################################################

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

    analyze_pearson_contrib(results_long, model_name)


def analyze_pearson_contrib(results_long, model_name):
    # Get pearson contrib
    df = results_long["pearson_contrib"]

    # iterate through each metric in pearson_contrib (pd.Series)
    for index, metric_pearson_contrib in df.items():
        # Get metric name
        metric_name = results_long.loc[index, "metric"]

        langs = set()
        
        if isinstance(metric_pearson_contrib.index[0], tuple):
            for index_langs in metric_pearson_contrib.index:
                langs.update(index_langs)

            langs = sorted(langs)
            # Initialize a square matrix
            matrix = pd.DataFrame(np.nan, index=langs, columns=langs)

            for index_langs in metric_pearson_contrib.index:
                matrix.loc[index_langs] = metric_pearson_contrib[index_langs]
                
            plot_pearson_contrib(matrix, metric_name, results_long, index)
            analyze_wikisize(matrix, metric_name, results_long, index, model_name)
        else:
            langs = sorted(metric_pearson_contrib.index) 
            # if it's not a tuple, it's a single language like FSI scale
            # analyze en row and col separately
            en_row = pd.DataFrame(np.nan, index=langs, columns=langs)
            en_col = pd.DataFrame(np.nan, index=langs, columns=langs)
            print(f"{metric_pearson_contrib.index=}")
            en_row.loc["en"] = metric_pearson_contrib[langs]
            # en_col.loc[:, "en"] = [metric_pearson_contrib[(lang_a, "en")] for lang_a in langs]
            print(f"{en_row=}")
            
            plot_pearson_contrib(en_row, f"{metric_name}_en_row", results_long, index)
            analyze_wikisize(en_row, f"{metric_name}_en_row", results_long, index, model_name)


def plot_pearson_contrib(matrix, metric_name, results_long, index):
    # Compute normalization to center at 0
    vmin = np.min(matrix.values)
    vmax = np.max(matrix.values)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap="coolwarm", norm=norm, interpolation="nearest")
    plt.colorbar(label="Score")
    plt.xticks(ticks=np.arange(len(matrix.columns)), labels=matrix.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(matrix.index)), labels=matrix.index)
    plt.title(
        f"{metric_name} Pearson Contrib (coef={results_long.loc[index, 'p_coef']:.3f}, pval={results_long.loc[index, 'p_pval']:.3f})"
    )
    plt.tight_layout()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]
    plt.savefig(f"{metric_name}_pearson_contrib_{current_time}.png")
    plt.close()


def analyze_wikisize(matrix, metric_name, index, model_name):
    # iterate through each language pair in matrix
    size_df = pd.DataFrame()
    for lang1 in matrix.index:
        for lang2 in matrix.columns:
            if lang1 == lang2:
                continue

            # print(f"{lang1=}, {lang2=}")

            if model_name == "bert-base-multilingual-cased":
                lang1_size = constants.language_wikisize[lang1]
                lang2_size = constants.language_wikisize[lang2]
            elif model_name == "FacebookAI/xlm-roberta-base":
                lang1_size = constants.XLMR_SIZE_LOG[lang1]
                lang2_size = constants.XLMR_SIZE_LOG[lang2]
            else:
                raise ValueError(f"Model {model_name} not supported")

            min_size = min(lang1_size, lang2_size)
            max_size = max(lang1_size, lang2_size)
            mean_size = (lang1_size + lang2_size) / 2
            diff_size = abs(lang1_size - lang2_size)

            # print(f"{lang1_size=} {lang2_size=}")

            # get the pearson contrib for the two languages
            pearson_contrib = matrix.loc[lang1, lang2]
            # print(f"{pearson_contrib=}")

            # Add contrib and wikisize to new dataframe
            size_df = pd.concat(
                [
                    size_df,
                    pd.DataFrame(
                        {
                            "lang1_size": lang1_size,
                            "lang2_size": lang2_size,
                            "pearson_contrib": pearson_contrib,
                            "min_size": min_size,
                            "max_size": max_size,
                            "mean_size": mean_size,
                            "diff_size": diff_size,
                        },
                        index=[index],
                    ),
                ]
            )

    pearson_and_spearman_df = calculate_coef_and_pval(size_df)
    print(
        f"{metric_name} ({len(size_df['pearson_contrib'].dropna())})\n{pearson_and_spearman_df.to_markdown(floatfmt='.3f')}"
    )

    # create a scatter plot of pearson_contrib vs min_size
    plt.figure(figsize=(10, 8))
    plt.scatter(size_df["min_size"], size_df["pearson_contrib"])
    plt.xlabel("Min Size")
    plt.ylabel("Pearson Contrib")
    plt.title(f"{metric_name} Pearson Contrib vs Min Size")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]
    plt.savefig(f"{metric_name}_pearson_contrib_vs_min_size_{current_time}.png")
    plt.close()
    # create a scatter plot of pearson_contrib vs max_size
    plt.figure(figsize=(10, 8))
    plt.scatter(size_df["max_size"], size_df["pearson_contrib"])
    plt.xlabel("Max Size")
    plt.ylabel("Pearson Contrib")
    plt.title(f"{metric_name} Pearson Contrib vs Max Size")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]
    plt.savefig(f"{metric_name}_pearson_contrib_vs_max_size_{current_time}.png")
    plt.close()

    # analyze effect with english always as one of the languages
    size_df = pd.DataFrame()

    for lang1 in matrix.index:
        for lang2 in matrix.columns:
            if lang1 == lang2:
                continue
            if lang1 != "en" and lang2 != "en":
                continue

            if model_name == "bert-base-multilingual-cased":
                lang1_size = constants.language_wikisize[lang1]
                lang2_size = constants.language_wikisize[lang2]
            elif model_name == "FacebookAI/xlm-roberta-base":
                lang1_size = constants.XLMR_SIZE_LOG[lang1]
                lang2_size = constants.XLMR_SIZE_LOG[lang2]
            else:
                raise ValueError(f"Model {model_name} not supported")

            min_size = min(lang1_size, lang2_size)

            # get the pearson contrib for the two languages
            pearson_contrib = matrix.loc[lang1, lang2]
            # print(f"{pearson_contrib=}")

            # Add contrib and wikisize to new dataframe
            size_df = pd.concat(
                [
                    size_df,
                    pd.DataFrame(
                        {
                            "lang1_size": lang1_size,
                            "lang2_size": lang2_size,
                            "pearson_contrib": pearson_contrib,
                            "min_size": min_size,
                        },
                        index=[index],
                    ),
                ]
            )

    pearson_and_spearman_df = calculate_coef_and_pval(size_df, cols=["min_size"])
    print(
        f"{metric_name} ({len(size_df['pearson_contrib'].dropna())})\n{pearson_and_spearman_df.to_markdown(floatfmt='.3f')}"
    )

    # create a scatter plot of pearson_contrib vs min_size
    plt.figure(figsize=(10, 8))
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
    # Loop through each column to correlate with 'pearson_contrib'
    for col in cols:
        x = size_df[col]
        y = size_df["pearson_contrib"]

        both_not_na = ~x.isna() & ~y.isna()
        x = x[both_not_na]
        y = y[both_not_na]

        if len(x) < 2:
            continue  # skip columns with too few data points

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

    # 1. Identify overlapping labels:
    intersection = list(set(labels_a).intersection(set(labels_b)))
    intersection.sort()  # sort for consistent ordering

    # This creates a boolean mask for the upper triangle (including diagonal)
    upper_triangle_mask = np.triu(np.ones(baseline.shape), k=0).astype(bool)

    # Use .where() with the mask so that values outside the upper triangle become NaN
    baseline_df = baseline.where(upper_triangle_mask) if symmetrical else baseline

    # 2. Subset and reorder each distance matrix
    df_a_sub = xnli_df.loc[intersection, intersection]
    df_b_sub = baseline_df.loc[intersection, intersection]

    series_a_sub = df_a_sub.values.flatten()
    # Store row and column indices using actual DataFrame labels
    row_labels = df_a_sub.index
    col_labels = df_a_sub.columns
    flattened_indices = [
        (row_labels[i // len(col_labels)], col_labels[i % len(col_labels)])
        for i in range(len(series_a_sub))
    ]
    # print(f"{flattened_indices=}")
    series_b_sub = df_b_sub.values.flatten()
    df = pd.DataFrame(
        {"fold": series_a_sub, baseline_name: series_b_sub}, index=flattened_indices
    )

    return df


def get_langs(use_bible=False, use_un6=False, use_bert=True):
    if use_bible:
        langs = ["en"]
        for lang_2l in list(constants.FSI_SCALE.keys()):
            lang_name = (
                langcodes.Language.make(language=lang_2l).display_name()
                if lang_2l != "sl"
                else "Slovene"
            )
            first_book_lang_file = Path("data/aligned/1-b.GEN") / f"{lang_name}.txt"
            if first_book_lang_file.exists():
                langs.append(lang_2l)
            # else:
            #     print(f"Excluding: {lang_2l}/{lang_name}")
        langs.sort()
    elif use_un6:
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
            # BERT supports "Norwegian (Bokmal)" and "Norwegian (Nynorsk)" but langcodes doesn't
            if lang == "Norwegian (Bokmal)":
                lang_2l = "no"
            else:
                lang_2l = langcodes.Language.find_name("language", lang).language
            if lang_2l is not None:
                model_langs_2l.append(lang_2l)
            else:
                print(
                    f"{'BERT' if use_bert else 'XLMR'} Language not found in langcodes: {lang}"
                )
        except LookupError as e:
            print(
                f"{'BERT' if use_bert else 'XLMR'} Language not found in langcodes: {lang}-{e}"
            )
            continue

    # Filter out languages that are not in the model
    filtered_langs = [lang for lang in langs if lang in model_langs_2l]
    # Get languages that were removed
    langs_removed = [lang for lang in langs if lang not in model_langs_2l]
    print(f"Langs: {len(filtered_langs)=}, {filtered_langs=}")
    print(f"Removed langs: {langs_removed}") if langs_removed else None
    return filtered_langs


if __name__ == "__main__":
    # Clear CUDA cache and set memory management
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(
            0.8
        )  # Use 80% of available GPU memory
        torch.cuda.memory.set_per_process_memory_fraction(0.8)
        # Enable memory efficient attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True

    if torch.cuda.is_available():
        fold_globals.DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        fold_globals.DEVICE = torch.device("mps")
    else:
        fold_globals.DEVICE = torch.device("cpu")
    print("Using device:", fold_globals.DEVICE)

    use_bible = True
    use_un6 = False
    use_spectra = True
    straight_spectra = False
    use_bert = True
    model_name = (
        "bert-base-multilingual-cased" if use_bert else "FacebookAI/xlm-roberta-base"
    )

    cachedir = Path(f".cache/joblib/tmp/{'bert' if use_bert else 'xlmr'}")
    pipeline_memory = Memory(cachedir, verbose=0)

    mask_token_id = AutoTokenizer.from_pretrained(
        model_name, clean_up_tokenization_spaces=True
    ).mask_token_id
    # layers = range(1, 12)
    num_bands = 1
    if num_bands > 1:
        beginning_freqs = np.linspace(0, 1, num=num_bands, endpoint=False)
        freq_bands = zip(
            beginning_freqs, np.linspace(beginning_freqs[1], 1, num=num_bands)
        )
    else:
        freq_bands = [(0, 1)]

    # print config options
    print(
        f"{use_bible=}, {use_un6=}, {use_spectra=}, {straight_spectra=}, {use_bert=}, {model_name=}"
    )

    langs = get_langs(use_bible, use_un6, use_bert)
    likelihood_pipeline_components = [
        ("load_bible", BibleTransformer(Path("data/aligned"), langs=langs))
        if use_bible
        else ("load_un6", Un6Transformer(Path("data/6way"), langs=langs, nrows=2000))
        if use_un6
        else ("load_tsv", TsvToDataFrame(Path("data/XNLI-15way/xnli.15way.orig.tsv"))),
        ("tokenize", TokenTransform(model_name=model_name)),
        ("sample", SampleTokens(num_samples=600, minimum_tokens=20, seed=0)),
        (
            "est_likelihood",
            LikelihoodEstimator(model_name=model_name, mask_token_id=mask_token_id),
        ),
    ]

    spectra_component = [
        *(
            # if is_spectra is True, we add just the SpectralTransformer
            [("spectra", SpectralTransformer())]
            if straight_spectra
            # otherwise, we add the two PSD-related transforms
            else [("est_psd", PsdEstimator()), ("norm_psd", PsdNormalizer())]
        )
    ]

    # metric_funs = [compute_overlaps, kl_divergence_matrix, mae_matrix, coherence_fun]
    # metric_funs = [kl_divergence_matrix]

    short_model_name = "bert" if use_bert else "xlmr"
    f = None
    try:
        prefix = "bible" if use_bible else "un6" if use_un6 else "xnli"
        f = open(
            Path(f"./{prefix}_{short_model_name}_likelihood_output.txt"),
            "w+",
            encoding="utf-8",
        )
        # print config options to file
        print(
            f"{use_bible=}, {use_un6=}, {use_spectra=}, {straight_spectra=}, {use_bert=}, {model_name=}",
            file=f,
        )
        for band in freq_bands:
            coherence_fun = partial(coherence_matrix, nperseg=10, freq_band=band)
            coherence_fun.__name__ = "coherence_fun"
            metric_funs = [compute_overlaps, kl_divergence_matrix, coherence_fun]

            print(f"{band=}", file=f)
            band_component = (
                f"{band[0]:.3f}-{band[1]:.3f} selector",
                BandSelectTransformer(freq_band=band),
            )
            for fun in metric_funs:
                metric_transformer = MetricTransformer(
                    name=fun.__name__, metric_fun=fun, verbose=True
                )
                metric_component = (metric_transformer.name, metric_transformer)
                pipeline = Pipeline(
                    likelihood_pipeline_components
                    + (
                        spectra_component
                        if use_spectra and fun != coherence_fun
                        else []
                    )
                    + ([band_component] if fun != coherence_fun else [])
                    + [metric_component],
                    memory=pipeline_memory,
                    verbose=True,
                )

                # pass None because TSVToDataFrame ignores X and reads from file_path
                output = pipeline.fit_transform(None)
                print(metric_transformer.name, output.shape, file=f)

                analyze_output(output, langs, f=f, model_name=model_name)
    finally:
        if f is not None:
            f.close()
