from pathlib import Path
from joblib import Memory
import langcodes
from sklearn.pipeline import Pipeline
import torch
from scipy.stats import pearsonr, spearmanr
from rich.progress import track
from functools import partial
from transformers import AutoTokenizer

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


def analyze_output(output, langs, f=None):
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
    results_long = calculate_correlations(dataframes)
    print(
        results_long.to_markdown(index=False, floatfmt=".3f", tablefmt="github"),
        file=f,
        flush=True,
    )


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
    series_b_sub = df_b_sub.values.flatten()
    df = pd.DataFrame({"fold": series_a_sub, baseline_name: series_b_sub})

    return df

def get_langs(use_bible=False, use_bert=True):
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
    else:
        langs = constants.XNLI_LANGUAGES
    bert_langs_2l = []
    print(f"{len(langs)=}, {langs=}")
    model_langs = constants.BERT_MULTILINGUAL_LANGS if use_bert else constants.XLMR_LANGS
    for lang in model_langs:
        try:
            # BERT supports "Norwegian (Bokmal)" and "Norwegian (Nynorsk)" but langcodes doesn't
            if lang == "Norwegian (Bokmal)":
                lang_2l = "no"
            else:
                lang_2l = langcodes.Language.find_name('language', lang).language
            if lang_2l is not None:
                bert_langs_2l.append(lang_2l)
            else:
                print(f"{'BERT' if use_bert else 'XLMR'} Language not found in langcodes: {lang}")
        except LookupError as e:
            print(f"{'BERT' if use_bert else 'XLMR'} Language not found in langcodes: {lang}-{e}")
            continue

    langs = [lang for lang in langs if lang in bert_langs_2l]
    langs_removed = [lang for lang in langs if lang not in bert_langs_2l]
    print(f"langs after filtering: {len(langs)=}, {langs=}")
    print(f"Removed langs: {langs_removed}") if langs_removed else None
    return langs


if __name__ == "__main__":
    cachedir = Path(".cache/joblib/tmp/coherence_pipeline")
    pipeline_memory = Memory(cachedir, verbose=0)

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
    use_spectra = True
    straight_spectra = False
    use_bert = True
    model_name = "bert-base-multilingual-cased" if use_bert else "FacebookAI/xlm-roberta-base"

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
    print(f"{use_bible=}, {use_spectra=}, {straight_spectra=}, {use_bert=}, {model_name=}")


    langs = get_langs(use_bible, use_bert)
    likelihood_pipeline_components = [
        ("load_bible", BibleTransformer(Path("data/aligned"), langs=langs))
        if use_bible
        else ("load_tsv", TsvToDataFrame(Path("data/XNLI-15way/xnli.15way.orig.tsv"))),
        ("tokenize", TokenTransform(model_name=model_name)),
        ("sample", SampleTokens(num_samples=600, minimum_tokens=20, seed=0)),
        ("est_likelihood", LikelihoodEstimator(model_name=model_name, mask_token_id=mask_token_id)),
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
    f = open(Path(f"./{short_model_name}_likelihood_output.txt"), "w+", encoding="utf-8")
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
            metric_transformer = MetricTransformer(name=fun.__name__, metric_fun=fun, verbose=True)
            metric_component = (metric_transformer.name, metric_transformer)
            pipeline = Pipeline(
                likelihood_pipeline_components
                + (spectra_component if use_spectra and fun != coherence_fun else [])
                + ([band_component] if fun != coherence_fun else [])
                + [metric_component],
                memory=pipeline_memory,
                verbose=True,
            )

            # pass None because TSVToDataFrame ignores X and reads from file_path
            output = pipeline.fit_transform(None)
            print(metric_transformer.name, output.shape, file=f)

            analyze_output(output, langs, f=f)
    f.close()
