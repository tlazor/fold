from pathlib import Path
from joblib import Memory
from sklearn.pipeline import Pipeline
import torch
from scipy.stats import pearsonr, spearmanr

import pandas as pd

from PsdEstimator import PsdEstimator
from PsdNormalizer import PsdNormalizer
from SampleTokens import SampleTokens
from SpectralTransformer import SpectralTransformer
from TokenTransform import TokenTransform
from TsvToDataFrame import TsvToDataFrame
from LikelihoodEstimator import LikelihoodEstimator
from MetricTransformer import MetricTransformer, compute_overlaps, kl_divergence_matrix, mae_matrix

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


def analyze_output(output):
    readable_names = [
        Language.make(language=lang).display_name() for lang in constants.LANGUAGES
    ]

    # show_heatmap(np.average(output, axis=0), readable_names)

    en_index = constants.LANGUAGES.index("en")
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
            axes[i].set_title(f"Distribution {constants.LANGUAGES[i]}")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")

        # Adjust layout
        plt.tight_layout()
        plt.show()

    en_source_distances = np.median(output, axis=0)[en_index]
    df = pd.DataFrame(columns=["lang", "fsi", "fold"])
    # df = pd.DataFrame(columns=['lang', 'fsi', 'fold', 'mut_int'])

    for lang, fold_distance in zip(constants.LANGUAGES, en_source_distances):
        if lang == "en":
            continue

        #     mut_int = np.nan
        #     for mutual_intelligibilty in [constants.GERMANIC_INTELLIGABILITY, constants.ROMANCE_INTELLIGABILITY]:
        #         if lang in mutual_intelligibilty.columns:
        #             mut_int = mutual_intelligibilty.loc[lang, "en"]

        df.loc[len(df)] = [lang, constants.FSI_SCALE[lang], fold_distance]

    df.set_index("lang", inplace=True)
    # print(df)
    # Print correlation value
    for corr_name, corr_func in [("pearson", pearsonr), ("spearman", spearmanr)]:
        corr = df.corr(method=lambda x, y: corr_func(x, y)[0])
        pvalues = df.corr(method=lambda x, y: corr_func(x, y)[1]) - np.eye(
            len(df.columns)
        )

        combined_df = pd.DataFrame("", index=corr.index, columns=corr.columns)

        # Pair each correlation coefficient with its p-value in string form
        for row in corr.index:
            for col in corr.columns:
                coef = corr.loc[row, col]
                pval = pvalues.loc[row, col]
                combined_df.loc[row, col] = f"{coef:.2f} (p={pval:.2f})"

        print(f"{corr_name} correlation:\n{combined_df}")


if __name__ == "__main__":
    cachedir = Path(".cache/joblib")
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

    from transformers import AutoTokenizer

    mask_token_id = AutoTokenizer.from_pretrained(
        "bert-base-multilingual-cased", clean_up_tokenization_spaces=True
    ).mask_token_id

    likelihood_pipeline_components = [
            ("load_tsv", TsvToDataFrame(Path("data/XNLI-15way/xnli.15way.orig.tsv"))),
            ("tokenize", TokenTransform()),
            ("sample", SampleTokens(num_samples=500, minimum_tokens=20, seed=0)),
            ("est_likelihood", LikelihoodEstimator(mask_token_id=mask_token_id)),
            # ("spectra", SpectralTransformer()),
            ("est_psd", PsdEstimator()),
            ("norm_psd", PsdNormalizer()),
        ]
    
    
    metric_funs = [compute_overlaps, kl_divergence_matrix, mae_matrix]
    for fun in metric_funs:
        metric_transformer = MetricTransformer(name=fun.__name__, metric_fun=fun)
        metric_component = (metric_transformer.name, metric_transformer)
        pipeline = Pipeline(
            likelihood_pipeline_components + [metric_component],
            memory=pipeline_memory,
            verbose=False,
        )

        # pass None because TSVToDataFrame ignores X and reads from file_path
        output = pipeline.fit_transform(None)

        print(metric_transformer.name, output.shape)

        analyze_output(output)
