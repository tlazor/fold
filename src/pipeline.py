from pathlib import Path
from joblib import Memory
from sklearn.pipeline import Pipeline
import torch

import pandas as pd

from KlTransformer import KlTransformer
from PsdEstimator import PsdEstimator
from PsdNormalizer import PsdNormalizer
from SampleTokens import SampleTokens
from SpectralTransformer import SpectralTransformer
from TokenTransform import TokenTransform
from TsvToDataFrame import TsvToDataFrame
from LikelihoodEstimator import LikelihoodEstimator
from OverlapTransformer import OverlapTransformer

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
        ax = sns.heatmap(overlap_matrix, annot=True, fmt=".2f",
                         xticklabels=lang_labels, yticklabels=lang_labels,
                         cmap="Blues")
    else:
        ax = sns.heatmap(overlap_matrix, annot=True, fmt=".2f", cmap="Blues")

    ax.set_title("Overlap Matrix Heatmap")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cachedir = Path(".cache/joblib")
    pipeline_memory = Memory(cachedir, verbose=0)

    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.capture_scalar_outputs = True

    if torch.cuda.is_available():
        fold_globals.DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        fold_globals.DEVICE = torch.device('mps')
    else:
        fold_globals.DEVICE = torch.device('cpu')
    print('Using device:', fold_globals.DEVICE)

    from transformers import AutoTokenizer
    mask_token_id = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", clean_up_tokenization_spaces=True).mask_token_id

    pipeline = Pipeline([
            ("load_tsv", TsvToDataFrame(Path("data/XNLI-15way/xnli.15way.orig.tsv"))),
            ("tokenize", TokenTransform()),
            ("sample", SampleTokens(num_samples=500, minimum_tokens=20, seed=0)),
            ("est_likelihood", LikelihoodEstimator(mask_token_id=mask_token_id)),
            # ("spectra", SpectralTransformer()),
            # ("est_psd", PsdEstimator()),
            # ("norm_psd", PsdNormalizer()),
            # ("overlap", OverlapTransformer()),
            # ("KL Divergence", KlTransformer())
        ],
        memory=pipeline_memory,
        verbose=True
    )

    # pass None because TSVToDataFrame ignores X and reads from file_path
    output = pipeline.fit_transform(None)  
    
    print(output.shape)

    readable_names = [Language.make(language=lang).display_name() for lang in constants.LANGUAGES]

    # show_heatmap(np.average(output, axis=0), readable_names)

    en_index = constants.LANGUAGES.index('en')
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
            axes[i].hist(en_distances[:, i], bins=100, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution {constants.LANGUAGES[i]}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')

        # Adjust layout
        plt.tight_layout()
        plt.show()


    en_source_distances = np.median(output, axis=0)[en_index]
    df = pd.DataFrame(columns=['lang', 'fsi', 'fold', 'ger_int'])

    for lang, fold_distance in zip(constants.LANGUAGES, en_source_distances):
        if lang == 'en':
            continue
        if lang in constants.GERMANIC_INTELLIGABILITY.columns:
            ger_int = constants.GERMANIC_INTELLIGABILITY.loc[lang, "en"]
        else:
            ger_int = np.nan
        df.loc[len(df)] = [lang, constants.FSI_SCALE[lang], fold_distance, ger_int]
    
    
    df.set_index("lang", inplace=True)
    print(df)
    # Print correlation value
    print(f"Correlation:\n{round(df.corr(), 2)}")
