from pathlib import Path
from joblib import Memory
from sklearn.pipeline import Pipeline
import torch

import pandas as pd

from PsdEstimator import PsdEstimator
from PsdNormalizer import PsdNormalizer
from SpectralTransformer import SpectralTransformer
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
    memory = Memory(cachedir, verbose=0)

    torch.set_float32_matmul_precision('high')

    if torch.cuda.is_available():
        fold_globals.DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        fold_globals.DEVICE = torch.device('mps')
    else:
        fold_globals.DEVICE = torch.device('cpu')
    print('Using device:', fold_globals.DEVICE)

    pipeline = Pipeline([
            ("load_tsv", TsvToDataFrame(Path("data/XNLI-15way/xnli.15way.orig.tsv"), nrows=5000)),
            ("est_likelihood", LikelihoodEstimator()),
            ("spectra", SpectralTransformer()),
            # ("est_psd", PsdEstimator()),
            ("norm_psd", PsdNormalizer()),
            ("overlap", OverlapTransformer()),
        ],
        memory=memory,
        verbose=True
    )

    # pass None because TSVToDataFrame ignores X and reads from file_path
    output = pipeline.fit_transform(None)  
    
    print(output.shape)

    readable_names = [Language.make(language=lang).display_name() for lang in constants.LANGUAGES]

    # show_heatmap(np.average(output, axis=0), readable_names)

    en_index = constants.LANGUAGES.index('en')
    en_source_distances = np.average(output, axis=0)[en_index]
    print(en_source_distances)

    df = pd.DataFrame(columns=['lang', 'fsi', 'fold'])

    for lang, fold_distance in zip(constants.LANGUAGES, en_source_distances):
        if lang == 'en':
            continue
        df.loc[len(df)] = [lang, constants.FSI_SCALE[lang], fold_distance]
    
    # Calculate correlation between 'column_1' and 'column_2'
    correlation = df["fsi"].corr(df["fold"])

    # Print correlation value
    print("Correlation:", correlation)
