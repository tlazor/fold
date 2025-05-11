from pathlib import Path
from joblib import Memory
import langcodes
from sklearn.pipeline import Pipeline
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

from BandSelectTransformer import BandSelectTransformer
from BibleTransformer import BibleTransformer
from PsdEstimator import PsdEstimator
from PsdNormalizer import PsdNormalizer
from SampleTokens import SampleTokens
from SpectralTransformer import SpectralTransformer
from TokenTransform import TokenTransform
from LikelihoodEstimator import LikelihoodEstimator
from MetricTransformer import (
    MetricTransformer,
    kl_divergence_matrix,
)

from pipeline import get_langs

import fold_globals
import constants


def load_wals_feature(feature_id):
    """Load WALS feature data for a specific feature ID."""
    # Read the WALS CLDF data
    wals_dir = Path("data/wals-v2020.4/cldf-datasets-wals-0f5cd82/cldf")

    # Read the ValueTable which contains language-feature-value mappings
    values_df = pd.read_csv(wals_dir / "values.csv")

    # Filter for the specific feature
    feature_values = values_df[values_df["Parameter_ID"] == feature_id]

    # Create a mapping of language IDs to feature values
    lang_to_value = dict(zip(feature_values["Language_ID"], feature_values["Value"]))

    return lang_to_value


def get_langs_with_wals_feature(feature_id):
    """Get list of languages that have the specified WALS feature."""
    lang_to_value = load_wals_feature(feature_id)
    print(f"{len(lang_to_value)=}")

    # Get all available Bible languages
    langs = get_langs(use_bible=True)
    langs = [
        lang
        for lang in get_langs(use_bible=True)
        if langcodes.Language.get(lang).to_alpha3() in lang_to_value
    ]

    langs.sort()
    return langs, lang_to_value


def compute_spectra_for_langs(langs, pipeline_components):
    """Compute spectra for a list of languages using the pipeline components."""
    pipeline = Pipeline(
        pipeline_components,
        memory=Memory(".cache/joblib/tmp", verbose=0),
        verbose=True,
    )

    # Run the pipeline
    output = np.stack(pipeline.fit_transform(None))
    return output


def analyze_spectra_by_feature_value(
    output, langs, lang_to_value, plot_dir, feature_id
):
    """Analyze spectra grouped by WALS feature values."""
    # Group languages by their feature values
    value_groups = defaultdict(list)
    for lang in langs:
        alpha3 = langcodes.Language.get(lang).to_alpha3()
        if alpha3 in lang_to_value:
            value_groups[lang_to_value[alpha3]].append(lang)

    # For each feature value group
    avg_spectra = {}
    for value, group_langs in value_groups.items():
        print(f"\nAnalyzing feature value {value} with languages: {group_langs}")

        # Get indices of languages in this group
        group_indices = [langs.index(lang) for lang in group_langs]

        # Extract spectra for these languages
        group_spectra = output[:, group_indices, :]

        # Average the spectra across languages in this group
        avg_spectra[value] = np.mean(group_spectra, axis=1)  # Average across languages

    # Plot the average spectra for each feature value
    plt.figure(figsize=(12, 8))

    # Get the frequency values for x-axis
    num_freqs = output.shape[-1]
    freqs = np.linspace(0, 1, num_freqs)

    # Plot each group's average spectrum
    for value, spectrum in avg_spectra.items():
        # Average across samples to get a single spectrum
        mean_spectrum = np.mean(spectrum, axis=0)
        print(f"{mean_spectrum=}")
        plt.plot(freqs, mean_spectrum, label=f"Value {value}")

    plt.xlabel("Normalized Frequency")
    plt.ylabel("Power Spectral Density")
    plt.title(f"Average Power Spectra by WALS {feature_id} Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]
    plt.savefig(plot_dir / f"wals_{feature_id}_spectra_averages_{current_time}.png")
    plt.close()


def main():
    # Set up device
    if torch.cuda.is_available():
        fold_globals.DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        fold_globals.DEVICE = torch.device("mps")
    else:
        fold_globals.DEVICE = torch.device("cpu")
    print("Using device:", fold_globals.DEVICE)

    plot_dir = Path(".") / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    # Get mask token ID
    from transformers import AutoTokenizer

    mask_token_id = AutoTokenizer.from_pretrained(
        "bert-base-multilingual-cased", clean_up_tokenization_spaces=True
    ).mask_token_id

    # Specify WALS feature to analyze
    # feature_id = "17A"  # Example: Order of Subject, Object and Verb
    # wals_features = [f"{i}A" for i in range(1, 100)]
    wals_features = [
        "17A",
        "61A",
        "3A",
        "12A",
        "26A",
        "27A",
        "34A",
        "51A",
        "78A",
        "64A",
    ]

    for feature_id in wals_features:
        # Get languages with this feature
        langs, lang_to_value = get_langs_with_wals_feature(feature_id)
        print(f"Found {len(langs)} languages with WALS feature {feature_id}")
        print(f"{langs=}")

        # Set up pipeline components
        pipeline_components = [
            ("load_bible", BibleTransformer(Path("data/aligned"), langs=langs)),
            ("tokenize", TokenTransform()),
            ("sample", SampleTokens(num_samples=600, minimum_tokens=20, seed=0)),
            ("est_likelihood", LikelihoodEstimator(mask_token_id=mask_token_id)),
            ("est_psd", PsdEstimator(nperseg=56 * 2 - 1, axis=1)),
            ("norm_psd", PsdNormalizer(axis=1)),
        ]

        # Compute spectra
        output = compute_spectra_for_langs(langs, pipeline_components)

        print(f"{output.shape=}")

        # Analyze spectra by feature value
        analyze_spectra_by_feature_value(
            output, langs, lang_to_value, plot_dir, feature_id
        )


if __name__ == "__main__":
    main()
