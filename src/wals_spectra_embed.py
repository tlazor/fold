from collections import defaultdict
from datetime import datetime
from pathlib import Path

import langcodes
import matplotlib.pyplot as plt
import numpy as np

from BibleTransformer import BibleTransformer
from EmbedTransformer import EmbedTransformer
from pipeline_options import PipelineOptions
from PsdEstimator import PsdEstimator
from PsdNormalizer import PsdNormalizer
from SampleTokens import SampleTokens
from TokenTransform import TokenTransform
from wals_spectra import compute_spectra_for_langs, get_langs_with_wals_feature


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
        print(f"Analyzing feature value {value} with languages: {group_langs}")

        # Get indices of languages in this group
        group_indices = [langs.index(lang) for lang in group_langs]

        # Extract spectra for these languages
        print(f"{output.shape=}")
        # num_samples x num_langs x num_freqs x num_hidden_dims
        group_spectra = output[:, group_indices, :, :]
        hidden_dim_avg = np.mean(group_spectra, axis=-1)
        # hidden_dim_avg = group_spectra[:, :, :, 1]

        # Average the spectra across languages in this group
        avg_spectra[value] = np.mean(hidden_dim_avg, axis=1)  # Average across languages
        print(f"{avg_spectra[value].shape=}")

    # Plot the average spectra for each feature value
    plt.figure(figsize=(12, 8))

    # Get the frequency values for x-axis
    num_freqs = output.shape[2]
    freqs = np.linspace(0, 1, num_freqs)

    # Plot each group's average spectrum
    for value, spectrum in avg_spectra.items():
        # Average across samples to get a single spectrum
        mean_spectrum = np.mean(spectrum, axis=0)
        print(f"{mean_spectrum=}")
        plt.plot(freqs, mean_spectrum, label=f"Value {value}")

    plt.xlabel("Normalized Frequency")
    plt.ylabel("Power Spectral Density")
    plt.title(f"Average Power Spectra by WALS {feature_id} Value (Embeddings)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]
    plt.savefig(plot_dir / f"wals_{feature_id}_spectra_embeddings_{current_time}.png")
    plt.close()


def main():
    config = PipelineOptions()

    plot_dir = Path(".") / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    # Specify WALS feature to analyze
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

        # Set up pipeline components using shared config
        pipeline_components = [
            ("load_bible", BibleTransformer(Path("data/aligned"), langs=langs)),
            ("tokenize", TokenTransform(model_name=config.model_name)),
            ("sample", SampleTokens(num_samples=600, minimum_tokens=20, seed=0)),
            (
                "embeddings",
                EmbedTransformer(model_name=config.model_name, layer=config.layers[0]),
            ),  # Using first layer from config
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
