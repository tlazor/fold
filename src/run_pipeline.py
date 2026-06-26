"""
FOLD unified pipeline entry point.

Replaces the separate pipeline.py / embed_pipeline.py scripts.

Usage:
    uv run fold-pipeline --signal-mode likelihood [OPTIONS]
    uv run fold-pipeline --signal-mode embedding  [OPTIONS]
    uv run python src/run_pipeline.py --help
"""

from functools import partial
from pathlib import Path

from joblib import Memory
from sklearn.pipeline import Pipeline
import torch

from BandSelectTransformer import BandSelectTransformer
from BibleTransformer import BibleTransformer
from EmbedTransformer import EmbedTransformer
from LikelihoodEstimator import LikelihoodEstimator
from MetricTransformer import (
    MetricTransformer,
    coherence_matrix,
    compute_overlaps,
    kl_divergence_matrix,
)
from NoOpTransformer import NoOpTransformer
from PsdEstimator import PsdEstimator
from PsdNormalizer import PsdNormalizer
from SampleTokens import SampleTokens
from SpectralTransformer import SpectralTransformer
from TokenTransform import TokenTransform
from TsvToDataFrame import TsvToDataFrame
from Un6Transformer import Un6Transformer
from pipeline_options import PipelineOptions
from analysis import analyze_output, get_langs
from paths import DATA_DIR, auto_device


def _build_data_components(config, langs):
    if config.dataset == "bible":
        loader = ("load_bible", BibleTransformer(DATA_DIR / "aligned", langs=langs))
    elif config.dataset == "un6":
        loader = ("load_un6", Un6Transformer(DATA_DIR / "6way", langs=langs, nrows=2000))
    else:
        loader = ("load_tsv", TsvToDataFrame(DATA_DIR / "XNLI-15way" / "xnli.15way.orig.tsv"))
    return [
        loader,
        ("tokenize", TokenTransform(model_name=config.model_name)),
        ("sample", SampleTokens(num_samples=600, minimum_tokens=20, seed=0)),
    ]


def _build_spectra_components(config):
    """Spectral transform step; PSD parameters differ between signal modes."""
    if config.spectral_mode == "none":
        return [("noop", NoOpTransformer())]
    if config.spectral_mode == "fft":
        return [("spectra", SpectralTransformer())]
    # welch PSD — embedding uses a longer window and explicit axis
    if config.signal_mode == "embedding":
        return [
            ("est_psd", PsdEstimator(nperseg=56 * 2 - 1, axis=1)),
            ("norm_psd", PsdNormalizer(axis=1)),
        ]
    return [("est_psd", PsdEstimator()), ("norm_psd", PsdNormalizer())]


def main():
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True

    device = auto_device()
    print("Using device:", device)

    config = PipelineOptions.from_args()
    pipeline_memory = Memory(config.cachedir, verbose=0)
    config.print_config()

    langs = get_langs(config.dataset, config.use_bert)
    data_components = _build_data_components(config, langs)
    spectra_components = _build_spectra_components(config)

    output_path = Path(config.get_output_filename())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(output_path.with_suffix(".json"))

    with open(output_path, "w+", encoding="utf-8") as f:
        config.print_config(file=f)

        for band in config.freq_bands:
            coherence_fun = partial(coherence_matrix, nperseg=10, freq_band=band)
            coherence_fun.__name__ = "coherence_fun"
            metric_funs = (
                [compute_overlaps, kl_divergence_matrix]
                if config.spectral_mode == "none"
                else [compute_overlaps, kl_divergence_matrix, coherence_fun]
            )

            print(f"{band=}", file=f)
            band_component = (
                f"{band[0]:.3f}-{band[1]:.3f} selector",
                BandSelectTransformer(freq_band=band),
            )

            for fun in metric_funs:
                metric_transformer = MetricTransformer(
                    name=fun.__name__,
                    metric_fun=fun,
                    verbose=(fun == coherence_fun),
                )
                metric_component = (metric_transformer.name, metric_transformer)
                print(metric_transformer.name, file=f)

                # likelihood runs once (layer=None); embedding loops over layers
                layers = config.layers if config.signal_mode == "embedding" else [None]

                for layer in layers:
                    if config.signal_mode == "embedding":
                        signal_component = (
                            "embeddings",
                            EmbedTransformer(
                                layer=layer,
                                model_name=config.model_name,
                                device=device,
                            ),
                        )
                    else:
                        signal_component = (
                            "est_likelihood",
                            LikelihoodEstimator(
                                model_name=config.model_name,
                                mask_token_id=config.mask_token_id,
                                device=device,
                            ),
                        )

                    pipeline = Pipeline(
                        data_components
                        + [signal_component]
                        + (spectra_components if fun != coherence_fun else [])
                        + ([band_component] if fun != coherence_fun else [])
                        + [metric_component],
                        memory=pipeline_memory,
                        verbose=True,
                    )
                    output = pipeline.fit_transform(None)

                    if layer is not None:
                        print(f"{layer=}", file=f)
                    analyze_output(
                        output, langs, f=f,
                        model_name=config.model_name,
                        flag_analyze_pearson_contrib=config.analyze_pearson_contrib,
                    )


if __name__ == "__main__":
    main()
