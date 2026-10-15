"""
FOLD unified pipeline entry point.

Replaces the separate pipeline.py / embed_pipeline.py scripts.

Usage:
    uv run fold-pipeline --signal-mode likelihood [OPTIONS]
    uv run fold-pipeline --signal-mode embedding  [OPTIONS]
    uv run python src/run_pipeline.py --help
"""

from pathlib import Path

import numpy as np
from rich.progress import track
from sklearn.pipeline import Pipeline
import torch

from BandSelectTransformer import BandSelectTransformer
from BibleTransformer import BibleTransformer
from EmbedTransformer import EmbedTransformer
from LikelihoodEstimator import LikelihoodEstimator
from MetricTransformer import (
    MetricTransformer,
    coherence_band_means,
    coherence_spectra,
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
        loader = (
            "load_un6",
            Un6Transformer(DATA_DIR / "6way", langs=langs, nrows=2000),
        )
    else:
        loader = (
            "load_tsv",
            TsvToDataFrame(
                DATA_DIR / "XNLI-15way" / "xnli.15way.orig.tsv", langs=langs
            ),
        )
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


# Metrics computed on the (band-selected) spectra, by output name.
_SPECTRAL_METRICS = {
    "compute_overlaps": compute_overlaps,
    "kl_divergence_matrix": kl_divergence_matrix,
}
# Coherence is computed on the raw signal instead.
_COHERENCE = "coherence_fun"
_COHERENCE_NPERSEG = 10


def _signal_transformer(config, layer, device):
    if config.signal_mode == "embedding":
        # All layers come from one forward pass, cached together.
        return EmbedTransformer(
            layer=layer,
            layers=tuple(config.layers),
            model_name=config.model_name,
            device=device,
        )
    return LikelihoodEstimator(
        model_name=config.model_name,
        mask_token_id=config.mask_token_id,
        device=device,
    )


def _metric_outputs(config, signals, spectra_components):
    """
    Every metric for one signal (one layer), keyed by (band index, metric name).

    Spectra are computed once and band-selected per band; coherence spectra are
    computed once and averaged per band. Each value is (n_samples, n_langs, n_langs).
    """
    spectra = Pipeline(spectra_components, verbose=True).fit_transform(signals)
    outputs = {}
    for k, band in enumerate(config.freq_bands):
        banded = BandSelectTransformer(freq_band=band).fit_transform(spectra)
        for name, fun in _SPECTRAL_METRICS.items():
            metric = MetricTransformer(name=name, metric_fun=fun)
            outputs[(k, name)] = metric.fit_transform(banded)

    if config.spectral_mode != "none":
        per_sample = []
        for x in track(signals, description="coherence"):
            freqs, coh = coherence_spectra(x, nperseg=_COHERENCE_NPERSEG)
            per_sample.append(coherence_band_means(freqs, coh, config.freq_bands))
        for k in range(len(config.freq_bands)):
            outputs[(k, _COHERENCE)] = np.stack([means[k] for means in per_sample])
    return outputs


def configure_torch():
    """GPU settings for a run: cap this process at 80% of VRAM, TF32 matmuls."""
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True


def experiment_layers(config):
    """Layers a run iterates over: None (one pass) in likelihood mode."""
    return config.layers if config.signal_mode == "embedding" else [None]


def metric_names(config):
    """Metric names a run produces, in output order."""
    names = list(_SPECTRAL_METRICS)
    if config.spectral_mode != "none":
        names.append(_COHERENCE)
    return names


def run_experiment(config, device=None):
    """
    Run every band, metric and layer of one configuration.

    Returns (langs, outputs): outputs[(band index, metric name, layer)] is the
    (n_samples, n_langs, n_langs) stack of per-sample metric matrices.
    """
    device = device if device is not None else auto_device()
    langs = get_langs(config.dataset, config.use_bert)
    spectra_components = _build_spectra_components(config)

    # Build the samples once. Signal extraction is the only expensive step and
    # caches per sample (.cache/joblib), so nothing else is cached.
    samples = Pipeline(
        _build_data_components(config, langs), verbose=True
    ).fit_transform(None)

    outputs = {}
    for layer in experiment_layers(config):
        signals = _signal_transformer(config, layer, device).fit_transform(samples)
        for (k, name), output in _metric_outputs(
            config, signals, spectra_components
        ).items():
            outputs[(k, name, layer)] = output
        del signals  # one layer's signals in memory at a time
    return langs, outputs


def write_results(config, langs, outputs, f):
    """Write the config and every correlation table, band -> metric -> layer."""
    config.print_config(file=f)

    for k, band in enumerate(config.freq_bands):
        print(f"{band=}", file=f)
        for name in metric_names(config):
            print(name, file=f)
            for layer in experiment_layers(config):
                if layer is not None:
                    print(f"{layer=}", file=f)
                analyze_output(
                    outputs[(k, name, layer)],
                    langs,
                    f=f,
                    model_name=config.model_name,
                    flag_analyze_pearson_contrib=config.analyze_pearson_contrib,
                )


def main():
    configure_torch()

    device = auto_device()
    print("Using device:", device)

    config = PipelineOptions.from_args()
    config.print_config()

    langs, outputs = run_experiment(config, device)

    output_path = Path(config.get_output_filename())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w+", encoding="utf-8") as f:
        write_results(config, langs, outputs, f)


if __name__ == "__main__":
    main()
