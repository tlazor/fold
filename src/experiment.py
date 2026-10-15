"""
The steps of a FOLD experiment.

1. sample_parallel_sentences: sentences that are translations of each other,
   tokenized for the model.
2. extract_signals: a signal per token for every sentence and language:
   how predictable the token is (likelihood) or its hidden state (embedding).
3. compare_languages: each signal turned into a spectrum, and every pair of
   languages compared sentence by sentence, giving one FOLD matrix per
   sentence for every frequency band and metric.

run_experiment chains them for one configuration; write_results writes the
correlations with the ground-truth measures as the run_pipeline.py text file.
set_verbose(False) hides step logs and model-loading messages (for notebooks).
"""

import contextlib
import io

import numpy as np
import torch
import transformers
from rich.progress import track
from sklearn.pipeline import Pipeline

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
from analysis import analyze_output, get_langs
from paths import DATA_DIR, auto_device

# Step logs and model-loading messages; progress bars are always shown.
_verbose = True

# Metrics computed on the (band-selected) spectra, by output name.
_SPECTRAL_METRICS = {
    "compute_overlaps": compute_overlaps,
    "kl_divergence_matrix": kl_divergence_matrix,
}
# Coherence is computed on the raw signal instead.
_COHERENCE = "coherence_fun"
_COHERENCE_NPERSEG = 10


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


def set_verbose(verbose):
    """Show (the default) or hide step logs and model-loading messages."""
    global _verbose
    _verbose = verbose
    if not verbose:
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()


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


def languages(config):
    """The languages: those with text in the corpus that the model was trained on."""
    quiet = contextlib.redirect_stdout(io.StringIO())
    with contextlib.nullcontext() if _verbose else quiet:
        return get_langs(config.dataset, config.use_bert)


def sample_parallel_sentences(config, langs):
    """
    600 sentences that are translations of each other, tokenized for the model.

    Returns one list per sentence holding an (input_ids, attention_mask) pair
    per language, in the order of `langs`.
    """
    return Pipeline(
        _build_data_components(config, langs), verbose=_verbose
    ).fit_transform(None)


def extract_signals(sentences, config, layer=None, device=None):
    """
    A signal per token for every sentence and language.

    Likelihood mode: how predictable each token is when masked, as arrays of
    shape (n_langs, n_tokens). Embedding mode: each token's hidden state at
    `layer`, shape (n_langs, n_tokens, hidden_dim). Cached per sentence.
    """
    device = device if device is not None else auto_device()
    return _signal_transformer(config, layer, device).fit_transform(sentences)


def compare_languages(signals, config):
    """
    Compare every pair of languages, sentence by sentence.

    Each signal becomes a spectrum (per config.spectral_mode); overlap and KL
    divergence compare the spectra within each frequency band, and coherence
    compares the raw signals. Returns {(band index, metric name): FOLD
    matrices of shape (n_sentences, n_langs, n_langs)}.
    """
    spectra = Pipeline(
        _build_spectra_components(config), verbose=_verbose
    ).fit_transform(signals)
    fold = {}
    for k, band in enumerate(config.freq_bands):
        banded = BandSelectTransformer(freq_band=band).fit_transform(spectra)
        for name, fun in _SPECTRAL_METRICS.items():
            metric = MetricTransformer(name=name, metric_fun=fun)
            fold[(k, name)] = metric.fit_transform(banded)

    if config.spectral_mode != "none":
        # Coherence spectra are computed once and averaged over each band.
        per_sample = []
        for x in track(signals, description="coherence"):
            freqs, coh = coherence_spectra(x, nperseg=_COHERENCE_NPERSEG)
            per_sample.append(coherence_band_means(freqs, coh, config.freq_bands))
        for k in range(len(config.freq_bands)):
            fold[(k, _COHERENCE)] = np.stack([means[k] for means in per_sample])
    return fold


def run_experiment(config, device=None):
    """
    Run every band, metric and layer of one configuration.

    Returns (langs, fold): fold[(band index, metric name, layer)] is the
    (n_sentences, n_langs, n_langs) stack of FOLD matrices.
    """
    langs = languages(config)
    sentences = sample_parallel_sentences(config, langs)

    fold = {}
    for layer in experiment_layers(config):
        signals = extract_signals(sentences, config, layer, device)
        for (k, name), matrices in compare_languages(signals, config).items():
            fold[(k, name, layer)] = matrices
        del signals  # one layer's signals in memory at a time
    return langs, fold


def write_results(config, langs, fold, f):
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
                    fold[(k, name, layer)],
                    langs,
                    f=f,
                    model_name=config.model_name,
                    flag_analyze_pearson_contrib=config.analyze_pearson_contrib,
                )
