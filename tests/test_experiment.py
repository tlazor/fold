from functools import partial

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from BandSelectTransformer import BandSelectTransformer
from MetricTransformer import (
    MetricTransformer,
    coherence_matrix,
    compute_overlaps,
    kl_divergence_matrix,
)
from pipeline_options import PipelineOptions
from experiment import _build_spectra_components, compare_languages


@pytest.mark.parametrize("spectral_mode", ["welch", "fft", "none"])
@pytest.mark.parametrize("signal_mode", ["likelihood", "embedding"])
def test_compare_languages_matches_one_pipeline_per_metric(signal_mode, spectral_mode):
    # The pipeline used to be rebuilt for every (band, metric): spectra, band
    # selection and metric for overlaps/KL, coherence_matrix on the raw signal.
    config = PipelineOptions(
        signal_mode=signal_mode, spectral_mode=spectral_mode, num_bands=2
    )
    rng = np.random.default_rng(0)
    shape = (4, 64) if signal_mode == "likelihood" else (4, 64, 8)
    signals = [rng.random(shape) for _ in range(5)]

    outputs = compare_languages(signals, config)

    for k, band in enumerate(config.freq_bands):
        for fun in (compute_overlaps, kl_divergence_matrix):
            per_metric = Pipeline(
                _build_spectra_components(config)
                + [
                    ("band", BandSelectTransformer(freq_band=band)),
                    ("metric", MetricTransformer(fun.__name__, fun)),
                ]
            ).fit_transform(signals)
            np.testing.assert_allclose(outputs[(k, fun.__name__)], per_metric)

        if spectral_mode == "none":
            assert (k, "coherence_fun") not in outputs
        else:
            coherence = partial(coherence_matrix, nperseg=10, freq_band=band)
            np.testing.assert_allclose(
                outputs[(k, "coherence_fun")],
                np.stack([coherence(x) for x in signals]),
                rtol=1e-10,
            )
