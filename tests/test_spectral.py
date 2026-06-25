import numpy as np
import pytest
from SpectralTransformer import SpectralTransformer, circular_optimized
from PsdEstimator import PsdEstimator
from PsdNormalizer import PsdNormalizer
from BandSelectTransformer import BandSelectTransformer


# ===========================================================================
# SpectralTransformer / circular_optimized
# ===========================================================================

class TestSpectralTransformer:
    def test_circular_optimized_shape(self):
        arr = np.random.default_rng(0).random((3, 16))
        _, ps = circular_optimized(arr)
        assert ps.shape == (3, 8)

    def test_transform_returns_list(self, signal_list_2d):
        result = SpectralTransformer().transform(signal_list_2d)
        assert isinstance(result, list)

    def test_transform_list_length(self, signal_list_2d):
        result = SpectralTransformer().transform(signal_list_2d)
        assert len(result) == len(signal_list_2d)

    def test_transform_element_shapes(self, signal_list_2d):
        result = SpectralTransformer().transform(signal_list_2d)
        for arr in result:
            assert arr.shape == (3, 8)

    def test_power_spectrum_nonneg(self, signal_list_2d):
        result = SpectralTransformer().transform(signal_list_2d)
        for arr in result:
            assert np.all(arr >= 0)

    def test_fit_returns_self(self):
        t = SpectralTransformer()
        assert t.fit(None) is t


# ===========================================================================
# PsdEstimator
# ===========================================================================

class TestPsdEstimator:
    def test_output_is_ndarray(self, signal_list_2d):
        result = PsdEstimator(nperseg=8).transform(signal_list_2d)
        assert isinstance(result, np.ndarray)

    def test_output_shape(self, signal_list_2d):
        # nperseg=8 → F = 8//2 + 1 = 5; 4 samples, 3 langs
        result = PsdEstimator(nperseg=8).transform(signal_list_2d)
        assert result.shape == (4, 3, 5)

    def test_output_nonneg(self, signal_list_2d):
        result = PsdEstimator(nperseg=8).transform(signal_list_2d)
        assert np.all(result >= 0)

    def test_consistent_freq_dim_across_samples(self):
        rng = np.random.default_rng(5)
        X = [rng.random((2, 16)) for _ in range(3)]
        result = PsdEstimator(nperseg=8).transform(X)
        assert result.ndim == 3
        shapes = {result[i].shape for i in range(len(result))}
        assert len(shapes) == 1

    def test_fit_returns_self(self):
        assert PsdEstimator().fit(None) is PsdEstimator().fit(None) or True
        t = PsdEstimator()
        assert t.fit(None) is t


# ===========================================================================
# PsdNormalizer
# ===========================================================================

class TestPsdNormalizer:
    def test_output_is_list(self, psd_list_2d):
        result = PsdNormalizer().transform(psd_list_2d)
        assert isinstance(result, list)

    def test_output_list_length(self, psd_list_2d):
        result = PsdNormalizer().transform(psd_list_2d)
        assert len(result) == len(psd_list_2d)

    def test_rows_sum_to_one(self, psd_list_2d):
        result = PsdNormalizer().transform(psd_list_2d)
        for arr in result:
            np.testing.assert_allclose(arr.sum(axis=-1), 1.0, atol=1e-6)

    def test_shape_preserved(self, psd_list_2d):
        result = PsdNormalizer().transform(psd_list_2d)
        for original, normalized in zip(psd_list_2d, result):
            assert original.shape == normalized.shape

    def test_nonneg_inputs_stay_nonneg(self, psd_list_2d):
        result = PsdNormalizer().transform(psd_list_2d)
        for arr in result:
            assert np.all(arr >= 0)

    def test_fit_returns_self(self):
        t = PsdNormalizer()
        assert t.fit(None) is t


# ===========================================================================
# BandSelectTransformer
# ===========================================================================

class TestBandSelectTransformer:
    def test_full_range_2d_shape(self, psd_list_2d):
        result = BandSelectTransformer(freq_band=(0, 1)).transform(psd_list_2d)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 3, 8)

    def test_half_range_2d_shape(self, psd_list_2d):
        result = BandSelectTransformer(freq_band=(0.5, 1.0)).transform(psd_list_2d)
        assert result.shape == (4, 3, 4)

    def test_values_match_slice_2d(self, psd_list_2d):
        result = BandSelectTransformer(freq_band=(0.5, 1.0)).transform(psd_list_2d)
        for i, x in enumerate(psd_list_2d):
            np.testing.assert_array_equal(result[i], x[:, 4:8])

    def test_full_range_values_unchanged_2d(self, psd_list_2d):
        result = BandSelectTransformer(freq_band=(0, 1)).transform(psd_list_2d)
        for i, x in enumerate(psd_list_2d):
            np.testing.assert_array_equal(result[i], x)

    def test_half_range_3d_shape(self, psd_list_3d):
        # Each input (3, 4, 8); freq_band=(0.5,1.0) → last dim 4:8 → stacked (4, 3, 4, 4)
        result = BandSelectTransformer(freq_band=(0.5, 1.0)).transform(psd_list_3d)
        assert result.shape == (4, 3, 4, 4)

    def test_values_match_slice_3d(self, psd_list_3d):
        result = BandSelectTransformer(freq_band=(0.5, 1.0)).transform(psd_list_3d)
        for i, x in enumerate(psd_list_3d):
            np.testing.assert_array_equal(result[i], x[:, :, 4:8])

    def test_fit_returns_self(self):
        t = BandSelectTransformer(freq_band=(0, 1))
        assert t.fit(None) is t
