import numpy as np
import pytest
from scipy.signal import coherence
from MetricTransformer import (
    compute_overlaps,
    kl_divergence_matrix,
    mae_matrix,
    coherence_band_means,
    coherence_matrix,
    coherence_spectra,
)

try:
    import cupy as _cupy
    import cupyx.scipy.signal  # noqa: F401

    _cupy.array([1.0])  # trigger actual CUDA initialisation
    _CUPY_AVAILABLE = True
except Exception:
    _CUPY_AVAILABLE = False


# ===========================================================================
# compute_overlaps
# ===========================================================================


class TestComputeOverlaps2D:
    def test_shape(self, freq_2d):
        assert compute_overlaps(freq_2d).shape == (3, 3)

    def test_symmetric(self, freq_2d):
        M = compute_overlaps(freq_2d)
        np.testing.assert_array_equal(M, M.T)

    def test_nonneg(self, freq_2d):
        assert np.all(compute_overlaps(freq_2d) >= 0)

    def test_diagonal_equals_row_sum(self):
        P = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.4, 0.4, 0.2]])
        M = compute_overlaps(P)
        for i in range(3):
            np.testing.assert_allclose(M[i, i], P[i].sum(), rtol=1e-14)

    def test_identical_rows_give_max_overlap(self):
        P = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
        M = compute_overlaps(P)
        np.testing.assert_allclose(M[0, 1], M[0, 0], rtol=1e-14)
        np.testing.assert_allclose(M[0, 1], M[1, 1], rtol=1e-14)

    def test_disjoint_distributions_zero_overlap(self):
        P = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        M = compute_overlaps(P)
        assert M[0, 1] == 0.0


class TestComputeOverlaps3D:
    def test_shape(self, freq_3d):
        assert compute_overlaps(freq_3d).shape == (3, 3)

    def test_symmetric(self, freq_3d):
        M = compute_overlaps(freq_3d)
        np.testing.assert_allclose(M, M.T, rtol=1e-12)

    def test_nonneg(self, freq_3d):
        assert np.all(compute_overlaps(freq_3d) >= 0)

    def test_diagonal_formula(self):
        rng = np.random.default_rng(0)
        P = np.abs(rng.random((2, 4, 3))) + 0.1
        M = compute_overlaps(P)
        for i in range(2):
            expected = np.mean(np.sum(P[i], axis=0))
            np.testing.assert_allclose(M[i, i], expected, rtol=1e-12)


# ===========================================================================
# kl_divergence_matrix
# ===========================================================================


class TestKLDivergence2D:
    def test_shape(self, freq_2d):
        assert kl_divergence_matrix(freq_2d).shape == (3, 3)

    def test_diagonal_near_zero(self, freq_2d):
        M = kl_divergence_matrix(freq_2d)
        np.testing.assert_allclose(np.diag(M), 0.0, atol=1e-10)

    def test_nonneg(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
        M = kl_divergence_matrix(P)
        assert np.all(M >= -1e-12)

    def test_asymmetric(self):
        # KL(P||Q) ≠ KL(Q||P) for these distributions.
        # KL(P||Q) ≈ 1.40, KL(Q||P) ≈ 1.29 — verified analytically.
        P = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
        M = kl_divergence_matrix(P)
        assert abs(M[0, 1] - M[1, 0]) > 1e-4


class TestKLDivergence3D:
    def test_shape(self, freq_3d):
        assert kl_divergence_matrix(freq_3d).shape == (3, 3)

    def test_diagonal_near_zero(self, freq_3d):
        M = kl_divergence_matrix(freq_3d)
        np.testing.assert_allclose(np.diag(M), 0.0, atol=1e-10)


# ===========================================================================
# mae_matrix
# ===========================================================================


class TestMAEMatrix2D:
    def test_shape(self, freq_2d):
        assert mae_matrix(freq_2d).shape == (3, 3)

    def test_diagonal_exactly_zero(self, freq_2d):
        M = mae_matrix(freq_2d)
        np.testing.assert_array_equal(np.diag(M), 0.0)

    def test_symmetric(self, freq_2d):
        M = mae_matrix(freq_2d)
        np.testing.assert_array_equal(M, M.T)

    def test_nonneg(self, freq_2d):
        assert np.all(mae_matrix(freq_2d) >= 0)

    def test_known_value(self):
        # Both rows already sum to 1 so normalization is a no-op.
        # MAE[0,1] = mean(|0.5-0.3|, |0.5-0.7|) = mean(0.2, 0.2) = 0.2
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        M = mae_matrix(P)
        np.testing.assert_allclose(M[0, 1], 0.2, rtol=1e-10)


class TestMAEMatrix3D:
    # The 3-D branch computes cosine similarity, not MAE (mislabeled in source).
    def test_shape(self, freq_3d):
        assert mae_matrix(freq_3d).shape == (3, 3)

    def test_diagonal_near_one(self, freq_3d):
        M = mae_matrix(freq_3d)
        np.testing.assert_allclose(np.diag(M), 1.0, atol=1e-12)

    def test_symmetric(self, freq_3d):
        M = mae_matrix(freq_3d)
        np.testing.assert_allclose(M, M.T, atol=1e-12)


# ===========================================================================
# coherence_matrix (2-D only in CPU tests)
# ===========================================================================


class TestCoherenceMatrix2D:
    def _signals(self, seed=0):
        return np.random.default_rng(seed).random((3, 20))

    def test_shape(self):
        M = coherence_matrix(self._signals(), fs=1.0, nperseg=10)
        assert M.shape == (3, 3)

    def test_diagonal_is_one(self):
        M = coherence_matrix(self._signals(), fs=1.0, nperseg=10)
        np.testing.assert_allclose(np.diag(M), 1.0, atol=1e-10)

    def test_values_in_unit_interval(self):
        M = coherence_matrix(self._signals(seed=1), fs=1.0, nperseg=10)
        assert np.all(M >= -1e-12)
        assert np.all(M <= 1.0 + 1e-10)

    def test_symmetric(self):
        M = coherence_matrix(self._signals(seed=2), fs=1.0, nperseg=10)
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_freq_band_filter(self):
        M = coherence_matrix(
            self._signals(seed=3), fs=1.0, nperseg=10, freq_band=(0.2, 0.8)
        )
        assert M.shape == (3, 3)
        assert np.all(M >= -1e-12)
        assert np.all(M <= 1.0 + 1e-10)


def _scipy_coherence_matrix(P, nperseg, freq_band=None):
    """Reference: scipy coherence for every pair, along the token axis."""
    L = P.shape[0]
    M = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            f, Cxy = coherence(P[i], P[j], nperseg=nperseg, axis=0)
            if freq_band is not None:
                Cxy = Cxy[(f >= freq_band[0] / 2) & (f <= freq_band[1] / 2)]
            M[i, j] = Cxy.mean()
    return M


class TestCoherenceMatchesScipy:
    @pytest.mark.parametrize("nperseg", [10, 9, None])
    @pytest.mark.parametrize("freq_band", [None, (0.2, 0.8)])
    def test_2d(self, nperseg, freq_band):
        P = np.random.default_rng(0).random((4, 30))
        M = coherence_matrix(P, nperseg=nperseg, freq_band=freq_band)
        expected = _scipy_coherence_matrix(P, nperseg, freq_band)
        np.testing.assert_allclose(M, expected, rtol=1e-10)

    @pytest.mark.parametrize("nperseg", [8, 7])
    def test_3d_numpy(self, nperseg):
        P = np.random.default_rng(1).random((3, 24, 5))  # (langs, tokens, hidden)
        _, coh = coherence_spectra(P, nperseg=nperseg, xp=np)
        np.testing.assert_allclose(
            coh.mean(axis=-1), _scipy_coherence_matrix(P, nperseg), rtol=1e-10
        )

    def test_band_means_match_per_band_calls(self):
        P = np.random.default_rng(2).random((3, 40))
        bands = [(0.0, 0.5), (0.5, 1.0), None]
        freqs, coh = coherence_spectra(P, nperseg=10)
        for band, M in zip(bands, coherence_band_means(freqs, coh, bands)):
            np.testing.assert_allclose(
                M, coherence_matrix(P, nperseg=10, freq_band=band), rtol=1e-12
            )


@pytest.mark.skipif(
    not _CUPY_AVAILABLE, reason="3-D coherence branch requires CuPy / CUDA"
)
class TestCoherenceMatrix3D:
    def test_shape(self, freq_3d):
        import cupy as cp

        M = coherence_matrix(cp.asarray(freq_3d), fs=1.0, nperseg=4)
        assert M.shape == (3, 3)

    def test_matches_scipy_along_tokens(self, freq_3d):
        import cupy as cp
        from scipy.signal import coherence

        M = coherence_matrix(cp.asarray(freq_3d), fs=1.0, nperseg=4)
        for i in range(3):
            for j in range(3):
                _, Cxy = coherence(freq_3d[i], freq_3d[j], nperseg=4, axis=0)
                np.testing.assert_allclose(M[i, j], Cxy.mean(), rtol=1e-6)

    def test_scaled_hidden_dims_fully_coherent(self):
        import cupy as cp

        # Scaling each hidden dimension's token signal by a constant leaves
        # coherence along the tokens at exactly 1; along the hidden dimensions
        # (the old, wrong axis) it would not be.
        rng = np.random.default_rng(0)
        base = rng.random((16, 32))  # (tokens, hidden_dim); old axis gives ~0.74
        P = np.stack([base, base * rng.uniform(0.5, 2.0, size=32)])
        M = coherence_matrix(cp.asarray(P), fs=1.0, nperseg=8)
        np.testing.assert_allclose(M[0, 1], 1.0, atol=1e-6)
