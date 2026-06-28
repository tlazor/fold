import numpy as np
import pytest
from MetricTransformer import (
    compute_overlaps,
    kl_divergence_matrix,
    mae_matrix,
    coherence_matrix,
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
        P = np.array([[0.5, 0.3, 0.2],
                      [0.1, 0.6, 0.3],
                      [0.4, 0.4, 0.2]])
        M = compute_overlaps(P)
        for i in range(3):
            np.testing.assert_allclose(M[i, i], P[i].sum(), rtol=1e-14)

    def test_identical_rows_give_max_overlap(self):
        P = np.array([[0.5, 0.3, 0.2],
                      [0.5, 0.3, 0.2]])
        M = compute_overlaps(P)
        np.testing.assert_allclose(M[0, 1], M[0, 0], rtol=1e-14)
        np.testing.assert_allclose(M[0, 1], M[1, 1], rtol=1e-14)

    def test_disjoint_distributions_zero_overlap(self):
        P = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
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
        P = np.array([[0.5, 0.3, 0.2],
                      [0.2, 0.5, 0.3]])
        M = kl_divergence_matrix(P)
        assert np.all(M >= -1e-12)

    def test_asymmetric(self):
        # KL(P||Q) ≠ KL(Q||P) for these distributions.
        # KL(P||Q) ≈ 1.40, KL(Q||P) ≈ 1.29 — verified analytically.
        P = np.array([[0.8, 0.1, 0.1],
                      [0.1, 0.2, 0.7]])
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
        P = np.array([[0.5, 0.5],
                      [0.3, 0.7]])
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
        M = coherence_matrix(self._signals(seed=3), fs=1.0, nperseg=10,
                             freq_band=(0.2, 0.8))
        assert M.shape == (3, 3)
        assert np.all(M >= -1e-12)
        assert np.all(M <= 1.0 + 1e-10)


@pytest.mark.skipif(not _CUPY_AVAILABLE, reason="3-D coherence branch requires CuPy / CUDA")
class TestCoherenceMatrix3D:
    def test_shape(self, freq_3d):
        import cupy as cp
        M = coherence_matrix(cp.asarray(freq_3d), fs=1.0, nperseg=4)
        assert M.shape == (3, 3)
