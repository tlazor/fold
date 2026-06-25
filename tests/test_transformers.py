import numpy as np
import pytest
from SampleTokens import SampleTokens
from NoOpTransformer import NoOpTransformer


def _make_sample(n_langs: int, n_tokens: int, n_valid: list):
    """Build one sample: list of (ids, mask) tuples, one per lang."""
    sample = []
    for v in n_valid:
        ids = np.ones(n_tokens, dtype=np.int64)
        mask = np.array([1] * v + [0] * (n_tokens - v), dtype=np.int64)
        sample.append((ids, mask))
    return sample


# ===========================================================================
# SampleTokens
# ===========================================================================

class TestSampleTokens:
    N_LANGS = 2
    N_TOKENS = 20

    def _valid(self):
        return _make_sample(self.N_LANGS, self.N_TOKENS, [15, 12])

    def _invalid(self):
        # lang 0 has 5 non-pad tokens → filtered when minimum_tokens=10
        return _make_sample(self.N_LANGS, self.N_TOKENS, [5, 12])

    def test_output_length(self):
        X = [self._valid() for _ in range(50)]
        result = SampleTokens(num_samples=10, minimum_tokens=5, seed=0).transform(X)
        assert len(result) == 10

    def test_filters_short_samples(self):
        X = [self._valid() for _ in range(30)] + [self._invalid() for _ in range(20)]
        result = SampleTokens(num_samples=10, minimum_tokens=10, seed=0).transform(X)
        for sample in result:
            for ids, mask in sample:
                assert mask.sum() >= 10

    def test_deterministic_with_seed(self):
        X = [self._valid() for _ in range(50)]
        a = SampleTokens(num_samples=5, minimum_tokens=5, seed=42).transform(X)
        b = SampleTokens(num_samples=5, minimum_tokens=5, seed=42).transform(X)
        for sa, sb in zip(a, b):
            for (ia, ma), (ib, mb) in zip(sa, sb):
                np.testing.assert_array_equal(ia, ib)
                np.testing.assert_array_equal(ma, mb)

    def test_all_langs_present_in_output(self):
        X = [_make_sample(3, 20, [15, 12, 10]) for _ in range(30)]
        result = SampleTokens(num_samples=5, minimum_tokens=5, seed=0).transform(X)
        for sample in result:
            assert len(sample) == 3

    def test_fit_returns_self(self):
        t = SampleTokens()
        assert t.fit(None) is t


# ===========================================================================
# NoOpTransformer
# (zero-pads arrays to uniform length along axis 1, not a true identity)
# ===========================================================================

class TestNoOpTransformer:
    def test_same_length_output_equals_input(self):
        X = [
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        ]
        result = NoOpTransformer().transform(X)
        np.testing.assert_array_equal(result[0], X[0])
        np.testing.assert_array_equal(result[1], X[1])

    def test_shorter_array_padded_with_zeros(self):
        X = [
            np.array([[1.0, 2.0, 3.0]]),
            np.array([[4.0, 5.0]]),
        ]
        result = NoOpTransformer().transform(X)
        assert result[1].shape == (1, 3)
        np.testing.assert_array_equal(result[1], [[4.0, 5.0, 0.0]])

    def test_all_outputs_have_same_shape(self):
        rng = np.random.default_rng(9)
        X = [rng.random((3, 10)), rng.random((3, 8)), rng.random((3, 12))]
        result = NoOpTransformer().transform(X)
        shapes = {arr.shape for arr in result}
        assert len(shapes) == 1
        assert result[0].shape == (3, 12)

    def test_max_length_array_unchanged(self):
        rng = np.random.default_rng(0)
        long_arr = rng.random((3, 12))
        X = [rng.random((3, 8)), long_arr.copy()]
        result = NoOpTransformer().transform(X)
        np.testing.assert_array_equal(result[1], long_arr)

    def test_fit_returns_self(self):
        t = NoOpTransformer()
        assert t.fit(None) is t
