from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def compute_overlaps(freq_spectra):
    """
    Computes the overlap between every pair of languages in freq_spectra.

    Parameters
    ----------
    freq_spectra : np.ndarray of shape (num_langs, freq_bins)
        Each row i represents the frequency distribution for language i.

    Returns
    -------
    overlap_matrix : np.ndarray of shape (num_langs, num_langs)
        overlap_matrix[i, j] = sum_k min(freq_spectra[i, k], freq_spectra[j, k])
    """
    # Broadcast the arrays so we can take the elementwise minimum of every pair (i, j)
    # Resulting shape: (num_langs, num_langs, freq_bins)
    pairwise_mins = np.minimum(freq_spectra[:, None, :], freq_spectra[None, :, :])

    # Sum along the frequency dimension (the last axis), resulting in (num_langs, num_langs)
    overlap_matrix = pairwise_mins.sum(axis=-1)

    # double check normalization
    for temp in np.sum(freq_spectra, axis=-1):
        if not np.isclose(temp, 1):
            print(f"{temp=}")

    return overlap_matrix


class OverlapTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, temp="fourier"):
        self.temp = temp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # samples x langs x langs
        return np.stack([compute_overlaps(x) for x in X], axis=0)
