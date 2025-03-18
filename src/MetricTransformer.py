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
    # for temp in np.sum(freq_spectra, axis=-1):
    #     if not np.isclose(temp, 1):
    #         print(f"{temp=}")

    return overlap_matrix


def kl_divergence_matrix(P, epsilon=1e-15):
    """
    Compute the KL divergence for all pairs of distributions
    in a 2D array P of shape (num_langs, num_samples),
    returning an array of shape (num_langs, num_langs).
    """
    # 1) Normalize each row (if not already).
    P = P / P.sum(axis=1, keepdims=True)

    # 2) Compute log(P). Add a small epsilon to avoid log(0).
    LP = np.log(P + epsilon)

    # 3) alpha[i] = sum_x p[i, x] * log(p[i, x])
    alpha = np.sum(P * LP, axis=1)

    # 4) beta[i, j] = sum_x p[i, x] * log(p[j, x])
    beta = P.dot(LP.T)  # shape (L, L)

    # 5) KL[i, j] = alpha[i] - beta[i, j]
    KL = alpha[:, np.newaxis] - beta  # shape (L, L)
    return KL


def mae_matrix(P):
    """
    Compute the mean absolute error for all pairs of distributions
    in a 2D array P of shape (num_langs, num_samples),
    returning an array of shape (num_langs, num_langs).
    """
    # 1) Normalize each row (if not already).
    P = P / P.sum(axis=1, keepdims=True)

    # 2) Compute pairwise differences using broadcasting:
    #    diff[i, j, :] = P[i, :] - P[j, :]
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]

    # 3) Take absolute value and average across the last axis.
    mae = np.mean(np.abs(diff), axis=2)

    return mae


class MetricTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name, metric_fun):
        self.name = name
        self.metric_fun = metric_fun

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # samples x langs x langs
        return np.stack([self.metric_fun(x) for x in X], axis=0)
