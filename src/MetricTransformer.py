from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.signal import coherence


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
    # print(f"{freq_spectra.shape=}")

    if len(freq_spectra.shape) == 2:
        # Broadcast the arrays so we can take the elementwise minimum of every pair (i, j)
        # Resulting shape: (num_langs, num_langs, freq_bins)
        pairwise_mins = np.minimum(freq_spectra[:, None, :], freq_spectra[None, :, :])

        # Sum along the frequency dimension (the last axis), resulting in (num_langs, num_langs)
        overlap_matrix = pairwise_mins.sum(axis=-1)
    elif len(freq_spectra.shape) == 3:
        # overlap(i,j)=embed_dim1​d=1∑embed_dim​(f=1∑num_freq_bins​min(freq_spectra[i,f,d],freq_spectra[j,f,d]))
        num_lang, num_freq, embed_dim = freq_spectra.shape
        overlap_matrix = np.zeros((num_lang, num_lang), dtype=float)

        for i in range(num_lang):
            for j in range(num_lang):
                # min_spec has shape (num_freq, embed_dim)
                min_spec = np.minimum(freq_spectra[i], freq_spectra[j])
                # sum over the frequency axis -> shape (embed_dim,)
                sum_over_freq = np.sum(min_spec, axis=0)
                # average across the embedding dimension -> scalar
                overlap_value = np.mean(sum_over_freq)
                overlap_matrix[i, j] = overlap_value

    return overlap_matrix


def kl_divergence_matrix(P, epsilon=1e-15):
    """
    Compute the KL divergence for all pairs of distributions
    in a 2D array P of shape (num_langs, num_samples),
    returning an array of shape (num_langs, num_langs).
    """
    if len(P.shape) == 2:
        # --------------------------------------------
        # 2D case: (num_langs, num_samples)
        # --------------------------------------------
        # 1) Normalize each row.
        P = P / P.sum(axis=1, keepdims=True)

        # 2) Compute log(P).
        LP = np.log(P + epsilon)

        # 3) alpha[i] = sum_x p[i, x] * log(p[i, x])
        alpha = np.sum(P * LP, axis=1)  # shape [L]

        # 4) beta[i, j] = sum_x p[i, x] * log(p[j, x])
        beta = P.dot(LP.T)  # shape [L, L]

        # 5) KL[i, j] = alpha[i] - beta[i, j]
        KL = alpha[:, np.newaxis] - beta  # shape [L, L]
        return KL

    elif len(P.shape) == 3:
        # --------------------------------------------
        # 3D case: (num_langs, num_tokens, num_samples)
        # --------------------------------------------
        # Suppose P[i, t, :] represents the distribution
        # for language i at token t across 'num_samples'.
        L, T, S = P.shape

        # 1) Normalize each distribution for each language & token
        P_norm = P / P.sum(axis=-1, keepdims=True)  # still shape (L, T, S)

        # 2) Compute log(P_norm)
        LP = np.log(P_norm + epsilon)  # shape (L, T, S)

        # 3) alpha[i, t] = sum_x p[i, t, x] * log(p[i, t, x])
        alpha = np.sum(P_norm * LP, axis=-1)  # shape (L, T)

        # 4) beta[i, j, t] = sum_x p[i, t, x] * log(p[j, t, x])
        # We'll use einsum to handle pairwise i, j at each token.
        # shape becomes (L, L, T)
        beta = np.einsum("lts,Lts->lLt", P_norm, LP)

        # 5) KL[i, j, t] = alpha[i, t] - beta[i, j, t]
        KL_per_token = alpha[:, None, :] - beta  # shape (L, L, T)

        # Option A: Take the mean over tokens
        KL = KL_per_token.mean(axis=-1)  # shape (L, L)

        # If you prefer summing over tokens, use .sum(axis=-1) instead.
        return KL


def mae_matrix(P):
    """
    Compute the mean absolute error for all pairs of distributions
    in a 2D array P of shape (num_langs, num_samples),
    returning an array of shape (num_langs, num_langs).
    """
    if len(P.shape) == 2:
        # 1) Normalize each row (if not already).
        P = P / P.sum(axis=1, keepdims=True)

        # 2) Compute pairwise differences using broadcasting:
        #    diff[i, j, :] = P[i, :] - P[j, :]
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]

        # 3) Take absolute value and average across the last axis.
        mae = np.mean(np.abs(diff), axis=2)

    elif len(P.shape) == 3:
        num_lang, num_tokens, embed_dim = P.shape
        # mae = np.zeros((num_lang, num_lang), dtype=float)

        # Compute norms: shape -> [num_lang, num_tokens]
        norms = np.linalg.norm(P, axis=-1)

        # Compute dot products (using Einstein summation for broadcasted pairwise dot):
        # dots[i, j, t] = dot( emb[i, t, :], emb[j, t, :] )
        dots = np.einsum("lte,Lte->lLt", P, P)  # shape [num_lang, num_lang, num_tokens]

        # Compute cosine similarities for each (i, j) language pair, at each token
        # cos[i, j, t] = dots[i, j, t] / (norms[i, t] * norms[j, t])
        # We can broadcast norms if we reshape them properly (adding the new axis).
        cos_sim = dots / (norms[:, None, :] * norms[None, :, :])

        # Finally, average over the token dimension
        # avg_cos_sim[i, j] = mean_t cos_sim[i, j, t]
        mae = cos_sim.mean(axis=-1)  # shape [num_lang, num_lang]

    return mae


import multiprocessing as mp
import numpy as np

def pairwise_coherence(args):
    i, j, P, hidden_dim = args
    s = np.array(0.0, dtype="float128")
    for k in range(hidden_dim):
        s += np.min(coherence(P[i, :, k], P[j, :, k])[1])
    return i, j, s / hidden_dim

def coherence_matrix_p(P):
    num_lang, num_freq, hidden_dim = P.shape
    overlap_matrix = np.zeros((num_lang, num_lang), dtype=float)

    # Prepare argument list for each (i, j) pair in the upper triangle (i <= j)
    # tasks = [(i, j, P, hidden_dim)
    tasks = [(i, j, P, 1) 
             for i in range(num_lang) 
             for j in range(i, num_lang)]

    with mp.Pool() as pool:
        results = pool.map(pairwise_coherence, tasks)

    # Fill overlap_matrix with results
    for i, j, val in results:
        overlap_matrix[i, j] = val
        overlap_matrix[j, i] = val

    return overlap_matrix

from rich.progress import track
def coherence_matrix(P, nperseg=10):
    """
    Compute the mean absolute error for all pairs of distributions
    in a 2D array P of shape (num_langs, num_samples),
    returning an array of shape (num_langs, num_langs).
    """
    if len(P.shape) == 2:
        num_lang, num_freq = P.shape
        overlap_matrix = np.zeros((num_lang, num_lang), dtype=float)

        for i in range(num_lang):
            for j in range(num_lang):
                overlap_matrix[i, j] = np.mean(coherence(P[i], P[j])[1])

    elif len(P.shape) == 3:
        overlap_matrix = coherence_matrix_p(P)

    return overlap_matrix


class MetricTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name, metric_fun, verbose=False):
        self.name = name
        self.metric_fun = metric_fun
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # samples x langs x langs
        return np.stack([self.metric_fun(x) for x in (track(X) if self.verbose else X)], axis=0)
