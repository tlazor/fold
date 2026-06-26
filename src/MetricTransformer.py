from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from joblib import Parallel, delayed
from rich.progress import track


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
        # overlap(i,j) = mean_d( sum_f min(s[i,f,d], s[j,f,d]) )
        # Broadcast to (L, L, num_freq, embed_dim), take elem-wise min, sum over freq, mean over dim.
        pairwise_mins = np.minimum(
            freq_spectra[:, np.newaxis, :, :],  # (L, 1, F, D)
            freq_spectra[np.newaxis, :, :, :],  # (1, L, F, D)
        )  # → (L, L, F, D)
        overlap_matrix = pairwise_mins.sum(axis=2).mean(axis=2)  # (L, L)

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
        # 3-D case: (num_langs, num_tokens, hidden_dim)
        # --------------------------------------------
        L, T, S = P.shape

        # 1) Soft-max   — convert logits -> probabilities in a stable way
        #    subtract max over last axis for numerical stability
        P_shift = P - P.max(axis=-1, keepdims=True)
        expP    = np.exp(P_shift)
        Z       = expP.sum(axis=-1, keepdims=True)           # partition function
        P_norm  = expP / (Z + epsilon)                       # shape (L, T, S)

        # 2) log(P_norm)
        LP = np.log(P_norm + epsilon)                        # shape (L, T, S)

        # 3) α[i, t] = Σ_x  p[i,t,x] log p[i,t,x]
        alpha = np.sum(P_norm * LP, axis=-1)                 # shape (L, T)

        # 4) β[i, j, t] = Σ_x  p[i,t,x] log p[j,t,x]
        #    Pair-wise across languages with einsum
        beta = np.einsum("lts,Lts->lLt", P_norm, LP)         # shape (L, L, T)

        # 5) KL[i, j, t] = α[i,t] − β[i,j,t]
        KL_per_token = alpha[:, None, :] - beta              # shape (L, L, T)

        # Option A: mean over tokens (default)
        KL = KL_per_token.mean(axis=-1)                      # shape (L, L)
        # Option B: sum over tokens → use .sum(axis=-1)

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


def coherence_matrix(P, fs=1.0, nperseg=None, freq_band=None):
    """
    Compute the coherence between all pairs of distributions
    in a 2D array P of shape (num_langs, num_samples),
    returning an array of shape (num_langs, num_langs).

    Parameters
    ----------
    P : np.ndarray
        Input array of shape (num_langs, num_samples) or (num_langs, num_tokens, num_samples)
    fs : float, optional
        Sampling frequency of the signals. Default is 1.0.
    nperseg : int, optional
        Length of each segment for coherence calculation. Default is None (uses scipy's default).
    freq_band : tuple, optional
        Frequency band to filter coherence calculation (min_freq, max_freq) as a proportion of Nyquist frequency.
        For example, (0.4, 0.5) will only use frequencies between 40% and 50% of the Nyquist frequency.
        Default is None (uses all frequencies).

    Returns
    -------
    coherence_matrix : np.ndarray
        Matrix of shape (num_langs, num_langs) containing coherence values between all pairs
    """
    if len(P.shape) == 2:
        from scipy.signal import coherence

        num_langs = P.shape[0]
        coherence_mat = np.zeros((num_langs, num_langs))

        for i in range(num_langs):
            for j in range(num_langs):
                # Calculate coherence between signals using cached function
                f, Cxy = coherence(P[i], P[j], fs=fs, nperseg=nperseg)

                # Filter frequency band if specified
                if freq_band is not None:
                    nyquist = fs / 2
                    min_freq = freq_band[0] * nyquist
                    max_freq = freq_band[1] * nyquist
                    mask = (f >= min_freq) & (f <= max_freq)
                    Cxy = Cxy[mask]

                # Take mean of coherence across selected frequencies
                coherence_mat[i, j] = np.mean(Cxy)

        return coherence_mat

    elif len(P.shape) == 3:
        import cupy as cp
        from cupyx.scipy.signal import coherence

        num_langs, num_tokens, _ = P.shape
        coherence_mat = cp.zeros((num_langs, num_langs))

        # Process language pairs in parallel where possible
        for i in range(num_langs):
            # Get all signals for language i
            sigs_i = P[i]  # shape: (num_tokens, num_samples)

            for j in range(num_langs):
                # Get all signals for language j
                sigs_j = P[j]  # shape: (num_tokens, num_samples)

                # Calculate coherence for all token positions at once using cached function
                f, Cxy = coherence(sigs_i, sigs_j, fs=fs, nperseg=nperseg)

                # Filter frequency band if specified
                if freq_band is not None:
                    nyquist = fs / 2
                    min_freq = freq_band[0] * nyquist
                    max_freq = freq_band[1] * nyquist
                    mask = (f >= min_freq) & (f <= max_freq)
                    Cxy = Cxy[:, mask]

                # Cxy shape: (num_tokens, n_freqs)
                # Take mean across frequencies for each token
                token_coherences = cp.mean(Cxy, axis=1)  # shape: (num_tokens,)

                # Take mean across all tokens
                coherence_mat[i, j] = cp.mean(token_coherences)

        return coherence_mat.get()


class MetricTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name, metric_fun, verbose=False):
        self.name = name
        self.metric_fun = metric_fun
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X shape: (samples, langs, ...) — each sample is independent
        if self.name == "coherence_fun" and len(X[0].shape) == 3:
            import cupy as cp

            X_gpu = [cp.asarray(x) for x in X]
            iterable = track(X_gpu) if self.verbose else X_gpu
            return np.stack([self.metric_fun(x) for x in iterable], axis=0)

        # CPU path: parallelise across samples (metric_fun is stateless, no GIL contention
        # because NumPy/SciPy release the GIL for the heavy arithmetic).
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(self.metric_fun)(x) for x in X
        )
        return np.stack(results, axis=0)
