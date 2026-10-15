from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.signal import get_window
from joblib import Parallel, delayed
from rich.progress import track

try:
    import numba

    # Limit Numba's internal thread pool to 1: transform() uses joblib Parallel
    # across samples, so Numba parallelism is already provided at that level.
    # Without this, each joblib thread spawns a full Numba TBB/OpenMP sub-pool,
    # creating up to cpu_count² threads and risking TBB deadlocks.
    numba.set_num_threads(1)

    @numba.njit(parallel=True, cache=True)
    def _overlap_3d_nb(freq_spectra):
        """Pairwise min-sum overlap for (L, F, D) spectra without a large (L,L,F,D) temporary."""
        L, F, D = freq_spectra.shape
        result = np.zeros((L, L), dtype=np.float64)
        for i in numba.prange(L):
            for j in range(L):
                total = 0.0
                for f in range(F):
                    for d in range(D):
                        a = freq_spectra[i, f, d]
                        b = freq_spectra[j, f, d]
                        total += a if a < b else b
                result[i, j] = total / D
        return result

    @numba.njit(parallel=True, cache=True)
    def _kl_beta_3d_nb(P_norm, LP):
        """β[li, L, t] = Σ_s P_norm[li,t,s] * LP[L,t,s] — parallel over the li axis."""
        L, T, S = P_norm.shape
        beta = np.zeros((L, L, T), dtype=np.float64)
        for li in numba.prange(L):
            for ll in range(L):
                for t in range(T):
                    acc = 0.0
                    for s in range(S):
                        acc += P_norm[li, t, s] * LP[ll, t, s]
                    beta[li, ll, t] = acc
        return beta

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


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
        if _NUMBA_AVAILABLE:
            try:
                # Avoids allocating the large (L, L, F, D) intermediate array.
                overlap_matrix = _overlap_3d_nb(
                    np.ascontiguousarray(freq_spectra, dtype=np.float64)
                )
            except Exception:
                pairwise_mins = np.minimum(
                    freq_spectra[:, np.newaxis, :, :],
                    freq_spectra[np.newaxis, :, :, :],
                )
                overlap_matrix = pairwise_mins.sum(axis=2).mean(axis=2)
        else:
            pairwise_mins = np.minimum(
                freq_spectra[:, np.newaxis, :, :],
                freq_spectra[np.newaxis, :, :, :],
            )
            overlap_matrix = pairwise_mins.sum(axis=2).mean(axis=2)

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
        expP = np.exp(P_shift)
        Z = expP.sum(axis=-1, keepdims=True)  # partition function
        P_norm = expP / (Z + epsilon)  # shape (L, T, S)

        # 2) log(P_norm)
        LP = np.log(P_norm + epsilon)  # shape (L, T, S)

        # 3) α[i, t] = Σ_x  p[i,t,x] log p[i,t,x]
        alpha = np.sum(P_norm * LP, axis=-1)  # shape (L, T)

        # 4) β[i, j, t] = Σ_x  p[i,t,x] log p[j,t,x]
        if _NUMBA_AVAILABLE:
            try:
                beta = _kl_beta_3d_nb(
                    np.ascontiguousarray(P_norm, dtype=np.float64),
                    np.ascontiguousarray(LP, dtype=np.float64),
                )
            except Exception:
                beta = np.einsum("lts,Lts->lLt", P_norm, LP)  # shape (L, L, T)
        else:
            beta = np.einsum("lts,Lts->lLt", P_norm, LP)  # shape (L, L, T)

        # 5) KL[i, j, t] = α[i,t] − β[i,j,t]
        KL_per_token = alpha[:, None, :] - beta  # shape (L, L, T)

        # Option A: mean over tokens (default)
        KL = KL_per_token.mean(axis=-1)  # shape (L, L)
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


def _array_module(P):
    """CuPy for CuPy input, or for 3-D (embedding) input when a GPU is usable."""
    try:
        import cupy as cp
    except ImportError:
        return np
    if isinstance(P, cp.ndarray):
        return cp
    if P.ndim == 3:
        try:
            cp.cuda.runtime.getDeviceCount()
            return cp
        except cp.cuda.runtime.CUDARuntimeError:
            return np
    return np


def coherence_spectra(P, fs=1.0, nperseg=None, xp=None):
    """
    Magnitude-squared coherence between every pair of languages, per frequency.

    Same estimator as scipy.signal.coherence (Welch averaging with a Hann
    window, 50% overlap and constant detrending), but each language's segment
    FFTs are computed once and all pairs are formed in one array operation,
    instead of calling coherence() for every pair.

    Parameters
    ----------
    P : array of shape (num_langs, num_tokens) or (num_langs, num_tokens, hidden_dim)
        Coherence is computed along the token axis; for 3-D input, separately
        for each hidden dimension and then averaged over them.
    fs : float, optional
        Sampling frequency. Default is 1.0.
    nperseg : int, optional
        Segment length. Default (None) is scipy's: 256, capped at num_tokens.
    xp : module, optional
        numpy or cupy. Default picks CuPy for 3-D input when a GPU is usable.

    Returns
    -------
    freqs : np.ndarray of shape (n_freqs,)
    coh : array of shape (num_langs, num_langs, n_freqs), on the device of `xp`
    """
    xp = _array_module(P) if xp is None else xp
    P = xp.asarray(P, dtype=xp.float64)
    if P.ndim == 3:
        P = P.transpose(0, 2, 1)  # (langs, hidden_dim, tokens): one signal per dim

    n_tokens = P.shape[-1]
    nperseg = min(256 if nperseg is None else nperseg, n_tokens)
    step = nperseg - nperseg // 2  # scipy's default noverlap is nperseg // 2
    n_seg = (n_tokens - nperseg) // step + 1
    idx = xp.arange(nperseg)[None, :] + step * xp.arange(n_seg)[:, None]

    seg = P[..., idx]  # (langs, [hidden_dim,] n_seg, nperseg)
    seg = seg - seg.mean(axis=-1, keepdims=True)
    window = xp.asarray(get_window("hann", nperseg))
    X = xp.fft.rfft(seg * window, axis=-1)  # (langs, [hidden_dim,] n_seg, n_freqs)

    # Welch's scaling factors appear in both numerator and denominator of
    # |Pxy|^2 / (Pxx Pyy) and cancel, so plain segment averages suffice.
    pxx = (X.real**2 + X.imag**2).mean(axis=-2)
    pxy = xp.einsum("i...sf,j...sf->ij...f", X.conj(), X) / n_seg
    coh = (pxy.real**2 + pxy.imag**2) / (pxx[:, None] * pxx[None, :])
    if coh.ndim == 4:
        coh = coh.mean(axis=2)  # average over hidden dimensions

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, coh


def coherence_band_means(freqs, coh, freq_bands, fs=1.0):
    """
    Average coherence spectra over each frequency band.

    freq_bands are (min, max) pairs as proportions of the Nyquist frequency,
    e.g. (0.4, 0.5) for 40-50%; None means all frequencies. Returns one
    (num_langs, num_langs) NumPy array per band.
    """
    nyquist = fs / 2
    means = []
    for band in freq_bands:
        selected = np.arange(len(freqs))
        if band is not None:
            selected = np.flatnonzero(
                (freqs >= band[0] * nyquist) & (freqs <= band[1] * nyquist)
            )
        # freqs is ascending, so a band is a contiguous slice (works for CuPy too)
        band_slice = slice(selected[0], selected[-1] + 1) if len(selected) else slice(0)
        band_mean = coh[..., band_slice].mean(axis=-1)
        means.append(band_mean.get() if hasattr(band_mean, "get") else band_mean)
    return means


def coherence_matrix(P, fs=1.0, nperseg=None, freq_band=None):
    """
    Compute the mean coherence between all pairs of languages in P.

    Parameters
    ----------
    P : np.ndarray
        Input array of shape (num_langs, num_tokens) or (num_langs, num_tokens, hidden_dim).
        Coherence is computed along the token axis; for 3-D input, separately for
        each hidden dimension and then averaged.
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
    freqs, coh = coherence_spectra(P, fs=fs, nperseg=nperseg)
    return coherence_band_means(freqs, coh, [freq_band], fs=fs)[0]


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
            # Runs on the GPU one sample at a time (see coherence_spectra).
            iterable = track(X) if self.verbose else X
            return np.stack([self.metric_fun(x) for x in iterable], axis=0)

        # CPU path: parallelise across samples (metric_fun is stateless, no GIL contention
        # because NumPy/SciPy release the GIL for the heavy arithmetic).
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(self.metric_fun)(x) for x in X
        )
        return np.stack(results, axis=0)
