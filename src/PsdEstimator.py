from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import welch
import numpy as np
from scipy.interpolate import interp1d


class PsdEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, nperseg=10, window="hann", scaling="density", axis=-1):
        self.nperseg = nperseg
        self.window = window
        self.scaling = scaling
        self.axis = axis
        # Lazily computed on first transform() call; reused for every sample.
        self._common_freqs = None

    def fit(self, X, y=None):
        return self

    def _aligned_psd(self, psd, freqs):
        """Interpolate one sample's PSD onto the shared frequency grid."""
        if psd.ndim == 3:
            n_langs, n_freqs_orig, n_dims = psd.shape
            psd_2d = psd.transpose(0, 2, 1).reshape(n_langs * n_dims, n_freqs_orig)
            interp_2d = interp1d(freqs, psd_2d, axis=1, fill_value="extrapolate")(
                self._common_freqs
            )
            return interp_2d.reshape(n_langs, n_dims, len(self._common_freqs)).transpose(0, 2, 1)
        interp_func = interp1d(freqs, psd, axis=1, fill_value="extrapolate")
        return interp_func(self._common_freqs)

    def transform(self, X):
        all_freqs = []
        all_psds = []
        for x in X:
            freqs, psd = welch(
                x,
                fs=1,
                nperseg=self.nperseg,
                window=self.window,
                scaling=self.scaling,
                axis=self.axis,
            )
            all_freqs.append(freqs)
            all_psds.append(psd)

        # Build (or reuse) the common frequency grid once for the lifetime of this
        # transformer.  All samples share the same nperseg / fs, so the grid is
        # identical across calls — computing it 600 times per run is wasteful.
        if self._common_freqs is None:
            min_freq = max(f[0] for f in all_freqs)
            max_freq = min(f[-1] for f in all_freqs)
            n_freqs = min(len(f) for f in all_freqs)
            self._common_freqs = np.linspace(min_freq, max_freq, n_freqs)

        return np.array([self._aligned_psd(psd, freqs) for psd, freqs in zip(all_psds, all_freqs)])
