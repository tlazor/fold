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

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # First get all frequency ranges
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

        # Find the common frequency range
        min_freq = max(freqs[0] for freqs in all_freqs)
        max_freq = min(freqs[-1] for freqs in all_freqs)
        n_freqs = min(len(freqs) for freqs in all_freqs)
        common_freqs = np.linspace(min_freq, max_freq, n_freqs)

        # Interpolate each PSD to the common frequency range
        aligned_psds = []
        for freqs, psd in zip(all_freqs, all_psds):
            # Reshape psd to 2D if it's 3D
            if psd.ndim == 3:
                n_langs, n_freqs_orig, n_dims = psd.shape
                # Transpose to (langs, dims, freqs) so each row after reshape is one
                # (lang, dim) PSD curve across all frequency bins.
                psd_reshaped = psd.transpose(0, 2, 1).reshape(n_langs * n_dims, n_freqs_orig)
            else:
                psd_reshaped = psd

            # Create interpolation function
            interp_func = interp1d(
                freqs, psd_reshaped, axis=1, fill_value="extrapolate"
            )
            # Interpolate to common frequencies
            aligned_psd = interp_func(common_freqs)

            # Reshape back to original dimensions if needed
            if psd.ndim == 3:
                aligned_psd = aligned_psd.reshape(n_langs, n_dims, len(common_freqs)).transpose(0, 2, 1)

            aligned_psds.append(aligned_psd)

        return np.array(aligned_psds)
