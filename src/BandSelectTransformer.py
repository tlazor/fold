import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BandSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, freq_band, verbose=False):
        self.freq_band = freq_band
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Frequency axis is always axis=1: (langs, freq) or (langs, freq, hidden_dim).
        n_freq = X[0].shape[1]
        min_idx = int(n_freq * self.freq_band[0])
        max_idx = int(n_freq * self.freq_band[1])

        if len(X[0].shape) == 2:
            return np.stack([x[:, min_idx:max_idx] for x in X], axis=0)
        elif len(X[0].shape) == 3:
            return np.stack([x[:, min_idx:max_idx, :] for x in X], axis=0)
