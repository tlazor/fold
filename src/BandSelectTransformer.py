from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from rich.progress import track

class BandSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, freq_band, verbose=False):
        self.freq_band = freq_band
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Calculate indices for band boundaries
        min_idx = int(X[0].shape[-1] * self.freq_band[0])
        max_idx = int(X[0].shape[-1] * self.freq_band[1])

        print(f"{min_idx=} {max_idx=}")
        if len(X[0].shape) == 2:
            # 2D case: (num_samples, num_langs, num_freq)
            return np.stack([x[:, min_idx:max_idx] for x in X], axis=0)
        elif len(X[0].shape) == 3:
            # 3D case: (num_samples, num_langs, num_tokens, num_freq)
            return np.stack([x[:, :, min_idx:max_idx] for x in X], axis=0)

