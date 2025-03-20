from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import welch


class PsdEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, nperseg=10, window="hann", scaling="density", axis=-1):
        self.nperseg = nperseg
        self.window = window
        self.scaling = scaling
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # shape: [num_langs, num_tokens, hidden_dim]
        print(f"{X[0].shape=}")

        return [
            welch(
                x, fs=1, nperseg=self.nperseg, window=self.window, scaling=self.scaling, axis=self.axis
            )[1]
            for x in X
        ]
