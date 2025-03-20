import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import welch


class PsdNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, axis=-1):
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"{X[0].shape=}")
        eps=1e-9
        return [x / (np.sum(x, axis=self.axis, keepdims=True)+eps) for x in X]
