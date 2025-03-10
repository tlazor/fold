import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import welch


class PsdNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [x / np.sum(x, axis=-1, keepdims=True) for x in X]
