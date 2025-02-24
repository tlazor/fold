from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import welch


class PsdEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, nperseg=10, window='hann', scaling='density'):
        self.nperseg = nperseg
        self.window = window
        self.scaling = scaling

    def fit(self, X, y=None):
        return self
    

    def transform(self, X):
        # f, Pxx = welch
        # return [welch(x, average='median')[1] for x in X]
        return [welch(x, fs=1, nperseg=self.nperseg, window=self.window, scaling=self.scaling)[1] for x in X]
