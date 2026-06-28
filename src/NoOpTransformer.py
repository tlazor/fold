from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class NoOpTransformer(BaseEstimator, TransformerMixin):
    """Zero-pads arrays in X to a common shape along axis 1."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        max_len = max(x.shape[1] for x in X)
        return [
            np.pad(
                x,
                pad_width=[(0, 0)]                     # axis 0 — keep as-is
                        + [(0, max_len - x.shape[1])]  # axis 1 — pad on the right
                        + [(0, 0)] * (x.ndim - 2),     # any remaining axes — keep as-is
                mode="constant",
                constant_values=0,
            )
            for x in X
        ]
