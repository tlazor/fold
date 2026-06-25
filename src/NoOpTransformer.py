from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class NoOpTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that zero-pads arrays in X so they have the same shape.
    This is useful as a placeholder when no transformation is needed but
    shape consistency is required.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        This transformer doesn't learn anything from the data, so fit does nothing.
        """
        return self

    def transform(self, X):
        """
        Zero-pads arrays in X so they have the same shape.
        
        Parameters
        ----------
        X : list of np.ndarray
            Input data where each array may have different shapes.
            Supports 2D arrays (num_langs, num_tokens), 
            3D arrays (num_langs, num_tokens, hidden_dim)
            
        Returns
        -------
        list of np.ndarray
            The same input data with all arrays padded to the same shape.
        """
        # Find the longest size along axis 1
        max_len = max(x.shape[1] for x in X)
        print(f"{max_len=}")

        # Pad just that axis
        padded_X = [
            np.pad(
                x,
                pad_width=[(0, 0)]                    # axis 0 — keep as-is
                        + [(0, max_len - x.shape[1])] # axis 1 — pad on the right
                        + [(0, 0)] * (x.ndim - 2),    # any remaining axes — keep as-is
                mode="constant",
                constant_values=0
            )
            for x in X
        ]
        print(f"2D case unique shapes: {list(set(x.shape for x in padded_X))}")
        return padded_X
