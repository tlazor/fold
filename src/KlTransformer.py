from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def kl_divergence_matrix(P, epsilon=1e-15):
    """
    Compute the KL divergence for all pairs of distributions
    in a 2D array P of shape (num_langs, num_samples),
    returning an array of shape (num_langs, num_langs).
    """
    # 1) Normalize each row (if not already).
    P = P / P.sum(axis=1, keepdims=True)
    
    # 2) Compute log(P). Add a small epsilon to avoid log(0).
    LP = np.log(P + epsilon)
    
    # 3) alpha[i] = sum_x p[i, x] * log(p[i, x])
    alpha = np.sum(P * LP, axis=1)
    
    # 4) beta[i, j] = sum_x p[i, x] * log(p[j, x])
    beta = P.dot(LP.T)  # shape (L, L)
    
    # 5) KL[i, j] = alpha[i] - beta[i, j]
    KL = alpha[:, np.newaxis] - beta  # shape (L, L)
    return KL


class KlTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, temp='fourier'):
        self.temp = temp

    def fit(self, X, y=None):
        return self
    

    def transform(self, X):
        # samples x langs x langs
        return np.stack([kl_divergence_matrix(x) for x in X], axis=0)
