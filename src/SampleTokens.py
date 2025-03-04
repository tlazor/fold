import random
from sklearn.base import BaseEstimator, TransformerMixin

import constants

class SampleTokens(BaseEstimator, TransformerMixin):
    def __init__(self, num_samples=500, minimum_tokens=10, seed=0):
        self.num_samples = num_samples
        self.minimum_tokens = minimum_tokens
        self.seed = seed


    def fit(self, X, y=None):
        return self
    

    def transform(self, X):
        random.seed(self.seed)
        # list of samples of 15 langs
        filtered = [
            [(lang_ids, lang_masks)
            for (lang_ids, lang_masks) in sample
            if lang_masks.sum() >= self.minimum_tokens]
            for sample in X
        ]

        filtered = [
            sample
            for sample in filtered
            if len(sample) == len(constants.LANGUAGES)
        ]

        return random.sample(filtered, self.num_samples)