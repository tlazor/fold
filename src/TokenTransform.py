from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoTokenizer


def tokenize(
    tokenizer, 
    text: str,
    max_tokens:int = 256
) -> list:
    
    encoded = tokenizer(
        text,
        max_length=max_tokens,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    )

    return encoded["input_ids"], encoded["attention_mask"]

class TokenTransform(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-multilingual-cased'):
        self.model_name = model_name

    def fit(self, X, y=None):
        """
        Learn something from the data if needed.
        
        X : array-like or dataframe of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or None
        """
        # This transformer doesn't learn anything from the data,
        # so we just return self.
        return self
    

    def transform(self, X):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, clean_up_tokenization_spaces=True)

        n_langs, n_features = X.shape

        results = []
        for i in range(n_langs):
            # Collect token likelihood arrays for each language in row i
            token_arrays = []
            for j in range(n_features):
                text = X.iloc[i, j]
                # get_token_likelihood returns an array of likelihoods for each token in the text
                # input_ids, attention_mask = tokenize(model, tokenizer, text)
                # token_likelihoods = compute_token_likelihoods_minibatch(model, input_ids, attention_mask, tokenizer.mask_token_id)
                token_arrays.append(tokenize(tokenizer, text))

            # Find the maximum token length for this sample
            # max_length = max(len(arr) for arr in token_arrays)

            # # Create a 2D array for row i: shape = (n_features, max_length)
            # padded_arr = np.zeros((n_features, max_length), dtype=float)

            # # Pad each language's likelihood array
            # for j, arr in enumerate(token_arrays):
            #     padded_arr[j, :len(arr)] = arr

            # Store the padded 2D array in the results list
            results.append(token_arrays)

        return results