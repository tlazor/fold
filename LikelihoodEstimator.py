from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM

import fold_globals


# @torch.compile
def get_token_likelihood(model, tokenizer, text):
    """
    Masks each token (except [CLS] and [SEP]) in the input and returns 
    the likelihood (probability) of the original token under the model.
    
    Parameters
    ----------
    model : PreTrainedModel (e.g., a BERT-like model)
        A Hugging Face transformer model.
    tokenizer : PreTrainedTokenizer
        The corresponding tokenizer for the model.
    text : str
        The input text.

    Returns
    -------
    token_likelihoods : np.ndarray, shape = (num_normal_tokens,)
        The likelihood/probability for each 'normal' token in the text 
        (skipping special tokens).
    """
    max_tokens = 256
    device = model.device  # The device on which the model is loaded

    # Encode text with padding/truncation up to max_tokens
    # Also returns an attention_mask that shows which positions are real tokens (1) vs. padding (0)
    encoded = tokenizer(
        text,
        max_length=max_tokens,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Count how many tokens are actually present (excluding padding)
    total_tokens = attention_mask[0].sum().item()  # number of non-pad tokens
    # For a typical BERT-like model, index 0 is [CLS], and index = total_tokens - 1 is [SEP]
    num_normal_tokens = max(0, int(total_tokens) - 2)

    token_likelihoods = []

    # Loop over the 'normal' tokens, skipping [CLS] at index 0 and [SEP] at index (total_tokens-1)
    for i in range(1, 1 + num_normal_tokens):
        # Clone the input so we don't overwrite anything
        masked_input_ids = input_ids.clone()
        # Mask the i-th token in the sequence
        masked_input_ids[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_input_ids)
            predictions = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Apply softmax to the i-th token's predictions
        softmax_probs = F.softmax(predictions[0, i], dim=-1)

        # The "original" token at position i in the unmasked sequence
        original_token_id = input_ids[0, i]
        original_token_prob = softmax_probs[original_token_id].item()
        token_likelihoods.append(original_token_prob)

    return np.array(token_likelihoods)


class LikelihoodEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-multilingual-uncased'):
        self.model_name = model_name

    def fit(self, X, y=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        self.model.to(fold_globals.DEVICE)
        self.model.eval()
        """
        Learn something from the data if needed.
        
        X : array-like or dataframe of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or None
        """
        # This transformer doesn't learn anything from the data,
        # so we just return self.
        return self
    

    def transform(self, X):
        """
        Computes token likelihoods for each cell in the DataFrame `X`.
        For each sample (row), we:
        1. Get an array of token likelihoods for each language (column).
        2. Find the max token length among all languages for that sample.
        3. Pad each likelihood array to that max length.
        4. Combine into an array of shape (n_features, max_length_for_sample).

        Parameters
        ----------
        X : pd.DataFrame
            Each row corresponds to one "sample", and each column is a language's text.
            E.g., X might have columns ["ar", "bg", "de", "en", ...].

        Returns
        -------
        results : list of np.ndarray
            A list of length n_samples, where each element has shape (n_features, max_len_for_that_sample).
            - n_features = number of language columns
            - max_len_for_that_sample = max token length among the language texts for that sample
        """
        n_samples, n_features = X.shape

        results = []
        for i in range(n_samples):
            # Collect token likelihood arrays for each language in row i
            token_arrays = []
            for j in range(n_features):
                text = X.iloc[i, j]
                # get_token_likelihood returns an array of likelihoods for each token in the text
                token_likelihoods = get_token_likelihood(self.model, self.tokenizer, text)
                token_arrays.append(token_likelihoods)

            # Find the maximum token length for this sample
            max_length = max(len(arr) for arr in token_arrays)

            # Create a 2D array for row i: shape = (n_features, max_length)
            padded_arr = np.zeros((n_features, max_length), dtype=float)

            # Pad each language's likelihood array
            for j, arr in enumerate(token_arrays):
                padded_arr[j, :len(arr)] = arr

            # Store the padded 2D array in the results list
            results.append(padded_arr)

        return results
