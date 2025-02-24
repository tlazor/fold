from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM

from rich.progress import track

import fold_globals


# @torch.compile
def tokenize(model, tokenizer, text):
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

    return input_ids, attention_mask


@torch.compile
def compute_token_likelihoods(model, input_ids, attention_mask, mask_token_id):
    """
    Compute the likelihood of each 'normal' token (excluding [CLS] and [SEP])
    under a masked-language-modeling BERT-like model, but do it in a single 
    batched forward pass for speed.
    """
    # Count how many tokens are actually present (excluding padding)
    total_tokens = attention_mask[0].sum().item()  # number of non-pad tokens
    
    # For a typical BERT, index 0 is [CLS], and index total_tokens-1 is [SEP]
    num_normal_tokens = max(0, total_tokens - 2)
    if num_normal_tokens == 0:
        return np.array([])

    # 1) Create a batch where each row is the same input_ids, except one token is masked.
    #    We'll replicate 'input_ids' and 'attention_mask' num_normal_tokens times.
    batch_input_ids = input_ids.repeat(num_normal_tokens, 1)           # shape: (num_normal_tokens, seq_len)
    batch_attention_mask = attention_mask.repeat(num_normal_tokens, 1) # shape: (num_normal_tokens, seq_len)

    # 2) Mask exactly one token in each row: row i will have token (i+1) masked
    #    (since we skip [CLS] at index 0).
    for i in range(num_normal_tokens):
        token_pos = i + 1  # skip [CLS] at index 0
        batch_input_ids[i, token_pos] = mask_token_id

    # 3) Single forward pass on the entire batch
    with torch.no_grad():
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
        # logits.shape => (num_normal_tokens, seq_len, vocab_size)
        logits = outputs.logits

    # 4) Extract probabilities for the originally unmasked token in each row
    token_likelihoods = []
    for i in range(num_normal_tokens):
        token_pos = i + 1
        original_token_id = input_ids[0, token_pos]

        # Softmax over vocab dimension for row i, token_pos
        probs = F.softmax(logits[i, token_pos], dim=-1)
        original_token_prob = probs[original_token_id].item()

        token_likelihoods.append(original_token_prob)

    return np.array(token_likelihoods)



@torch.compile
def compute_token_likelihoods_minibatch(
    model, 
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor, 
    mask_token_id, 
    batch_size: int = 16
):
    """
    Compute likelihood of each 'normal' token (excluding [CLS] and [SEP])
    under a BERT-like model in minibatches to avoid large memory usage.

    Args:
        model: A BERT-like language model (with MLM head).
        input_ids: Tensor of shape [1, seq_len].
        attention_mask: Tensor of shape [1, seq_len].
        tokenizer: Tokenizer with a `mask_token_id`.
        batch_size: Number of tokens to mask/process at once.

    Returns:
        A 1D NumPy array containing the likelihood (probability) of each
        normal token under MLM, in sequence order.
    """
    # Count how many tokens are actually present (excluding padding)
    total_tokens = attention_mask[0].sum().item()  # number of non-pad tokens
    # For a typical BERT-like model, index 0 is [CLS], and index total_tokens-1 is [SEP]
    num_normal_tokens = max(0, total_tokens - 2)

    # If there's nothing to compute, return an empty array
    if num_normal_tokens <= 0:
        return np.array([])

    # We'll store the probabilities for each normal token
    token_likelihoods = np.zeros(num_normal_tokens, dtype=np.float32)

    # Process in minibatches
    start_idx = 0
    while start_idx < num_normal_tokens:
        end_idx = min(start_idx + batch_size, num_normal_tokens)
        this_batch_size = end_idx - start_idx

        # 1) Create the chunked batch:
        #    We'll replicate input_ids and attention_mask `this_batch_size` times
        batch_input_ids = input_ids.repeat(this_batch_size, 1).clone()
        batch_attention_mask = attention_mask.repeat(this_batch_size, 1)

        # 2) For each row i in this minibatch, mask the token at position (start_idx + i + 1)
        for i in range(this_batch_size):
            token_pos = (start_idx + i) + 1  # skip [CLS] at index 0
            batch_input_ids[i, token_pos] = mask_token_id

        # 3) Single forward pass for this minibatch
        with torch.no_grad():
            logits = model(batch_input_ids, attention_mask=batch_attention_mask).logits
            # logits.shape => (this_batch_size, seq_len, vocab_size)
            # logits = outputs.logits

        # 4) Extract the probability of the original token in each row
        for i in range(this_batch_size):
            token_pos = (start_idx + i) + 1
            original_token_id = input_ids[0, token_pos]
            token_likelihoods[start_idx + i] = F.softmax(logits[i, token_pos], dim=-1)[original_token_id].item()

        # Advance to the next minibatch
        start_idx = end_idx

    return token_likelihoods


class LikelihoodEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-multilingual-uncased'):
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

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, clean_up_tokenization_spaces=True)
        model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        model.to(fold_globals.DEVICE)
        model.eval()

        results = []
        for i in track(range(n_samples)):
            # Collect token likelihood arrays for each language in row i
            token_arrays = []
            for j in range(n_features):
                text = X.iloc[i, j]
                # get_token_likelihood returns an array of likelihoods for each token in the text
                input_ids, attention_mask = tokenize(model, tokenizer, text)
                token_likelihoods = compute_token_likelihoods_minibatch(model, input_ids, attention_mask, tokenizer.mask_token_id)
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
