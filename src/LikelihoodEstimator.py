from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

import torch
from torch import nn
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
    batch_size: int = 1
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
    total_tokens = attention_mask[0].sum()  # number of non-pad tokens
    # For a typical BERT-like model, index 0 is [CLS], and index total_tokens-1 is [SEP]
    num_normal_tokens = max(0, total_tokens - 2)

    # If there's nothing to compute, return an empty array
    if num_normal_tokens <= 0:
        return np.array([])

    # We'll store the probabilities for each normal token
    token_likelihoods = np.zeros(num_normal_tokens, dtype=np.float16)

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


# @torch.compile
def get_token_likelihood_vec(
    model: nn.Module, 
    token_ids,
    attention_mask,
    mask_token_id,
    chunk_size: int = 10
) -> list:
    """
    Compute the likelihood (probability) of each non-special token in `text`
    by masking each token and querying the model for its probability.
    
    Uses a chunked approach so it fits in ~8GB GPU memory more easily.
    
    Args:
        model: HuggingFace-like model (e.g. BERT)
        tokenizer: Corresponding tokenizer
        text: Input text (single string)
        chunk_size: How many masked positions to handle per forward pass
    
    Returns:
        A list of probabilities (floats in [0,1]) for each non-special token.
    """
    device = model.device
    # # 1) Encode the text
    # max_tokens = 256
    # device = next(model.parameters()).device  # or model.device if you have that
    # encoded = tokenizer(
    #     text,
    #     max_length=max_tokens,
    #     padding="max_length",
    #     truncation=True,
    #     return_tensors="pt",
    #     add_special_tokens=True
    # )
    
    # token_ids = encoded["input_ids"]          # shape: (1, max_tokens) on CPU
    # attention_mask = encoded["attention_mask"]  # shape: (1, max_tokens) on CPU
    
    # 2) Determine how many real tokens (excluding special tokens) to consider
    #    For many BERT-like models, there's 1 CLS token (index 0) and 1 SEP token at the end.
    #    Adjust according to your special-token scheme if needed.
    valid_token_count = attention_mask[0].sum().item()  # number of non-pad
    num_normal_tokens = valid_token_count - 2           # skip [CLS] & [SEP]

    # 3) Create repeated copies of the input with one token masked each time.
    #    We'll do this on **CPU** to avoid excessive GPU usage.
    #    Shape: (num_normal_tokens, max_tokens)
    masked_input_ids_cpu = token_ids.repeat(num_normal_tokens, 1)
    masked_attention_cpu = attention_mask.repeat(num_normal_tokens, 1)
    
    # Mask the positions [1..num_normal_tokens] (skip the CLS token at index 0)
    for i in range(num_normal_tokens):
        position_to_mask = i + 1
        masked_input_ids_cpu[i, position_to_mask] = mask_token_id
    
    # We'll need the original token IDs for indexing probabilities:
    original_token_ids = token_ids[0, 1 : num_normal_tokens + 1]  # shape: (num_normal_tokens,)

    # 4) Forward pass in chunks
    all_likelihoods = []
    with torch.no_grad():
        for start in range(0, num_normal_tokens, chunk_size):
            end = min(start + chunk_size, num_normal_tokens)
            
            # Slice out this chunk (still on CPU); then move to GPU
            masked_ids_chunk = masked_input_ids_cpu[start:end].to(device)
            attention_chunk = masked_attention_cpu[start:end].to(device)
            
            # Model forward
            logits_chunk = model(masked_ids_chunk, attention_mask=attention_chunk).logits
            # shape: (chunk_size, max_tokens, vocab_size)
            
            # The position masked in sample `i` (global index) is (i+1).
            # For chunked indexing:
            #   - global_i in [start..end-1]
            #   - local_i in [0..(end-start)-1]
            #   - col_idx = global_i + 1
            local_range = torch.arange(start, end, device=device)
            col_indices = local_range + 1  # the masked positions for the chunk
            local_i = local_range - start  # local row indices
            
            # Extract the logits for the masked position
            # shape: (chunk_size, vocab_size)
            selected_logits_chunk = logits_chunk[local_i, col_indices, :]
            
            # Convert to log_probs (then we can exponentiate if we want probabilities)
            selected_log_probs_chunk = F.log_softmax(selected_logits_chunk, dim=-1)
            
            # Gather the original token ID
            # original_token_ids is on CPU, so move it or index directly
            chunk_original_ids = original_token_ids[start:end].to(device)
            
            # The log-prob for the correct token:
            # shape: (end - start)
            chunk_log_probs = selected_log_probs_chunk[
                torch.arange(end - start, device=device),
                chunk_original_ids
            ]
            
            # Convert to probabilities
            chunk_probs = chunk_log_probs.exp()
            
            # Store in a list (still on GPU). Move to CPU if you want to keep GPU free.
            all_likelihoods.append(chunk_probs.cpu())
    
    # 5) Concatenate all chunk results
    all_likelihoods = torch.cat(all_likelihoods, dim=0)  # shape: (num_normal_tokens,)

    return all_likelihoods


class LikelihoodEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-multilingual-cased', mask_token_id=250026):
        self.model_name = model_name
        self.mask_token_id = mask_token_id

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
        model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        model.to(fold_globals.DEVICE)
        model.eval()

        results = []
        for sample in track(X):
            # Collect token likelihood arrays for each language in row i
            token_arrays = []
            for input_ids, attention_mask in sample:
                print(f"{input_ids.shape=}") 
                token_likelihoods = get_token_likelihood_vec(model, input_ids, attention_mask, self.mask_token_id)
                token_arrays.append(token_likelihoods)

            results.append(token_arrays)

        return results
