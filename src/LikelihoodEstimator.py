from sklearn.base import BaseEstimator, TransformerMixin

import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModelForMaskedLM

from rich.progress import track

from joblib import Memory
from paths import CACHE_DIR

memory = Memory(CACHE_DIR / "joblib", verbose=0)


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@memory.cache(ignore=["model", "chunk_size"])
@torch.compile
def get_token_likelihood_vec(
    model: nn.Module,
    model_name: str,
    token_ids,
    attention_mask,
    mask_token_id,
    chunk_size: int = 10,
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

    # joblib cant currently deterministically hash tensors (they contain metadata about storage)
    token_ids = torch.from_numpy(token_ids)
    attention_mask = torch.from_numpy(attention_mask)

    # 2) Determine how many real tokens (excluding special tokens) to consider
    #    For many BERT-like models, there's 1 CLS token (index 0) and 1 SEP token at the end.
    #    Adjust according to your special-token scheme if needed.
    valid_token_count = attention_mask[0].sum().item()  # number of non-pad
    num_normal_tokens = valid_token_count - 2  # skip [CLS] & [SEP]

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
    original_token_ids = token_ids[
        0, 1 : num_normal_tokens + 1
    ]  # shape: (num_normal_tokens,)

    # 4) Forward pass in chunks
    all_likelihoods = []
    with torch.no_grad():
        for start in range(0, num_normal_tokens, chunk_size):
            end = min(start + chunk_size, num_normal_tokens)

            # Slice out this chunk (still on CPU); then move to GPU
            masked_ids_chunk = masked_input_ids_cpu[start:end].to(device)
            attention_chunk = masked_attention_cpu[start:end].to(device)

            # Model forward
            logits_chunk = model(
                masked_ids_chunk, attention_mask=attention_chunk
            ).logits
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
                torch.arange(end - start, device=device), chunk_original_ids
            ]

            # Convert to probabilities and move to CPU immediately
            chunk_probs = chunk_log_probs.exp().cpu()
            all_likelihoods.append(chunk_probs)

            # Clear GPU cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 5) Concatenate all chunk results
    all_likelihoods = torch.cat(all_likelihoods, dim=0)  # shape: (num_normal_tokens,)

    return all_likelihoods


class LikelihoodEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="bert-base-multilingual-cased", mask_token_id=103, device=None):
        self.model_name = model_name
        self.mask_token_id = mask_token_id
        self.device = device

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        device = self.device if self.device is not None else _auto_device()
        model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        model.to(device)
        model.eval()

        results = []
        for sample in track(X):
            token_arrays = []
            max_len = 0
            for input_ids, attention_mask in sample:
                token_likelihoods = get_token_likelihood_vec(
                    model,
                    self.model_name,
                    input_ids,
                    attention_mask,
                    self.mask_token_id,
                )
                token_arrays.append(token_likelihoods)

                max_len = max(max_len, token_likelihoods.shape[0])

            results.append(
                torch.vstack(
                    [
                        torch.nn.functional.pad(t, (0, max_len - t.shape[0]))
                        for t in token_arrays
                    ]
                ).numpy()
            )

        return results
