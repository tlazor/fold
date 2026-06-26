from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModelForMaskedLM

from rich.progress import track

from joblib import Memory
from paths import CACHE_DIR, auto_device

memory = Memory(CACHE_DIR / "joblib", verbose=0)


@memory.cache(ignore=["model", "chunk_size"])
def get_sample_likelihoods(
    model: nn.Module,
    model_name: str,
    all_token_ids,
    all_attention_masks,
    mask_token_id: int,
    chunk_size: int = 32,
) -> list:
    """
    Compute per-token likelihoods for all languages in one sample via a single
    batched forward-pass sequence.

    all_token_ids     : numpy array (n_langs, seq_len)
    all_attention_masks: numpy array (n_langs, seq_len)

    Returns a list of n_langs 1-D tensors, each holding the probability of
    every non-special token in that language's sentence.
    """
    device = model.device
    n_langs = all_token_ids.shape[0]

    ids_t = torch.from_numpy(all_token_ids)    # (n_langs, seq_len)
    masks_t = torch.from_numpy(all_attention_masks)  # (n_langs, seq_len)

    num_normal = masks_t.sum(dim=1).long() - 2  # exclude [CLS] and [SEP]

    # Build a mega-batch: for language l with N_l normal tokens we need N_l masked copies.
    mega_ids_parts = []
    mega_mask_parts = []
    lang_offsets: list[tuple[int, int]] = []
    offset = 0

    for l in range(n_langs):
        n_t = int(num_normal[l].item())
        lang_offsets.append((offset, n_t))
        offset += n_t

        repeated_ids = ids_t[l].unsqueeze(0).repeat(n_t, 1)    # (n_t, seq_len)
        repeated_mask = masks_t[l].unsqueeze(0).repeat(n_t, 1)  # (n_t, seq_len)

        # Row i masks position i+1 (skip [CLS] at 0)
        row_idx = torch.arange(n_t)
        repeated_ids[row_idx, row_idx + 1] = mask_token_id

        mega_ids_parts.append(repeated_ids)
        mega_mask_parts.append(repeated_mask)

    mega_ids = torch.cat(mega_ids_parts, dim=0)    # (total_tokens, seq_len)
    mega_masks = torch.cat(mega_mask_parts, dim=0)  # (total_tokens, seq_len)

    # Forward pass in chunks across the full mega-batch
    total_rows = mega_ids.shape[0]
    logit_chunks = []
    with torch.no_grad():
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            logits = model(
                mega_ids[start:end].to(device),
                attention_mask=mega_masks[start:end].to(device),
            ).logits
            logit_chunks.append(logits.cpu().float())

    all_logits = torch.cat(logit_chunks, dim=0)  # (total_rows, seq_len, vocab_size)

    results = []
    for l, (start_row, n_t) in enumerate(lang_offsets):
        lang_logits = all_logits[start_row : start_row + n_t]  # (n_t, seq_len, vocab_size)
        row_idx = torch.arange(n_t)
        selected = lang_logits[row_idx, row_idx + 1, :]  # (n_t, vocab_size)
        log_probs = F.log_softmax(selected, dim=-1)
        original_ids = ids_t[l, 1 : n_t + 1]
        probs = log_probs[row_idx, original_ids].exp()
        results.append(probs)

    return results


class LikelihoodEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="bert-base-multilingual-cased", mask_token_id=103, device=None):
        self.model_name = model_name
        self.mask_token_id = mask_token_id
        self.device = device

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        device = self.device if self.device is not None else auto_device()
        model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        model.to(device)
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = model.to(dtype)
        model.eval()

        results = []
        for sample in track(X):
            # Stack all language token arrays into (n_langs, seq_len) for batched inference.
            all_ids = np.vstack([ids for ids, _ in sample])      # (n_langs, seq_len)
            all_masks = np.vstack([mask for _, mask in sample])  # (n_langs, seq_len)

            token_arrays = get_sample_likelihoods(
                model,
                self.model_name,
                all_ids,
                all_masks,
                self.mask_token_id,
            )

            max_len = max(t.shape[0] for t in token_arrays)
            results.append(
                torch.vstack(
                    [F.pad(t, (0, max_len - t.shape[0])) for t in token_arrays]
                ).numpy()
            )

        return results
