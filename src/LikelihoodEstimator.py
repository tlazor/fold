import numpy as np
import torch
import torch.nn.functional as F
from joblib import Memory
from rich.progress import track
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn
from transformers import AutoModelForMaskedLM

from paths import CACHE_DIR, auto_device, load_hf_model

memory = Memory(CACHE_DIR / "joblib", verbose=0)


def _compute_chunk_size(
    model: nn.Module,
    seq_len: int,
    max_vram_bytes: int | None = None,
    headroom: float = 0.70,
) -> int:
    """
    Return the largest chunk_size whose forward-pass peak fits in VRAM.

    CPU RAM is not a concern: each chunk now produces a (chunk,) float tensor
    before moving to CPU, so accumulated CPU tensors are O(total_rows).

    The dominant VRAM activation is the logit tensor (chunk, seq_len, vocab_size).
    A 2x safety factor covers attention matrices and hidden states on top of that.

    Args:
        model:          loaded model (used for vocab_size, dtype, and device).
        seq_len:        padded sequence length from the tokenizer.
        max_vram_bytes: explicit VRAM budget in bytes; if None, queries free VRAM.
        headroom:       fraction of free VRAM to actually use (default 0.70).
    """
    device = next(model.parameters()).device
    if device.type != "cuda":
        return 64

    if max_vram_bytes is not None:
        model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        available = max(0, max_vram_bytes - model_bytes)
    else:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        available = int(free_bytes * headroom)

    vocab_size = model.config.vocab_size
    dtype_bytes = next(model.parameters()).element_size()
    bytes_per_row = seq_len * vocab_size * dtype_bytes * 2  # 2x for attn/hidden overhead

    return max(1, available // bytes_per_row)


@memory.cache(ignore=["model", "chunk_size"])
def get_sample_likelihoods(
    model: nn.Module,
    model_name: str,
    all_token_ids,
    all_attention_masks,
    mask_token_id: int,
    chunk_size: int = 10,
    model_dtype: str = "torch.float32",
) -> list:
    """
    Compute per-token likelihoods for all languages in one sample via a single
    batched forward-pass sequence.

    all_token_ids     : numpy array (n_langs, seq_len)
    all_attention_masks: numpy array (n_langs, seq_len)

    Returns a list of n_langs 1-D tensors, each holding the probability of
    every non-special token in that language's sentence.
    """
    device = next(model.parameters()).device
    n_langs = all_token_ids.shape[0]

    ids_t = torch.from_numpy(all_token_ids)  # (n_langs, seq_len)
    masks_t = torch.from_numpy(all_attention_masks)  # (n_langs, seq_len)

    num_normal = masks_t.sum(dim=1).long() - 2  # exclude [CLS] and [SEP]

    # Build a mega-batch: for language l with N_l normal tokens we need N_l masked copies.
    # Precompute per-row extraction targets so we can reduce on GPU before moving to CPU.
    mega_ids_parts = []
    mega_mask_parts = []
    lang_offsets: list[tuple[int, int]] = []
    extract_pos_parts = []   # which seq position holds the masked logit for each row
    orig_id_parts = []       # the original token id at that position
    offset = 0

    for lang_idx in range(n_langs):
        n_t = int(num_normal[lang_idx].item())
        lang_offsets.append((offset, n_t))
        offset += n_t

        repeated_ids = ids_t[lang_idx].unsqueeze(0).repeat(n_t, 1)  # (n_t, seq_len)
        repeated_mask = masks_t[lang_idx].unsqueeze(0).repeat(n_t, 1)  # (n_t, seq_len)

        row_idx = torch.arange(n_t)
        repeated_ids[row_idx, row_idx + 1] = mask_token_id

        mega_ids_parts.append(repeated_ids)
        mega_mask_parts.append(repeated_mask)
        extract_pos_parts.append(row_idx + 1)
        orig_id_parts.append(ids_t[lang_idx, 1 : n_t + 1])

    mega_ids = torch.cat(mega_ids_parts, dim=0)          # (total_rows, seq_len)
    mega_masks = torch.cat(mega_mask_parts, dim=0)        # (total_rows, seq_len)
    extract_pos = torch.cat(extract_pos_parts, dim=0)     # (total_rows,)
    orig_ids = torch.cat(orig_id_parts, dim=0)            # (total_rows,)

    total_rows = mega_ids.shape[0]
    if total_rows == 0:
        return [torch.empty(0) for _ in range(n_langs)]

    # Forward pass in chunks. Reduce (chunk, seq_len, vocab) → (chunk,) on GPU before
    # moving to CPU so that accumulated CPU tensors are O(total_rows) not
    # O(total_rows × seq_len × vocab_size). On OOM, halve chunk_size and retry.
    prob_chunks = []
    current_chunk = chunk_size
    start = 0
    with torch.no_grad():
        while start < total_rows:
            end = min(start + current_chunk, total_rows)
            try:
                logits = model(
                    mega_ids[start:end].to(device),
                    attention_mask=mega_masks[start:end].to(device),
                ).logits  # (chunk, seq_len, vocab_size) — stays on GPU

                chunk_idx = torch.arange(end - start, device=device)
                pos = extract_pos[start:end].to(device)
                tgt = orig_ids[start:end].to(device)

                logit_at_mask = logits[chunk_idx, pos, :].float()  # (chunk, vocab_size)
                log_probs = F.log_softmax(logit_at_mask, dim=-1)   # (chunk, vocab_size)
                probs = log_probs[chunk_idx, tgt].exp()             # (chunk,)

                prob_chunks.append(probs.cpu())
                start = end
            except torch.cuda.OutOfMemoryError:
                if current_chunk == 1:
                    raise
                torch.cuda.empty_cache()
                current_chunk = max(1, current_chunk // 2)

    all_probs = torch.cat(prob_chunks, dim=0)  # (total_rows,)

    return [
        all_probs[start_row : start_row + n_t]
        for start_row, n_t in lang_offsets
    ]


class LikelihoodEstimator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        mask_token_id=103,
        device=None,
        max_vram_bytes=None,
    ):
        self.model_name = model_name
        self.mask_token_id = mask_token_id
        self.device = device
        self.max_vram_bytes = max_vram_bytes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        device = self.device if self.device is not None else auto_device()
        model = load_hf_model(AutoModelForMaskedLM, self.model_name, device)
        model.to(device)
        model.eval()
        model_dtype = str(next(model.parameters()).dtype)

        seq_len = X[0][0][0].shape[0] if X else 512
        chunk_size = _compute_chunk_size(model, seq_len, self.max_vram_bytes)

        results = []
        for sample in track(X):
            # Stack all language token arrays into (n_langs, seq_len) for batched inference.
            all_ids = np.vstack([ids for ids, _ in sample])  # (n_langs, seq_len)
            all_masks = np.vstack([mask for _, mask in sample])  # (n_langs, seq_len)

            token_arrays = get_sample_likelihoods(
                model,
                self.model_name,
                all_ids,
                all_masks,
                self.mask_token_id,
                chunk_size=chunk_size,
                model_dtype=model_dtype,
            )

            max_len = max(t.shape[0] for t in token_arrays)
            results.append(
                torch.vstack(
                    [F.pad(t, (0, max_len - t.shape[0])) for t in token_arrays]
                ).numpy()
            )

        return results
