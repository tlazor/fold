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

# Masked copies (rows) per forward pass. The float32 logits of one chunk take
# rows x vocab_size x 4 bytes (1 GB for XLM-R); halved automatically on OOM.
_MAX_ROWS = 1024


def _mlm_head(model: nn.Module) -> nn.Module:
    """The masked-LM prediction head (BERT: .cls, RoBERTa / XLM-R: .lm_head)."""
    return model.cls if hasattr(model, "cls") else model.lm_head


def _masked_probs(
    encoder: nn.Module,
    head: nn.Module,
    ids: torch.Tensor,
    positions: torch.Tensor,
    targets: torch.Tensor,
    max_rows: int = _MAX_ROWS,
) -> torch.Tensor:
    """
    Probability of targets[r] at the masked position positions[r] of each row.

    Only the masked position's hidden state goes through the LM head, in
    float32: the head maps one hidden state to vocab_size logits, so running it
    on every position (as model(...).logits does) dominated the cost.
    """
    device = next(encoder.parameters()).device
    probs = []
    start, rows = 0, max_rows
    with torch.no_grad():
        while start < ids.shape[0]:
            end = min(start + rows, ids.shape[0])
            try:
                hidden = encoder(ids[start:end].to(device)).last_hidden_state
                row = torch.arange(end - start, device=device)
                masked_hidden = hidden[row, positions[start:end].to(device)]
                logits = head(masked_hidden.float())  # (chunk, vocab_size)
                nll = F.cross_entropy(
                    logits, targets[start:end].to(device), reduction="none"
                )
                probs.append(torch.exp(-nll).cpu())
                start = end
            except torch.cuda.OutOfMemoryError:
                if rows == 1:
                    raise
                torch.cuda.empty_cache()
                rows = max(1, rows // 2)
    return torch.cat(probs) if probs else torch.empty(0)


@memory.cache(ignore=["encoder", "head"])
def get_masked_token_probs(
    encoder: nn.Module,
    head: nn.Module,
    model_name: str,
    all_token_ids,
    all_attention_masks,
    mask_token_id: int,
    model_dtype: str = "torch.bfloat16",
) -> list:
    """
    Per-token likelihoods for all languages in one sample: the probability of
    each non-special token when it alone is masked.

    all_token_ids      : numpy array (n_langs, seq_len), right-padded
    all_attention_masks: numpy array (n_langs, seq_len)

    Returns a list of n_langs 1-D tensors, each holding the probability of
    every non-special token in that language's sentence.

    Each masked copy is trimmed to its sentence's length (languages of equal
    length share forward passes), so padding is never computed.
    """
    n_langs = all_token_ids.shape[0]
    lengths = all_attention_masks.sum(axis=1)
    probs = [torch.empty(0)] * n_langs

    for length in np.unique(lengths):
        langs = np.flatnonzero(lengths == length)
        n_tokens = int(length) - 2  # exclude [CLS] and [SEP]
        if n_tokens <= 0:
            continue

        # One copy of each sentence per normal token, with that token masked.
        ids = torch.from_numpy(all_token_ids[langs, :length]).repeat_interleave(
            n_tokens, dim=0
        )
        positions = torch.arange(1, n_tokens + 1).repeat(len(langs))
        rows = torch.arange(ids.shape[0])
        targets = ids[rows, positions].clone()
        ids[rows, positions] = mask_token_id

        group_probs = _masked_probs(encoder, head, ids, positions, targets)
        for k, lang in enumerate(langs):
            probs[lang] = group_probs[k * n_tokens : (k + 1) * n_tokens]

    return probs


class LikelihoodEstimator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        mask_token_id=103,
        device=None,
    ):
        self.model_name = model_name
        self.mask_token_id = mask_token_id
        self.device = device

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        device = self.device if self.device is not None else auto_device()
        model = load_hf_model(AutoModelForMaskedLM, self.model_name, device)
        model.to(device)
        model.eval()
        model_dtype = str(next(model.parameters()).dtype)

        # The head only sees one position per row, so it can run in float32.
        # Take its weights from a float32 load rather than up-casting the
        # bfloat16 ones, so the logits keep full precision.
        head = _mlm_head(
            AutoModelForMaskedLM.from_pretrained(self.model_name, dtype=torch.float32)
        )
        head.to(device).eval()

        results = []
        for sample in track(X):
            # Stack all language token arrays into (n_langs, seq_len).
            all_ids = np.vstack([ids for ids, _ in sample])  # (n_langs, seq_len)
            all_masks = np.vstack([mask for _, mask in sample])  # (n_langs, seq_len)

            token_arrays = get_masked_token_probs(
                model.base_model,
                head,
                self.model_name,
                all_ids,
                all_masks,
                self.mask_token_id,
                model_dtype=model_dtype,
            )

            max_len = max(t.shape[0] for t in token_arrays)
            results.append(
                torch.vstack(
                    [F.pad(t, (0, max_len - t.shape[0])) for t in token_arrays]
                ).numpy()
            )

        return results
