import numpy as np
import torch
from torch import nn
from joblib import Memory
from rich.progress import track
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoModel

from paths import CACHE_DIR, auto_device, load_hf_model

memory = Memory(CACHE_DIR / "joblib", verbose=0)


@memory.cache(ignore=["model"])
def get_sample_embeddings(
    model: nn.Module,
    model_name: str,
    all_input_ids,        # numpy (n_langs, seq_len)
    all_attention_masks,  # numpy (n_langs, seq_len)
    layer: int,
) -> torch.Tensor:
    """Hidden-state embeddings for all languages in one sample, one batched pass."""
    device = next(model.parameters()).device
    ids_t = torch.from_numpy(all_input_ids).to(device)
    masks_t = torch.from_numpy(all_attention_masks).to(device)

    with torch.no_grad():
        hidden = model(
            ids_t,
            attention_mask=masks_t,
            output_hidden_states=True,
        ).hidden_states[layer]  # (n_langs, seq_len, hidden_dim)

    return hidden.cpu().float()  # float32 for numpy compatibility


class EmbedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="bert-base-multilingual-cased", layer=12, device=None):
        self.model_name = model_name
        self.layer = layer
        self.device = device

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        device = self.device if self.device is not None else auto_device()
        model = load_hf_model(AutoModel, self.model_name, device)
        model.to(device)
        model.eval()

        results = []
        for sample in track(X):
            all_ids = np.vstack([ids for ids, _ in sample])     # (n_langs, seq_len)
            all_masks = np.vstack([mask for _, mask in sample])  # (n_langs, seq_len)

            hidden = get_sample_embeddings(
                model, self.model_name, all_ids, all_masks, self.layer
            )  # (n_langs, seq_len, hidden_dim) float32

            results.append(hidden.numpy())

        return results
