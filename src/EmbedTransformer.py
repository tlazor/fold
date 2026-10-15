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
def get_sample_hidden_states(
    model: nn.Module,
    model_name: str,
    all_input_ids,  # numpy (n_langs, seq_len)
    all_attention_masks,  # numpy (n_langs, seq_len)
    layers: tuple,
) -> torch.Tensor:
    """
    Hidden states of every layer in `layers` for all languages in one sample,
    from one batched forward pass.

    Returns a tensor (len(layers), n_langs, seq_len, hidden_dim) in the model's
    dtype: bfloat16 on CUDA, so the cache holds exactly what the model produced
    at half the size of float32.
    """
    device = next(model.parameters()).device
    ids_t = torch.from_numpy(all_input_ids).to(device)
    masks_t = torch.from_numpy(all_attention_masks).to(device)

    with torch.no_grad():
        hidden_states = model(
            ids_t,
            attention_mask=masks_t,
            output_hidden_states=True,
        ).hidden_states

    return torch.stack([hidden_states[layer] for layer in layers]).cpu()


class EmbedTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        layer=12,
        layers=None,
        device=None,
    ):
        """
        Parameters
        ----------
        layer : int
            The hidden layer this transformer returns.
        layers : tuple of int, optional
            Every layer to extract in the same forward pass and cache together
            (default: just `layer`). Pass all the layers a run needs, so that
            the other layers' transformers read them from the cache.
        """
        self.model_name = model_name
        self.layer = layer
        self.layers = layers
        self.device = device

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        layers = tuple(self.layers) if self.layers is not None else (self.layer,)
        if self.layer not in layers:
            raise ValueError(f"layer {self.layer} is not in layers {layers}")
        index = layers.index(self.layer)

        device = self.device if self.device is not None else auto_device()
        model = load_hf_model(AutoModel, self.model_name, device)
        model.to(device)
        model.eval()

        results = []
        for sample in track(X, description="hidden states"):
            all_ids = np.vstack([ids for ids, _ in sample])  # (n_langs, seq_len)
            all_masks = np.vstack([mask for _, mask in sample])  # (n_langs, seq_len)

            hidden = get_sample_hidden_states(
                model, self.model_name, all_ids, all_masks, layers
            )  # (n_layers, n_langs, seq_len, hidden_dim)

            # float32 for numpy compatibility; one layer in memory at a time
            results.append(hidden[index].float().numpy())

        return results
