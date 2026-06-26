from sklearn.base import BaseEstimator, TransformerMixin

import torch
from torch import nn

from transformers import AutoModel

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


@memory.cache(ignore=["model"])
def get_token_embeddings(
    model: nn.Module, model_name: str, input_ids, attention_mask, layer
) -> torch.Tensor:
    """
    Compute the likelihood (probability) of each non-special token in `text`
    by masking each token and querying the model for its probability.

    Uses a chunked approach so it fits in ~8GB GPU memory more easily.

    Args:
        model: HuggingFace-like model (e.g. BERT)
        input_ids, attention_mask
    """
    device = model.device

    # joblib cant currently deterministically hash tensors (they contain metadata about storage)
    input_ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.from_numpy(attention_mask).to(device)

    with torch.no_grad():
        hidden_state = model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        ).hidden_states[layer]
    return hidden_state  # shape: [batch_size, seq_length, hidden_dim]


class EmbedTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, model_name="bert-base-multilingual-cased", layer=12, device=None
    ):
        self.model_name = model_name
        self.layer = layer
        self.device = device

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        device = self.device if self.device is not None else _auto_device()
        model = AutoModel.from_pretrained(self.model_name)
        model.to(device)
        model.eval()

        results = []
        for sample in track(X):
            token_arrays = []
            max_len = 0
            for input_ids, attention_mask in sample:
                last_hidden_state = get_token_embeddings(
                    model, self.model_name, input_ids, attention_mask, self.layer
                ).cpu()

                token_arrays.append(last_hidden_state)
                max_len = max(max_len, last_hidden_state.shape[1])

            results.append(
                torch.vstack(
                    [
                        torch.nn.functional.pad(t, (0, 0, 0, max_len - t.shape[1]))
                        for t in token_arrays
                    ]
                )
                .detach()
                .numpy()
            )

        return results
