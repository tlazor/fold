from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel

from rich.progress import track

import fold_globals

from joblib import Memory

cachedir = Path(".cache/joblib")
memory = Memory(cachedir, verbose=0)


@memory.cache(ignore=["model"])
@torch.compile
def get_token_embeddings(model: nn.Module, input_ids, attention_mask) -> torch.Tensor:
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
        hidden_state = model(input_ids, attention_mask=attention_mask).last_hidden_state
    return hidden_state  # shape: [batch_size, seq_length, hidden_dim]


class EmbedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="bert-base-multilingual-cased", mask_token_id=103):
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
        model = AutoModel.from_pretrained(self.model_name)

        model.to(fold_globals.DEVICE)
        model.eval()

        results = []
        for sample in track(X):
            token_arrays = []
            max_len = 0
            for input_ids, attention_mask in sample:
                # print(f"{input_ids=}")
                # print(f"{attention_mask=}")
                # exit()
                last_hidden_state = get_token_embeddings(
                    model, input_ids, attention_mask
                ).cpu()

                token_arrays.append(last_hidden_state)

                max_len = max(max_len, last_hidden_state.shape[1])

            results.append(
                torch.vstack(
                    [
                        torch.nn.functional.pad(t, (0, max_len - t.shape[0]))
                        for t in token_arrays
                    ]
                )
                .detach()
                .numpy()
            )

        return results
