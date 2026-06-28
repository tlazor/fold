from functools import lru_cache

from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer

from paths import CACHE_DIR

memory = Memory(CACHE_DIR / "joblib", verbose=0)


@lru_cache(maxsize=4)
def _get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)


@memory.cache
def _tokenize(model_name: str, text: str, max_tokens: int = 64):
    tokenizer = _get_tokenizer(model_name)
    encoded = tokenizer(
        text,
        max_length=max_tokens,
        padding="max_length",
        truncation=True,
        return_tensors="np",
        add_special_tokens=True,
    )
    return encoded["input_ids"], encoded["attention_mask"]


class TokenTransform(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="bert-base-multilingual-cased"):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_langs, n_features = X.shape

        results = []
        for i in range(n_langs):
            token_arrays = []
            for j in range(n_features):
                text = X.iloc[i, j]
                token_arrays.append(_tokenize(self.model_name, text))
            results.append(token_arrays)

        return results
