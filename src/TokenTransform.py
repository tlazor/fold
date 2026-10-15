from functools import lru_cache

from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer

_MAX_TOKENS = 64


@lru_cache(maxsize=4)
def _get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)


class TokenTransform(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="bert-base-multilingual-cased"):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Tokenize every sentence of X (rows: parallel sentences, columns: languages).

        Returns one list per row holding an (input_ids, attention_mask) pair per
        language, each of shape (1, _MAX_TOKENS). Each language column is
        tokenized in one batched call, which is several times faster than one
        call per sentence and needs no cache.
        """
        tokenizer = _get_tokenizer(self.model_name)
        columns = [
            tokenizer(
                X.iloc[:, j].tolist(),
                max_length=_MAX_TOKENS,
                padding="max_length",
                truncation=True,
                return_tensors="np",
                add_special_tokens=True,
            )
            for j in range(X.shape[1])
        ]
        return [
            [
                (enc["input_ids"][i : i + 1], enc["attention_mask"][i : i + 1])
                for enc in columns
            ]
            for i in range(X.shape[0])
        ]
