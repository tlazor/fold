from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

from transformers import AutoTokenizer


class TokenTransform(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="bert-base-multilingual-cased"):
        self.model_name = model_name

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
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, clean_up_tokenization_spaces=True
        )

        from joblib import Memory

        cachedir = Path(".cache/joblib")
        memory = Memory(cachedir, verbose=0)

        @memory.cache
        def tokenize(model_name: str, text: str, max_tokens: int = 64) -> list:
            encoded = tokenizer(
                text,
                max_length=max_tokens,
                padding="max_length",
                truncation=True,
                return_tensors="np",
                add_special_tokens=True,
            )

            return encoded["input_ids"], encoded["attention_mask"]

        n_langs, n_features = X.shape

        results = []
        for i in range(n_langs):
            # Collect token likelihood arrays for each language in row i
            token_arrays = []
            for j in range(n_features):
                text = X.iloc[i, j]
                token_arrays.append(tokenize(self.model_name, text))

            results.append(token_arrays)

        return results
