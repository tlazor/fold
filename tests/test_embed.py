import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel

import EmbedTransformer as embed_module
from EmbedTransformer import EmbedTransformer


@pytest.fixture
def tiny_model_dir(tmp_path, monkeypatch):
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=30,
        hidden_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=16,
    )
    BertModel(config).save_pretrained(tmp_path)
    # Don't write test results into the project's joblib cache.
    monkeypatch.setattr(
        embed_module,
        "get_sample_hidden_states",
        embed_module.get_sample_hidden_states.func,
    )
    return str(tmp_path)


def _samples(n_samples=2, n_langs=3, seq_len=6):
    rng = np.random.default_rng(0)
    return [
        [
            (rng.integers(5, 30, (1, seq_len)), np.ones((1, seq_len), dtype=np.int64))
            for _ in range(n_langs)
        ]
        for _ in range(n_samples)
    ]


def test_returns_requested_layer(tiny_model_dir):
    samples = _samples()
    model = BertModel.from_pretrained(tiny_model_dir).eval()
    for layer in (1, 3):
        result = EmbedTransformer(
            model_name=tiny_model_dir,
            layer=layer,
            layers=(1, 3),
            device=torch.device("cpu"),
        ).transform(samples)
        for sample, got in zip(samples, result):
            ids = torch.from_numpy(np.vstack([i for i, _ in sample]))
            with torch.no_grad():
                expected = model(ids, output_hidden_states=True).hidden_states[layer]
            assert got.dtype == np.float32 and got.shape == (3, 6, 8)
            np.testing.assert_allclose(got, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_layers_default_to_layer(tiny_model_dir):
    a = EmbedTransformer(model_name=tiny_model_dir, layer=2, device=torch.device("cpu"))
    b = EmbedTransformer(
        model_name=tiny_model_dir, layer=2, layers=(0, 2), device=torch.device("cpu")
    )
    samples = _samples(n_samples=1)
    np.testing.assert_allclose(a.transform(samples)[0], b.transform(samples)[0])


def test_layer_must_be_in_layers(tiny_model_dir):
    with pytest.raises(ValueError):
        EmbedTransformer(model_name=tiny_model_dir, layer=2, layers=(1, 3)).transform(
            _samples(n_samples=1)
        )
