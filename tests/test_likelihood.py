import numpy as np
import pytest
import torch
from transformers import (
    BertConfig,
    BertForMaskedLM,
    XLMRobertaConfig,
    XLMRobertaForMaskedLM,
)

from LikelihoodEstimator import _mlm_head, get_masked_token_probs

MASK_ID = 4
_SIZES = dict(
    vocab_size=40,
    hidden_size=16,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=32,
    max_position_embeddings=40,
)


def _tiny_model(arch):
    torch.manual_seed(0)
    if arch == "bert":
        return BertForMaskedLM(BertConfig(pad_token_id=0, **_SIZES)).eval()
    # XLM-R derives position ids from the padding, so check it separately.
    return XLMRobertaForMaskedLM(XLMRobertaConfig(pad_token_id=1, **_SIZES)).eval()


def _right_padded_batch(lengths, seq_len, pad_id):
    rng = np.random.default_rng(0)
    ids = np.full((len(lengths), seq_len), pad_id, dtype=np.int64)
    masks = np.zeros((len(lengths), seq_len), dtype=np.int64)
    for i, n in enumerate(lengths):
        ids[i, :n] = rng.integers(5, _SIZES["vocab_size"], n)
        ids[i, 0], ids[i, n - 1] = 2, 3  # [CLS] ... [SEP]
        masks[i, :n] = 1
    return ids, masks


def _reference(model, ids, masks):
    """The original method: full logits on the padded input, one mask at a time."""
    probs = []
    with torch.no_grad():
        for row, mask in zip(ids, masks):
            lang = []
            for pos in range(1, int(mask.sum()) - 1):
                masked = torch.from_numpy(row.copy())[None]
                masked[0, pos] = MASK_ID
                logits = model(
                    masked, attention_mask=torch.from_numpy(mask)[None]
                ).logits
                lang.append(logits[0, pos].softmax(-1)[row[pos]].item())
            probs.append(np.array(lang))
    return probs


@pytest.mark.parametrize("arch", ["bert", "xlmr"])
def test_matches_full_logits_on_padded_input(arch):
    model = _tiny_model(arch)
    # Two languages share a length, so they share forward passes.
    ids, masks = _right_padded_batch(
        [12, 8, 8, 3], seq_len=14, pad_id=model.config.pad_token_id
    )

    probs = get_masked_token_probs.func(
        model.base_model, _mlm_head(model), arch, ids, masks, MASK_ID
    )

    for got, expected in zip(probs, _reference(model, ids, masks)):
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5, atol=1e-7)


def test_sentence_without_normal_tokens_gives_empty_array():
    model = _tiny_model("bert")
    ids, masks = _right_padded_batch([2, 6], seq_len=8, pad_id=0)
    probs = get_masked_token_probs.func(
        model.base_model, _mlm_head(model), "bert", ids, masks, MASK_ID
    )
    assert [len(p) for p in probs] == [0, 4]
