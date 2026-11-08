"""Verify first-forward capture and dev-only statistical operations using synthetic data."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend import frozen_scores
from training.dev_frozen_auc import bootstrap_auc, dev_fusion


def test_tokenizer_first_ids_must_differ():
    tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1], decode=lambda *a, **kw: "x")
    with pytest.raises(AssertionError, match="must differ"):
        frozen_scores.label_token_ids(tokenizer)


def test_score_uses_last_position_and_full_vocab():
    logits = torch.tensor([[[9., 0., 0., 0.], [0., 2., 5., 10.]]])
    result = frozen_scores.first_token_score(logits, {"Normal": 1, "Pneumonia": 2})
    expected = torch.log_softmax(logits[0, -1], dim=0)
    assert result["medgemma_score"] == pytest.approx(3.)
    assert result["normal_first_token_logprob"] == pytest.approx(expected[1].item())
    assert result["raw_argmax_token_id"] == 3


def test_hook_captures_first_forward_only_and_is_removed(monkeypatch):
    class Model(torch.nn.Module):
        def forward(self, scores):
            return SimpleNamespace(logits=torch.tensor([[scores]], dtype=torch.float32))

    model = Model()

    def ask(medgemma, image, prompt):
        assert prompt == frozen_scores.PROMPT
        model([0., 2., 5.])
        model([0., 50., 1.])  # Later decoding must not replace first-position evidence.
        return "Pneumonia"

    monkeypatch.setattr(frozen_scores, "ask_medgemma", ask)
    processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a, **kw: "Pneumonia"))
    result = frozen_scores.medgemma_score_and_answer(
        {"model": model, "processor": processor}, None,
        {"first_token_ids": {"Normal": 1, "Pneumonia": 2}})
    assert result["medgemma_score"] == pytest.approx(3.)
    assert result["medgemma_greedy_answer"] == "Pneumonia"
    assert not model._forward_hooks


def test_dev_scaling_and_bootstrap_are_reproducible():
    siglip = np.array([-3., -1., 1., 3.])
    medgemma = np.array([-6., -2., 2., 6.])
    scores, scaling = dev_fusion(siglip, medgemma)
    assert scaling["means"] == [0., 0.]
    assert np.allclose(scores[:, 2], siglip / siglip.std())
    labels = np.array([0, 0, 1, 1])
    a = bootstrap_auc(labels, scores, resamples=20)
    assert np.array_equal(a, bootstrap_auc(labels, scores, resamples=20))
    assert np.all(a == 1)


def test_nonfinite_scores_are_not_dropped():
    with pytest.raises(ValueError, match="Non-finite"):
        dev_fusion([0., float("nan")], [1., 2.])
