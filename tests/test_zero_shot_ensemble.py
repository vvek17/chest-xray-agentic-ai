"""Checks for the new experiment's answer parsing and evaluation accounting."""

import pytest

from backend.zero_shot_ensemble import parse_label
from backend import zero_shot_ensemble
from prompt_engineering.run_zero_shot_ensemble import evaluate_rows


@pytest.mark.parametrize("answer, expected", [
    ("Normal", "Normal"), ("Answer: Pneumonia.", "Pneumonia"),
    ("No pneumonia. Normal.", "Other"), ("Not normal", "Other"),
    ("Normal or Pneumonia", "Other"), ("", "Other"),
    ("Clear lungs.\nFinal answer: Normal", "Normal"),
    ("An opacity is present.\nFinal answer: Pneumonia", "Pneumonia"),
    ("Final answer: Normal\nFinal answer: Pneumonia", "Pneumonia"),
    ("Pneumonia is possible, but no final label.", "Other"),
])
def test_strict_label_parsing(answer, expected):
    assert parse_label(answer) == expected


def test_invalid_answers_remain_errors_in_metrics():
    metrics = evaluate_rows([
        {"true": "Normal", "predicted": "Normal"},
        {"true": "Pneumonia", "predicted": "Other"},
    ])
    assert metrics["accuracy"] == 0.5
    assert metrics["invalid_answers"] == 1
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 0, 0]]


@pytest.mark.parametrize("version", ["criteria_nohint", "criteria_hint",
                                     "criteria_nohint_softened", "criteria_hint_softened"])
def test_criteria_prompts_format_and_parse(monkeypatch, version):
    def ask(model, image, prompt):
        assert prompt["max_tokens"] == 100
        assert "{label}" not in prompt["user"]
        assert ("leaned toward Pneumonia" in prompt["user"]) == ("criteria_hint" in version)
        if version.endswith("_softened"):
            assert "If you cannot, answer Normal" not in prompt["user"]
        return "A focal opacity is visible.\nFinal answer: Pneumonia"

    monkeypatch.setattr(zero_shot_ensemble, "ask_medgemma", ask)
    raw, label = zero_shot_ensemble.ensemble_answer(
        None, None, {"Normal": 0.1, "Pneumonia": 0.2}, version)
    assert label == "Pneumonia"
