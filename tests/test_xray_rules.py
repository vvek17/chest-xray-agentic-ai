"""
Fast tests of the upload rules (no model needed).

Run from the chest-xray-pneumonia folder:
    python -m pytest tests/test_xray_rules.py
"""

import io
from pathlib import Path

import pytest

from backend import config
from backend import xray_check
from backend.xray_check import ACCEPTED, NEEDS_REVIEW, REJECTED, decide, medgemma_label

# Words that must never appear in what the user sees about a skipped or reviewed image
UNSUPPORTED = [name for name in config.CLASS_NAMES if name not in config.SUPPORTED_LABELS]
FORBIDDEN_WORDS = [name.lower() for name in UNSUPPORTED] + ["covid", "tuberculosis", " tb ", "(tb)"]

ALL_MESSAGES = [xray_check.MESSAGE_NOT_XRAY, xray_check.MESSAGE_OUT_OF_SCOPE,
                xray_check.MESSAGE_LOW_CONFIDENCE, xray_check.MESSAGE_INVALID_FILE,
                xray_check.DISCLAIMER]


def four_probs(covid=0.0, normal=0.0, pneumonia=0.0, tb=0.0):
    return {"Covid-19": covid, "Normal": normal, "Pneumonia": pneumonia, "Tuberculosis": tb}


def two_probs(normal, pneumonia):
    return {"Normal": normal, "Pneumonia": pneumonia}


GOOD_XRAY = config.XRAY_SIMILARITY_LIMIT + 0.2


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
def test_only_two_supported_labels():
    assert config.SUPPORTED_LABELS == ["Normal", "Pneumonia"]


def test_limits_are_sensible_numbers():
    assert 0 < config.XRAY_SIMILARITY_LIMIT < 1
    assert 0 < config.OUT_OF_SCOPE_LIMIT < 1
    assert 0.5 <= config.CONFIDENCE_LIMIT < 1


# ------------------------------------------------------------
# Check 1: not a chest X-ray
# ------------------------------------------------------------
def test_not_an_xray_is_rejected():
    result = decide(config.XRAY_SIMILARITY_LIMIT - 0.01, four_probs(normal=1.0), two_probs(1.0, 0.0))
    assert result["status"] == REJECTED
    assert result["reason"] == "not_chest_xray"
    assert "label" not in result


# ------------------------------------------------------------
# Check 2: outside the supported labels
# ------------------------------------------------------------
@pytest.mark.parametrize("four", [
    four_probs(covid=0.9, normal=0.1),                 # top class unsupported
    four_probs(tb=0.6, pneumonia=0.4),                 # top class unsupported
    four_probs(covid=0.15, tb=0.10, normal=0.75),      # top supported, but 25% unsupported
])
def test_outside_the_labels_needs_review(four):
    result = decide(GOOD_XRAY, four, two_probs(0.99, 0.01))
    assert result["status"] == NEEDS_REVIEW
    assert result["reason"] == "outside_supported_labels"
    assert "label" not in result


def test_out_of_scope_limit_is_inclusive():
    exactly = config.OUT_OF_SCOPE_LIMIT
    result = decide(GOOD_XRAY, four_probs(covid=exactly, normal=1 - exactly), two_probs(0.99, 0.01))
    assert result["status"] == NEEDS_REVIEW


# ------------------------------------------------------------
# Check 3: low confidence
# ------------------------------------------------------------
def test_low_confidence_needs_review_and_gives_no_label():
    low = config.CONFIDENCE_LIMIT - 0.01
    result = decide(GOOD_XRAY, four_probs(normal=1.0), two_probs(low, 1 - low))
    assert result["status"] == NEEDS_REVIEW
    assert result["reason"] == "low_confidence"
    assert "label" not in result               # never force a prediction


def test_confidence_exactly_at_the_limit_is_accepted():
    limit = config.CONFIDENCE_LIMIT
    result = decide(GOOD_XRAY, four_probs(pneumonia=1.0), two_probs(1 - limit, limit))
    assert result["status"] == ACCEPTED


# ------------------------------------------------------------
# Accepted
# ------------------------------------------------------------
@pytest.mark.parametrize("label, two", [
    ("Normal", two_probs(0.95, 0.05)),
    ("Pneumonia", two_probs(0.03, 0.97)),
])
def test_confident_supported_xray_is_accepted(label, two):
    result = decide(GOOD_XRAY, four_probs(normal=two["Normal"], pneumonia=two["Pneumonia"]), two)
    assert result["status"] == ACCEPTED
    assert result["label"] == label
    assert set(result["probabilities"]) == set(config.SUPPORTED_LABELS)
    assert xray_check.DISCLAIMER in result["message"]


# ------------------------------------------------------------
# The limits really come from config (not hardcoded)
# ------------------------------------------------------------
def test_changing_the_confidence_limit_changes_the_result(monkeypatch):
    four, two = four_probs(normal=1.0), two_probs(0.85, 0.15)
    monkeypatch.setattr(config, "CONFIDENCE_LIMIT", 0.80)
    assert decide(GOOD_XRAY, four, two)["status"] == ACCEPTED
    monkeypatch.setattr(config, "CONFIDENCE_LIMIT", 0.90)
    assert decide(GOOD_XRAY, four, two)["status"] == NEEDS_REVIEW


def test_changing_the_out_of_scope_limit_changes_the_result(monkeypatch):
    four, two = four_probs(covid=0.15, normal=0.85), two_probs(0.99, 0.01)
    monkeypatch.setattr(config, "OUT_OF_SCOPE_LIMIT", 0.20)
    assert decide(GOOD_XRAY, four, two)["status"] == ACCEPTED
    monkeypatch.setattr(config, "OUT_OF_SCOPE_LIMIT", 0.10)
    assert decide(GOOD_XRAY, four, two)["status"] == NEEDS_REVIEW


def test_changing_the_xray_limit_changes_the_result(monkeypatch):
    four, two = four_probs(normal=1.0), two_probs(0.99, 0.01)
    monkeypatch.setattr(config, "XRAY_SIMILARITY_LIMIT", 0.50)
    assert decide(0.55, four, two)["status"] == ACCEPTED
    monkeypatch.setattr(config, "XRAY_SIMILARITY_LIMIT", 0.60)
    assert decide(0.55, four, two)["status"] == REJECTED


# ------------------------------------------------------------
# Messages never mention other diseases
# ------------------------------------------------------------
@pytest.mark.parametrize("message", ALL_MESSAGES)
def test_messages_do_not_mention_other_diseases(message):
    text = f" {message.lower()} "
    for word in FORBIDDEN_WORDS:
        assert word not in text


def test_every_result_only_mentions_supported_labels():
    cases = [
        decide(0.10, four_probs(normal=1.0), two_probs(1.0, 0.0)),
        decide(GOOD_XRAY, four_probs(covid=1.0), two_probs(0.99, 0.01)),
        decide(GOOD_XRAY, four_probs(tb=1.0), two_probs(0.01, 0.99)),
        decide(GOOD_XRAY, four_probs(normal=1.0), two_probs(0.6, 0.4)),
        decide(GOOD_XRAY, four_probs(normal=1.0), two_probs(0.99, 0.01)),
    ]
    for result in cases:
        text = f" {str(result).lower()} "
        for word in FORBIDDEN_WORDS:
            assert word not in text, f"{word!r} found in {result}"
        if "label" in result:
            assert result["label"] in config.SUPPORTED_LABELS


# ------------------------------------------------------------
# Invalid files
# ------------------------------------------------------------
@pytest.mark.parametrize("data", [b"", b"this is not an image", b"\x89PNG\r\n\x1a\n broken"])
def test_invalid_files_are_rejected(data):
    result, image = xray_check.analyze_upload(checker=None, uploaded_file=io.BytesIO(data))
    assert result["status"] == REJECTED
    assert result["reason"] == "invalid_file"
    assert image is None


# ------------------------------------------------------------
# MedGemma answers: only the 2 labels are ever shown
# ------------------------------------------------------------
@pytest.mark.parametrize("answer, expected", [
    ("Normal", "Normal"),
    ("Pneumonia.", "Pneumonia"),
    ("The lungs are clear.\nAnswer: Normal", "Normal"),
    ("Covid-19", None),
    ("Tuberculosis", None),
    ("I cannot tell", None),
    ("Normal or Pneumonia", None),                  # both: not clear -> not shown
])
def test_medgemma_answers_are_limited_to_supported_labels(answer, expected):
    assert medgemma_label(answer) == expected
