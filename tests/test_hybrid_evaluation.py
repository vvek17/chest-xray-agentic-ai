"""Test leakage boundaries and metrics using tiny files, without model downloads."""

import csv
import json
from pathlib import Path

import pytest

from prompt_engineering import evaluate_hybrid as hybrid
from prompt_engineering.photos import get_examples


# A raw prediction is valid only when it is exactly one permitted label.
@pytest.mark.parametrize("answer, expected", [
    ("Normal", "Normal"), ("Pneumonia\n", "Pneumonia"),
    ("No pneumonia; normal", "Other"), ("Answer: Normal", "Other"),
    ("Normal or Pneumonia", "Other"), ("", "Other"),
])
def test_exact_output(answer, expected):
    assert hybrid.strict_answer(answer) == expected


# The binary hint must ignore higher scores for the two unsupported diseases.
def test_hint_uses_four_class_probe_but_compares_two_labels(monkeypatch):
    scores = {"Covid-19": .99, "Normal": .4, "Pneumonia": .3, "Tuberculosis": .8}
    monkeypatch.setattr(hybrid, "predict", lambda model, image: scores)
    hint = hybrid.get_hint("four-class-model", "image.png")
    assert hint["label"] == "Normal"
    assert hint["score"] == .4
    assert hint["scores"] == scores


# Invalid answers must count as broken correct hints rather than disappear.
def test_hint_fix_break_denominators():
    rows = [
        {"true": "Normal", "hint_label": "Pneumonia", "predicted": "Normal"},
        {"true": "Normal", "hint_label": "Normal", "predicted": "Other"},
        {"true": "Pneumonia", "hint_label": "Pneumonia", "predicted": "Pneumonia"},
        {"true": "Pneumonia", "hint_label": "Normal", "predicted": "Normal"},
    ]
    result = hybrid.hint_metrics(rows)
    assert result["hint_follow_rate"] == .5
    assert result["medsiglip_errors_fixed"] == 1
    assert result["correct_hints_broken"] == 1
    assert result["error_fix_rate"] == .5
    assert result["correct_hint_break_rate"] == .5
    assert hybrid.hint_metrics([])["error_fix_rate"] is None


# Hints are formatted only into the target, never copied into context questions.
def test_hybrid_context_has_no_target_hint():
    prompt = hybrid.target_prompt("hybrid_label_v1", {"label": "Normal", "score": .8})
    assert "suggests Normal" in prompt["user"]
    assert "suggests" not in prompt["example_user"]
    assert prompt["example_user"] == hybrid.PROMPTS["2c_fs1"]["user"]


# New examples use the caller's screened pool, preserving the original helper.
def test_examples_from_screened_pool():
    pool = [(Path("normal.png"), "Normal"), (Path("pneumonia.png"), "Pneumonia")]
    assert get_examples(hybrid.CLASSES, 1, photos=pool) == pool


# Build all cases from a tiny original Testing pool, excluding history and fitted copies.
def make_manifest(tmp_path, monkeypatch):
    source = tmp_path / "Testing"
    for label in hybrid.CLASSES:
        folder = source / label
        folder.mkdir(parents=True)
        for number in range(7):
            (folder / f"{number}.png").write_bytes(f"{label}-{number}".encode())
    old = hybrid.file_hash(source / "Normal/0.png")
    fitted = hybrid.file_hash(source / "Pneumonia/0.png")
    monkeypatch.setattr(hybrid, "SOURCE", source)
    monkeypatch.setattr(hybrid, "SIZES", {"dev": 1, "val": 1, "test": 1})
    monkeypatch.setattr(hybrid, "history_hashes", lambda: ({old}, []))
    monkeypatch.setattr(hybrid, "fitted_data_hashes", lambda: {fitted})
    monkeypatch.setattr(hybrid, "experiment_spec", lambda: {"test": "configuration"})
    return hybrid.create_manifest()


# SHA checks reject both duplicates and changes, independent of filenames.
def test_manifest_is_disjoint_and_excludes_history(tmp_path, monkeypatch):
    manifest = make_manifest(tmp_path, monkeypatch)
    hybrid.verify_manifest(manifest)
    cases = list(manifest["examples"])
    for part in manifest["splits"].values():
        cases.extend(part)
    assert len(cases) == 8
    assert len({case["sha256"] for case in cases}) == 8
    manifest["splits"]["test"][0] = manifest["splits"]["dev"][0]
    with pytest.raises(RuntimeError, match="duplicate"):
        hybrid.verify_manifest(manifest)


# A changed file must stop the run before any model is loaded.
def test_changed_image_is_rejected(tmp_path, monkeypatch):
    manifest = make_manifest(tmp_path, monkeypatch)
    Path(manifest["splits"]["test"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Changed"):
        hybrid.verify_manifest(manifest)


# Do not silently shrink a split when too few eligible images remain.
def test_insufficient_pool_fails(tmp_path, monkeypatch):
    make_manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(hybrid, "SIZES", {"dev": 20, "val": 40, "test": 70})
    with pytest.raises(RuntimeError, match="Not enough"):
        hybrid.create_manifest()


# An exact repeated image under a new filename must still be excluded.
def test_renamed_duplicate_is_excluded(tmp_path, monkeypatch):
    source = tmp_path / "Testing"
    for label in hybrid.CLASSES:
        (source / label).mkdir(parents=True)
        (source / label / "copy.png").write_bytes(label.encode())
    monkeypatch.setattr(hybrid, "SOURCE", source)
    blocked = {hybrid.file_hash(source / "Normal/copy.png")}
    assert hybrid.eligible_images(blocked) == [(source / "Pneumonia/copy.png", "Pneumonia")]


# Model/test inference cannot run before the explicit freeze stage.
def test_test_requires_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid, "OUT", tmp_path)
    with pytest.raises(RuntimeError, match="freeze"):
        hybrid.require_frozen({})


# Resume checks reject rows that do not match the selected cases and arm order.
def test_resume_rejects_wrong_case(tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid, "OUT", tmp_path)
    manifest = {"splits": {"dev": [{"path": "expected.png", "sha256": "123", "true": "Normal"}]}}
    with (tmp_path / "dev_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=hybrid.FIELDS)
        writer.writeheader()
        writer.writerow({"path": "wrong.png", "arm": "medsiglip"})
    with pytest.raises(RuntimeError, match="manifest"):
        hybrid.checked_rows("dev", manifest)


# Exercise all six arms and resume using fake models, so the real test stays untouched.
def test_all_arms_share_cases_and_resume(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(hybrid, "OUT", tmp_path)
    cases = [
        {"path": "normal.png", "sha256": "n", "true": "Normal"},
        {"path": "pneumonia.png", "sha256": "p", "true": "Pneumonia"},
    ]
    manifest = {"splits": {"dev": cases}, "examples": [
        {"path": "example-normal.png", "true": "Normal"},
        {"path": "example-pneumonia.png", "true": "Pneumonia"},
    ], "spec": {"medsiglip_revision": "sig-revision", "medgemma_revision": "gem-revision"}}
    calls = []
    monkeypatch.setattr(hybrid, "load_classifier", lambda *a, **kw: "probe")
    monkeypatch.setattr(hybrid, "load_medgemma", lambda **kw: {"model": torch.nn.Linear(1, 1)})
    monkeypatch.setattr(hybrid, "clean_xray", lambda path: path)
    monkeypatch.setattr(hybrid, "get_hint", lambda model, path: {
        "label": "Normal", "score": .8,
        "scores": {"Normal": .8, "Pneumonia": .2, "Covid-19": .1, "Tuberculosis": .1},
    })

    def ask(model, image, prompt, examples):
        calls.append((image, examples))
        assert "sha256" not in prompt["user"]
        return "Normal"

    monkeypatch.setattr(hybrid, "ask_medgemma", ask)
    rows = hybrid.run_split("dev", manifest)
    assert len(rows) == 12
    assert len(calls) == 10
    assert len(calls[0][1]) == 0
    for call in calls[1:5]:
        assert call[1] == calls[1][1]
        assert len(call[1]) == 2
    summary = hybrid.split_metrics(rows)
    for result in summary.values():
        assert result["n"] == 2
        assert result["accuracy"] == .5
    hybrid.run_split("dev", manifest)
    assert len(calls) == 10


# Freeze chooses validation metrics only, then rejects altered validation predictions.
def test_freeze_and_changed_validation_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid, "OUT", tmp_path)
    manifest = {"spec": {"selection_rule": "validation only"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "dev_predictions.csv").write_text("dev")
    (tmp_path / "val_predictions.csv").write_text("val")
    scores = {}
    for arm in hybrid.HYBRIDS:
        scores[arm] = {"macro_f1": .8, "per_class": {"Pneumonia": {"recall": .8, "specificity": .8}}}
    scores["hybrid_label_v1"]["macro_f1"] = .9
    monkeypatch.setattr(hybrid, "complete_metrics", lambda part, data: scores)
    hybrid.freeze(manifest)
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["selected_hybrid"] == "hybrid_label_v1"
    hybrid.require_frozen(manifest)
    (tmp_path / "val_predictions.csv").write_text("changed")
    with pytest.raises(RuntimeError, match="changed after freezing"):
        hybrid.require_frozen(manifest)


# A changed prompt/configuration cannot silently reuse the saved image selection.
def test_prepare_rejects_changed_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid, "OUT", tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"spec": {"prompt": "old"}}))
    monkeypatch.setattr(hybrid, "experiment_spec", lambda: {"prompt": "new"})
    with pytest.raises(RuntimeError, match="Experiment changed"):
        hybrid.prepare()
