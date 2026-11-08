import json

import numpy as np
import torch

from training import run_new_model_baselines as run


def test_manual_runner_trains_ten_epochs_freezes_selection_and_tests_once(tmp_path, monkeypatch):
    out, features, models = (tmp_path / name for name in ("results", "features", "models"))
    for path in (out, features, models):
        path.mkdir()
    monkeypatch.setattr(run, "OUT", out)
    monkeypatch.setattr(run, "FEATURES", features)
    monkeypatch.setattr(run, "MODELS", models)
    # Restore common-module settings after this integration test.
    for name in ("OUT", "MAX_EPOCHS", "PATIENCE", "LEARNING_RATES", "WEIGHT_DECAYS"):
        monkeypatch.setattr(run.common, name, getattr(run.common, name))
    classes = ["Covid-19", "Normal", "Pneumonia", "Tuberculosis"]
    spec = {"classes": classes, "manifest_sha256": {"test": "synthetic-test-only"}}
    (features / "extractor.json").write_text("{}")
    generator = torch.Generator().manual_seed(42)
    parts = {}
    for split, repeats in (("train", 8), ("val", 4), ("test", 4)):
        y = torch.arange(4).repeat_interleave(repeats)
        x = torch.randn(len(y), 1156, generator=generator) * .01
        x[torch.arange(len(y)), y] += 1
        x[:, :1152] = torch.nn.functional.normalize(x[:, :1152], dim=1)
        raw = np.full((len(y), 4), -5.)
        raw[np.arange(len(y)), y.numpy()] = -.01
        cases = [{"image_id": f"synthetic-{split}-{i}", "path": "synthetic-only",
                  "true_label": classes[int(label)]} for i, label in enumerate(y)]
        parts[split] = (x, y, raw, cases)
    texts = np.zeros((4, 1152), dtype=np.float32)
    texts[np.arange(4), np.arange(4)] = 1
    np.savez(features / "zero_shot_texts.npz", embeddings=texts)
    monkeypatch.setattr(run, "load_split", lambda split, labels: parts[split])
    test_feature_calls = []
    def prepare(specification, splits):
        assert (out / "selection.json").exists(), "Test opened before selection freeze"
        test_feature_calls.append(splits)
    monkeypatch.setattr(run, "prepare_features", prepare)
    # Bootstrap implementation was checked separately; keep this state-flow test short.
    monkeypatch.setattr(run.common, "bootstrap_test", lambda *args: {"macro_f1": [.8, 1.]})
    run.train(spec)
    selection = json.loads((out / "selection.json").read_text())
    assert selection["epochs_run"] == 10
    assert 1 <= selection["linear_probe_best_epoch"] <= 10
    trials = json.loads((out / "linear_probe/training_trials.json").read_text())
    assert len(trials) == 1 and len(trials[0]["history"]) == 10
    assert not (out / "test_metrics.json").exists()
    original_hash = run.common.digest(models / "linear_probe.pth")
    run.test_once(spec)
    assert test_feature_calls == [("test",)]
    report = json.loads((out / "test_metrics.json").read_text())
    assert set(report) == set(run.METHODS)
    assert all(row["samples"] == 16 for row in report.values())
    run.test_once(spec)
    assert test_feature_calls == [("test",)]
    assert run.common.digest(models / "linear_probe.pth") == original_hash


def test_zero_shot_uses_both_frozen_components_without_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "FEATURES", tmp_path)
    texts = np.zeros((4, 1152), dtype=np.float32)
    texts[np.arange(4), np.arange(4)] = 1
    np.savez(tmp_path / "zero_shot_texts.npz", embeddings=texts)
    features = torch.zeros(1, 1156)
    features[0, 0] = 1
    raw = np.array([[-5., 0., -5., -5.]])
    probabilities = run.zero_shot_probabilities(features, raw)
    expected = (run.softmax(np.array([[1., 0., 0., 0.]])) + run.softmax(raw)) / 2
    np.testing.assert_allclose(probabilities, expected)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_familiar_train_probe_entrypoint_starts_only_new_model_workflow(monkeypatch):
    from training.train_probe import main_new_model
    calls = []
    monkeypatch.setattr(run, "main", lambda arguments: calls.append(arguments))
    main_new_model()
    assert calls == [["--stage", "all"]]
