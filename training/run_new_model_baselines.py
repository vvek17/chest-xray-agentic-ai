"""Manual experiment for the new frozen MedSigLIP + MedGemma model only.

python -B -u -m training.run_new_model_baselines --stage all
Stages: features (train/val only), train (select on val), test (frozen, once).
Original datasets, old models and the superseded experiment are read-only.
"""

import argparse
import csv
import json
import random
import shutil
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from backend import config
from backend.dual_backbone import FusedHeads
from backend.zero_shot_ensemble import load_zero_shot
from training import train_dual_backbone as common

ROOT = config.PROJECT_ROOT
DATA = ROOT / "data/new_distribution"
FEATURES = ROOT / "outputs/features/medsiglip_medgemma_new_distribution"
OUT = ROOT / "outputs/new_model_baselines"
MODELS = ROOT / "models/new_model_baselines"
OLD = ROOT / "outputs/new_distribution_fusion"
EPOCHS = 10
K_VALUES = (1, 3, 5, 7, 11, 21)
METHODS = ("zero_shot", "knn", "logistic_regression", "linear_probe")


def protocol():
    classes = json.loads((DATA / "manifests/split_summary.json").read_text())["classes"]
    spec = {"classes": classes, "seed": 42, "epochs": EPOCHS, "model": "Frozen MedSigLIP + frozen MedGemma",
            "features": str(FEATURES), "feature_dim": 1156, "test_after_selection_only": True,
            "manifest_sha256": {s: common.digest(DATA / "manifests" / f"{s}.csv") for s in ("train", "val", "test")},
            "source_sha256": {str(p.relative_to(ROOT)): common.digest(p) for p in
                              (Path(__file__), ROOT / "backend/dual_backbone.py", ROOT / "backend/clean_image.py",
                               ROOT / "backend/medgemma.py", ROOT / "backend/zero_shot_ensemble.py",
                               ROOT / "training/train_dual_backbone.py", ROOT / "training/train_probe.py")},
            "zero_shot": "Equal average of softmax(MedSigLIP cosine text similarities) and softmax(MedGemma raw class logprobs); no fitting, temperatures or tuned weights",
            "zero_shot_texts": [config.ZERO_SHOT_PROMPTS[c] for c in classes],
            "knn": {"k_candidates": K_VALUES, "metric": "cosine", "weights": "uniform",
                    "scale": "Train-only mean/std followed by per-image L2 normalization", "ties": "Smaller k"},
            "logistic_regression": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"},
            "linear_probe": {"epochs": EPOCHS, "lr": 0.001, "weight_decay": 0.0001,
                             "heads": "Linear-softmax and Linear-sigmoid, jointly trained, averaged/renormalized"},
            "selection": "Validation macro-F1, ties balanced accuracy then lower validation loss for epoch selection",
            "no_backbone_fine_tuning": True, "no_NIH_or_other_backbones": True}
    OUT.mkdir(parents=True, exist_ok=True)
    FEATURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    return common.freeze_json(OUT / "protocol.json", spec)


def import_previous_cache(spec):
    """Copy compatible partial caches once, preserving both original artifacts."""
    if (FEATURES / "cache_import.json").exists() or not (OLD / "protocol.json").exists():
        return
    previous = json.loads((OLD / "protocol.json").read_text())
    for name, expected in previous["source_sha256"].items():
        assert common.digest(ROOT / name) == expected, f"Old extraction source differs: {name}"
    assert previous["classes"] == spec["classes"]
    for split in ("train", "val"):
        assert previous["manifest_sha256"][split] == spec["manifest_sha256"][split]
    counts = {}
    for split in ("train", "val"):
        dest = FEATURES / "cache" / split
        dest.mkdir(parents=True, exist_ok=True)
        count = 0
        for row in common.read_manifest(split):
            common.OUT = OLD
            source = common.record_path(split, row)
            if source.exists():
                common.read_record(split, row)  # Validate identity, checksum metadata and finite features.
                assert common.digest(row["path"]) == row["sha256"]
                target = dest / source.name
                if not target.exists():
                    shutil.copyfile(source, target)
                assert common.digest(source) == common.digest(target)
                count += 1
        counts[split] = count
    common.OUT = FEATURES
    if (OLD / "extractor.json").exists():
        common.freeze_json(FEATURES / "extractor.json", json.loads((OLD / "extractor.json").read_text()))
    common.atomic_json(FEATURES / "cache_import.json", {"source": str(OLD), "compatible_records_copied": counts})
    print("Reused existing feature caches:", counts, flush=True)


def cache_text_embeddings(classes):
    path = FEATURES / "zero_shot_texts.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            assert saved["classes"].tolist() == classes
            assert saved["texts"].tolist() == [config.ZERO_SHOT_PROMPTS[c] for c in classes]
        return
    parts = load_zero_shot()
    texts = [config.ZERO_SHOT_PROMPTS[c] for c in classes]
    inputs = parts["processor"](text=texts, padding="max_length", return_tensors="pt").to(config.DEVICE)
    with torch.inference_mode():
        embeddings = parts["model"].get_text_features(**inputs).pooler_output.float().cpu()
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    assert embeddings.shape == (4, 1152) and torch.isfinite(embeddings).all()
    with path.with_suffix(".tmp").open("wb") as handle:
        np.savez_compressed(handle, embeddings=embeddings.numpy(), classes=np.asarray(classes), texts=np.asarray(texts))
    path.with_suffix(".tmp").replace(path)
    del parts
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def prepare_features(spec, splits):
    import_previous_cache(spec)
    cache_text_embeddings(spec["classes"])
    common.OUT = FEATURES
    parts = {s: common.read_manifest(s) for s in splits}
    metadata = common.extract_features(parts, spec["classes"])
    for split, rows in parts.items():
        common.load_features(split, rows, spec["classes"])
    common.atomic_json(OUT / "status.json", {"stage": "features_ready", "splits": list(splits),
                                            "cached_features": str(FEATURES)})
    return metadata


def load_split(split, classes):
    common.OUT = FEATURES
    cases = common.read_manifest(split)
    x, y = common.load_features(split, cases, classes)
    with (FEATURES / f"{split}_raw_scores.csv").open(newline="") as handle:
        scores = list(csv.DictReader(handle))
    assert [r["image_id"] for r in scores] == [r["image_id"] for r in cases]
    raw = np.asarray([[float(r["logprob_" + c]) for c in classes] for r in scores])
    assert np.isfinite(raw).all()
    return x, y, raw, cases


def normalize_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    # A zero standardized vector has no direction; cosine distance remains finite.
    return values / np.maximum(norms, 1e-12)


def softmax(values):
    exponentials = np.exp(values - values.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def zero_shot_probabilities(features, raw):
    with np.load(FEATURES / "zero_shot_texts.npz", allow_pickle=False) as saved:
        texts = saved["embeddings"]
    similarities = features[:, :1152].numpy() @ texts.T
    return (softmax(similarities) + softmax(raw)) / 2


def fitted_outputs(features, raw, fitted, heads):
    scaled = (features.numpy() - fitted["mean"]) / fitted["std"]
    with torch.inference_mode():
        combined = heads.probabilities(features)["averaged"].numpy()
    probabilities = {"zero_shot": zero_shot_probabilities(features, raw),
                     "knn": fitted["knn"].predict_proba(normalize_rows(scaled)),
                     "logistic_regression": fitted["logistic_regression"].predict_proba(scaled),
                     "linear_probe": combined}
    for name, values in probabilities.items():
        assert np.isfinite(values).all() and np.allclose(values.sum(axis=1), 1, atol=1e-5), f"Invalid probabilities: {name}"
    return probabilities


def save_reports(split, outputs, y, cases, classes, bootstrap=False):
    results = {}
    for name, probabilities in outputs.items():
        result = common.metrics(y.numpy(), probabilities, classes)
        result.update({"samples": len(cases), "load_failures": 0, "nonfinite_features": 0,
                       "parse_failures": 0, "parse_note": "Numeric frozen scores; no generated label parsing"})
        if bootstrap:
            common.OUT = OUT / name
            common.OUT.mkdir(exist_ok=True)
            result["bootstrap_95_ci"] = common.bootstrap_test(y.numpy(), probabilities, classes)
            cm = np.asarray(result["confusion_matrix"])
            for i, label in enumerate(classes):
                tp = int(cm[i, i]); fn = int(cm[i].sum() - tp)
                fp = int(cm[:, i].sum() - tp); tn = int(cm.sum() - tp - fn - fp)
                result["per_class"][label].update({"sensitivity_95_wilson_ci": common.wilson(tp, tp + fn),
                                                    "specificity_95_wilson_ci": common.wilson(tn, tn + fp)})
        results[name] = result
    fields = ["image_id", "path", "true_label"] + [f"{m}_{field}" for m in METHODS for field in ("prediction", "correct")] + [f"{m}_{c}" for m in METHODS for c in classes]
    with (OUT / f"{split}_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, case in enumerate(cases):
            row = {key: case[key] for key in fields[:3]}
            for method, p in outputs.items():
                predicted = classes[int(p[i].argmax())]
                row.update({f"{method}_prediction": predicted, f"{method}_correct": predicted == case["true_label"]})
                row.update({f"{method}_{c}": float(p[i, j]) for j, c in enumerate(classes)})
            writer.writerow(row)
    plot_reports(split, outputs, y.numpy(), classes, results)
    common.atomic_json(OUT / f"{split}_metrics.json", results)
    print(f"{split.upper()} RESULTS (internal dataset only)", flush=True)
    for name, result in results.items():
        print(f"{name:22} accuracy={result['accuracy']:.4f} balanced={result['balanced_accuracy']:.4f} macro-F1={result['macro_f1']:.4f} AUC={result['macro_ovr_auc']:.4f}", flush=True)
    return results


def plot_reports(split, outputs, truth, classes, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, result) in zip(axes.flat, results.items()):
        matrix = np.asarray(result["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(4), classes, rotation=25)
        ax.set_yticks(range(4), classes)
        ax.set(title=name, xlabel="Predicted", ylabel="True")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.suptitle(split + ": frozen MedSigLIP + MedGemma")
    fig.tight_layout()
    fig.savefig(OUT / f"{split}_confusion_matrices.png", dpi=150)
    plt.close(fig)
    if split == "test":
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for ax, (method, probabilities) in zip(axes.flat, outputs.items()):
            for i, label in enumerate(classes):
                fpr, tpr, _ = roc_curve(truth == i, probabilities[:, i])
                ax.plot(fpr, tpr, label=f"{label} {roc_auc_score(truth == i, probabilities[:, i]):.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=.3)
            ax.set(title=method, xlabel="False positive rate", ylabel="Sensitivity")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "test_roc.png", dpi=150)
        plt.close(fig)


def train(spec):
    if (OUT / "selection.json").exists():
        print("Selection already frozen; skipping training.", flush=True)
        return
    classes = spec["classes"]
    x, y, _, _ = load_split("train", classes)
    val_x, val_y, raw, cases = load_split("val", classes)
    scaler = FusedHeads(1156, 4)
    scaler.fit_scaler(x)
    mean, std = scaler.mean.numpy(), scaler.std.numpy()
    scaled = (x.numpy() - mean) / std
    val_scaled = (val_x.numpy() - mean) / std
    candidates, best_key, knn = [], None, None
    for k in K_VALUES:
        candidate = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="uniform", algorithm="brute", n_jobs=4)
        candidate.fit(normalize_rows(scaled), y.numpy())
        predicted = candidate.predict(normalize_rows(val_scaled))
        key = (float(f1_score(val_y, predicted, average="macro", zero_division=0)),
               float(balanced_accuracy_score(val_y, predicted)), -k)
        candidates.append({"k": k, "validation_macro_f1": key[0], "validation_balanced_accuracy": key[1]})
        if best_key is None or key > best_key:
            best_key, knn = key, candidate
    common.atomic_json(OUT / "knn_selection.json", candidates)
    logistic = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        logistic.fit(scaled, y.numpy())
    common.OUT = OUT / "linear_probe"
    common.OUT.mkdir(exist_ok=True)
    # The beginner-readable ten-epoch loop lives beside the original probe loop.
    from training.train_probe import train_new_probe_with_epochs
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    heads, trial, ce_weights, pos_weights = train_new_probe_with_epochs(x, y, val_x, val_y, classes)
    assert trial["epochs_run"] == EPOCHS
    common.atomic_json(common.OUT / "training_trials.json", [trial])
    with (MODELS / "linear_probe.pth").open("xb") as handle:
        torch.save({"state_dict": heads.state_dict(), "classes": classes, "feature_dim": 1156,
                    "extractor_metadata": json.loads((FEATURES / "extractor.json").read_text()),
                    "protocol": spec, "best_epoch": trial["best_epoch"], "epochs_run": EPOCHS,
                    "ce_class_weights": ce_weights, "bce_positive_weights": pos_weights}, handle)
    fitted = {"knn": knn, "logistic_regression": logistic, "mean": mean, "std": std, "classes": classes}
    joblib.dump(fitted, MODELS / "classifiers.joblib")
    common.save_validation(heads, val_x, val_y, cases, classes, trial)
    outputs = fitted_outputs(val_x, raw, fitted, heads)
    save_reports("val", outputs, val_y, cases, classes)
    common.atomic_json(OUT / "selection.json", {"best_k": knn.n_neighbors, "linear_probe_best_epoch": trial["best_epoch"],
             "epochs_run": EPOCHS, "model_sha256": {name: common.digest(MODELS / name) for name in ("linear_probe.pth", "classifiers.joblib")},
             "test_evaluated": False, "no_further_tuning": True})
    print(f"Saved best epoch {trial['best_epoch']} of 10. Selection frozen before test.", flush=True)


def test_once(spec):
    if (OUT / "test_metrics.json").exists():
        print("Test already completed. Saved results:", (OUT / "test_metrics.json").read_text(), flush=True)
        return
    selection = json.loads((OUT / "selection.json").read_text())
    for name, checksum in selection["model_sha256"].items():
        assert common.digest(MODELS / name) == checksum, f"Frozen model changed: {name}"
    common.freeze_json(OUT / "test_protocol.json", {"selection_sha256": common.digest(OUT / "selection.json"),
                "manifest_sha256": spec["manifest_sha256"]["test"], "resamples": 2000, "seed": 42, "no_retuning": True})
    prepare_features(spec, ("test",))
    x, y, raw, cases = load_split("test", spec["classes"])
    fitted = joblib.load(MODELS / "classifiers.joblib")
    saved = torch.load(MODELS / "linear_probe.pth", map_location="cpu", weights_only=True)
    heads = FusedHeads(1156, 4).eval()
    heads.load_state_dict(saved["state_dict"])
    outputs = fitted_outputs(x, raw, fitted, heads)
    save_reports("test", outputs, y, cases, spec["classes"], bootstrap=True)
    with torch.inference_mode():
        head_information = {name: common.metrics(y.numpy(), p.numpy(), spec["classes"])
                            for name, p in heads.probabilities(x).items()}
    common.atomic_json(OUT / "test_head_information.json", head_information)
    for name, checksum in selection["model_sha256"].items():
        assert common.digest(MODELS / name) == checksum, "Model changed during final test"
    common.atomic_json(OUT / "status.json", {"stage": "complete", "test_images": len(cases), "retuning": False})


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("features", "train", "test", "all"), default="all")
    args = parser.parse_args(arguments)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    spec = protocol()
    try:
        if args.stage in ("features", "all"):
            prepare_features(spec, ("train", "val"))
        if args.stage in ("train", "all"):
            train(spec)
        if args.stage in ("test", "all"):
            test_once(spec)
    except Exception as error:
        with (OUT / "failures.jsonl").open("a") as handle:
            handle.write(json.dumps({"stage": args.stage, "error": f"{type(error).__name__}: {error}", "images_silently_dropped": 0}) + "\n")
        raise


if __name__ == "__main__":
    main()
