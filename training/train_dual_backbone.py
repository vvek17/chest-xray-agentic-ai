"""Cache TRAIN/VAL features, freeze one combined model, then evaluate TEST once.

python -B -u -m training.train_dual_backbone
Resume is cache-aware. Test is opened only after validation selection is frozen.
"""

import copy
import csv
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)

from backend import config
from backend.dual_backbone import FrozenFeatures, FusedHeads

ROOT = config.PROJECT_ROOT
DATA = ROOT / "data/new_distribution"
OUT = ROOT / "outputs/new_distribution_fusion"
CHECKPOINT = ROOT / "models/new_distribution_fusion.pth"
SEED, BATCH_SIZE = 42, 4
HEAD_BATCH, MAX_EPOCHS, PATIENCE = 128, 200, 20
LEARNING_RATES, WEIGHT_DECAYS = (0.001, 0.003, 0.01), (0.0001, 0.01)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def freeze_json(path, value):
    value = json.loads(json.dumps(value))
    if path.exists():
        assert json.loads(path.read_text()) == value, f"Frozen definition differs: {path}"
    else:
        atomic_json(path, value)
    return value


def read_manifest(split):
    with (DATA / "manifests" / f"{split}.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def record_path(split, row):
    key = hashlib.sha256(row["image_id"].encode()).hexdigest()
    return OUT / "cache" / split / (key + ".npz")


def read_record(split, row):
    path = record_path(split, row)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        assert metadata["image_id"] == row["image_id"] and metadata["sha256"] == row["sha256"]
        features = saved["features"].copy()
        assert features.shape == (1156,) and np.isfinite(features).all(), f"Invalid cache: {path}"
        return features, metadata


def write_record(split, row, features, raw):
    path = record_path(split, row)
    metadata = {**raw, "image_id": row["image_id"], "sha256": row["sha256"],
                "true_label": row["true_label"], "path": row["path"], "split": split}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, features=features.numpy(), metadata=json.dumps(metadata, allow_nan=False))
    temporary.replace(path)


def extract_features(parts, classes):
    missing = {split: [r for r in rows if read_record(split, r) is None] for split, rows in parts.items()}
    if not any(missing.values()):
        print("All train/val features cached; no backbone inference.", flush=True)
        return json.loads((OUT / "extractor.json").read_text())
    extractor = FrozenFeatures.load(classes)
    metadata = freeze_json(OUT / "extractor.json", extractor.metadata())
    print("SCORING TARGETS", json.dumps(metadata["targets"]), flush=True)
    started, done = time.monotonic(), 0
    total_missing = sum(map(len, missing.values()))
    for split, rows in missing.items():
        directory = OUT / "cache" / split
        directory.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            try:
                for row in batch:
                    assert digest(row["path"]) == row["sha256"], f"Dataset copy changed: {row['image_id']}"
                features, raw = extractor.extract([r["path"] for r in batch])
                assert features.shape == (len(batch), 1152 + len(classes)) and torch.isfinite(features).all()
                for row, feature, output in zip(batch, features, raw):
                    write_record(split, row, feature, output)
            except Exception as error:
                failure = {"stage": "frozen_feature_extraction", "split": split,
                           "image_ids": [r["image_id"] for r in batch], "error": f"{type(error).__name__}: {error}",
                           "action": "Stopped. No images silently skipped. Successful earlier batches remain cached."}
                with (OUT / "failures.jsonl").open("a") as handle:
                    handle.write(json.dumps(failure) + "\n")
                atomic_json(OUT / "status.json", failure)
                raise
            done += len(batch)
            elapsed = time.monotonic() - started
            status = {"stage": "extracting_frozen_features", "split": split, "newly_cached_this_run": done,
                      "missing_at_start": total_missing, "elapsed_seconds": round(elapsed, 1),
                      "estimated_remaining_seconds": round(elapsed / done * (total_missing - done), 1),
                      "test_images_processed": done if split == "test" else 0, "current_batch_errors": 0}
            atomic_json(OUT / "status.json", status)
            if start == 0 or done % 40 == 0 or start + len(batch) == len(rows):
                print(json.dumps(status), flush=True)
    del extractor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return metadata


def load_features(split, rows, classes):
    features, labels, raw = [], [], []
    for row in rows:
        result = read_record(split, row)
        assert result is not None, f"Missing features: {row['image_id']}"
        feature, metadata = result
        assert metadata["true_label"] == row["true_label"]
        features.append(feature)
        labels.append(classes.index(row["true_label"]))
        raw.append(metadata)
    array = np.stack(features)
    with (OUT / f"{split}_raw_scores.csv").open("w", newline="") as handle:
        fields = ["image_id", "true_label", "path", "sha256", "medgemma_argmax_token_id", "medgemma_argmax_token_text"] + ["logprob_" + c for c in classes]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metadata in raw:
            row = {key: metadata[key] for key in fields[:6]}
            row.update(zip(fields[6:], metadata["medgemma_raw_class_logprobs"]))
            writer.writerow(row)
    np.savez_compressed(OUT / f"{split}_features.npz", features=array, labels=np.asarray(labels),
                        image_ids=np.asarray([r["image_id"] for r in rows]))
    return torch.from_numpy(array), torch.tensor(labels, dtype=torch.long)


def metrics(labels, probabilities, classes):
    predicted = probabilities.argmax(axis=1)
    cm = confusion_matrix(labels, predicted, labels=list(range(len(classes))))
    precision, recall, _, support = precision_recall_fscore_support(labels, predicted, labels=list(range(len(classes))), zero_division=0)
    specificity = []
    for i in range(len(classes)):
        fp = int(cm[:, i].sum() - cm[i, i])
        tn = int(cm.sum() - cm[i].sum() - fp)
        specificity.append(tn / (tn + fp))
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
            "macro_ovr_auc": float(roc_auc_score(np.eye(len(classes))[labels], probabilities, average="macro")),
            "confusion_matrix": cm.tolist(), "class_order": classes,
            "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "specificity": specificity[i], "support": int(support[i])} for i, c in enumerate(classes)}}


def train_heads(train_x, train_y, val_x, val_y, classes):
    # Keep head optimization reproducible on CPU; backbones have no gradients.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    counts = torch.bincount(train_y, minlength=len(classes)).float()
    ce_weights = len(train_y) / (len(classes) * counts)
    pos_weights = (len(train_y) - counts) / counts
    ce = torch.nn.CrossEntropyLoss(weight=ce_weights)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    train_targets = torch.nn.functional.one_hot(train_y, len(classes)).float()
    val_targets = torch.nn.functional.one_hot(val_y, len(classes)).float()
    best_overall, winner, trials, winner_history = None, None, [], None
    for lr in LEARNING_RATES:
        for wd in WEIGHT_DECAYS:
            torch.manual_seed(SEED)
            generator = torch.Generator().manual_seed(SEED)
            heads = FusedHeads(train_x.shape[1], len(classes))
            heads.fit_scaler(train_x)
            optimizer = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=wd)
            history, best_key, state, best_epoch, stalled = [], None, None, 0, 0
            for epoch in range(1, MAX_EPOCHS + 1):
                heads.train()
                order = torch.randperm(len(train_y), generator=generator)
                total = 0.0
                for start in range(0, len(order), HEAD_BATCH):
                    indices = order[start:start + HEAD_BATCH]
                    optimizer.zero_grad()
                    a, b = heads(train_x[indices])
                    loss = ce(a, train_y[indices]) + bce(b, train_targets[indices])
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(indices)
                heads.eval()
                with torch.inference_mode():
                    a, b = heads(val_x)
                    val_loss = float(ce(a, val_y) + bce(b, val_targets))
                    probability = heads.probabilities(val_x)["averaged"].numpy()
                predicted = probability.argmax(axis=1)
                macro_f1 = float(f1_score(val_y, predicted, average="macro", zero_division=0))
                balanced = float(balanced_accuracy_score(val_y, predicted))
                key = (macro_f1, balanced, -val_loss)
                history.append({"epoch": epoch, "train_loss": total / len(train_y), "val_loss": val_loss,
                                "val_macro_f1": macro_f1, "val_balanced_accuracy": balanced})
                if best_key is None or key > best_key:
                    best_key, state, best_epoch, stalled = key, copy.deepcopy(heads.state_dict()), epoch, 0
                else:
                    stalled += 1
                if stalled >= PATIENCE:
                    break
            trial = {"lr": lr, "weight_decay": wd, "best_epoch": best_epoch,
                     "epochs_run": len(history), "validation_macro_f1": best_key[0],
                     "validation_balanced_accuracy": best_key[1], "history": history}
            trials.append(trial)
            atomic_json(OUT / "training_trials.json", trials)
            print("HEAD TRIAL", json.dumps({k: v for k, v in trial.items() if k != "history"}), flush=True)
            if best_overall is None or best_key > best_overall:
                best_overall = best_key
                heads.load_state_dict(state)
                winner, winner_history = heads, trial
    return winner.eval(), winner_history, ce_weights.tolist(), pos_weights.tolist()


def save_validation(heads, val_x, val_y, cases, classes, trial):
    with torch.inference_mode():
        outputs = {key: value.numpy() for key, value in heads.probabilities(val_x).items()}
    result = metrics(val_y.numpy(), outputs["averaged"], classes)
    result.update({"samples": len(cases), "load_failures": 0, "nonfinite_feature_rows": 0,
                   "nonfinite_probability_rows": 0, "parse_failures": 0,
                   "parse_note": "No generated answer parsing; numeric full-vocabulary scores only.",
                   "test_evaluated": False, "selection": "Validation macro-F1, ties balanced accuracy then lower validation loss"})
    atomic_json(OUT / "validation_metrics.json", result)
    with (OUT / "validation_predictions.csv").open("w", newline="") as handle:
        fields = ["image_id", "path", "true_label", "prediction", "correct"] + [f"{kind}_{c}" for kind in outputs for c in classes]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, case in enumerate(cases):
            predicted = classes[int(outputs["averaged"][i].argmax())]
            row = {"image_id": case["image_id"], "path": case["path"], "true_label": case["true_label"],
                   "prediction": predicted, "correct": predicted == case["true_label"]}
            row.update({f"{kind}_{c}": float(values[i, j]) for kind, values in outputs.items() for j, c in enumerate(classes)})
            writer.writerow(row)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = trial["history"]
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("train_loss", "val_loss"):
        axes[0].plot(epochs, [r[name] for r in history], label=name)
    for name in ("val_macro_f1", "val_balanced_accuracy"):
        axes[1].plot(epochs, [r[name] for r in history], label=name)
    for ax in axes:
        ax.axvline(trial["best_epoch"], color="gray", linestyle=":")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = np.asarray(result["confusion_matrix"])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=20)
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Validation: combined frozen-backbone model")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(OUT / "validation_confusion_matrix.png", dpi=160)
    plt.close(fig)
    print("VALIDATION CHECKPOINT", json.dumps(result), flush=True)
    return result


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def bootstrap_test(labels, probabilities, classes):
    rng = np.random.default_rng(SEED)
    strata = [np.flatnonzero(labels == i) for i in range(len(classes))]
    values = {key: [] for key in ("macro_ovr_auc", "macro_f1", "balanced_accuracy")}
    for _ in range(2000):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        truth, p = labels[indices], probabilities[indices]
        predicted = p.argmax(axis=1)
        values["macro_ovr_auc"].append(roc_auc_score(np.eye(len(classes))[truth], p, average="macro"))
        values["macro_f1"].append(f1_score(truth, predicted, average="macro", zero_division=0))
        values["balanced_accuracy"].append(balanced_accuracy_score(truth, predicted))
    np.savez_compressed(OUT / "test_bootstrap.npz", **{key: np.asarray(v) for key, v in values.items()})
    return {key: np.quantile(v, [0.025, 0.975]).tolist() for key, v in values.items()}


def evaluate_test_once():
    """No optimization: frozen checkpoint, cached features and one test report."""
    if (OUT / "test_metrics.json").exists():
        print("Final test already completed; no inference or retuning.", flush=True)
        return json.loads((OUT / "test_metrics.json").read_text())
    checkpoint_hash = digest(CHECKPOINT)
    freeze_json(OUT / "test_protocol.json", {"checkpoint_sha256": checkpoint_hash,
                "test_manifest_sha256": digest(DATA / "manifests/test.csv"), "seed": SEED,
                "bootstrap": {"resamples": 2000, "stratified_by_class": True, "ci": "95% percentile"},
                "per_class_ci": "95% Wilson sensitivity/specificity", "retuning_allowed": False,
                "authorization": "User requested train validation and test all"})
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    classes = saved["classes"]
    cases = read_manifest("test")
    metadata = extract_features({"test": cases}, classes)
    assert metadata == saved["extractor_metadata"], "Test extractor changed"
    test_x, test_y = load_features("test", cases, classes)
    heads = FusedHeads(saved["feature_dim"], len(classes)).eval()
    heads.load_state_dict(saved["state_dict"])
    with torch.inference_mode():
        outputs = {key: value.numpy() for key, value in heads.probabilities(test_x).items()}
    truth, probabilities = test_y.numpy(), outputs["averaged"]
    result = metrics(truth, probabilities, classes)
    result["bootstrap_95_ci"] = bootstrap_test(truth, probabilities, classes)
    cm = np.asarray(result["confusion_matrix"])
    for i, label in enumerate(classes):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        result["per_class"][label].update({"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                  "sensitivity_95_wilson_ci": wilson(tp, tp + fn),
                  "specificity_95_wilson_ci": wilson(tn, tn + fp)})
    result.update({"samples": len(cases), "checkpoint_sha256": checkpoint_hash, "test_evaluated_once": True,
                   "parse_failures": 0, "parse_note": "Numeric logits only; no generated label parsing.",
                   "load_failures": 0, "nonfinite_features": 0, "nonfinite_probabilities": 0,
                   "internal_dataset_only": True, "patient_metadata_incomplete": True,
                   "retuning_performed": False,
                   "head_information": {name: metrics(truth, p, classes) for name, p in outputs.items()}})
    with (OUT / "test_predictions.csv").open("w", newline="") as handle:
        fields = ["image_id", "path", "true_label", "prediction", "correct"] + [f"{kind}_{c}" for kind in outputs for c in classes]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, case in enumerate(cases):
            predicted = classes[int(probabilities[i].argmax())]
            row = {"image_id": case["image_id"], "path": case["path"], "true_label": case["true_label"],
                   "prediction": predicted, "correct": predicted == case["true_label"]}
            row.update({f"{kind}_{c}": float(p[i, j]) for kind, p in outputs.items() for j, c in enumerate(classes)})
            writer.writerow(row)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=20)
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Final test: frozen combined model")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(OUT / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, label in enumerate(classes):
        fpr, tpr, _ = roc_curve(truth == i, probabilities[:, i])
        auc = roc_auc_score(truth == i, probabilities[:, i])
        ax.plot(fpr, tpr, label=f"{label}: AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="False positive rate", ylabel="Sensitivity", title="Final test: per-class ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "test_roc.png", dpi=160)
    plt.close(fig)
    assert digest(CHECKPOINT) == checkpoint_hash, "Checkpoint changed during test"
    atomic_json(OUT / "test_metrics.json", result)
    atomic_json(OUT / "status.json", {"stage": "complete", "test_images_processed": len(cases),
                                     "checkpoint_sha256": checkpoint_hash, "test_metrics": result})
    print("FINAL TEST", json.dumps(result), flush=True)
    return result


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {split: read_manifest(split) for split in ("train", "val")}
    classes = sorted({r["true_label"] for r in parts["train"]})
    assert len(classes) == 4
    assert not {r["group_id"] for r in parts["train"]} & {r["group_id"] for r in parts["val"]}
    source_files = [Path(__file__), ROOT / "backend/dual_backbone.py", ROOT / "backend/clean_image.py",
                    ROOT / "backend/medgemma.py", ROOT / "backend/zero_shot_ensemble.py"]
    protocol = freeze_json(OUT / "protocol.json", {
        "seed": SEED, "classes": classes, "dataset": str(DATA),
        "manifest_sha256": {s: digest(DATA / "manifests" / f"{s}.csv") for s in ("train", "val", "test")},
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_files},
        "selection": "Validation macro-F1, ties balanced accuracy then lower validation loss; first trial wins remaining ties",
        "head_training": {"lr": LEARNING_RATES, "weight_decay": WEIGHT_DECAYS, "batch_size": HEAD_BATCH,
                          "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "loss": "Class-weighted CE + positive-weighted BCEWithLogits"},
        "feature_batch": BATCH_SIZE, "feature_dim": 1152 + len(classes), "optional_pooled_hidden_state": False,
        "scaler": "Population mean/std of TRAIN only, std floor 1e-6, MedGemma logprobs clipped minimum -20",
        "backbones_trainable": False, "outputs": "Linear-softmax + Linear-sigmoid; average probabilities then renormalize",
        "test_access": "After validation selection and immutable checkpoint only; evaluate once, no retuning", "NIH_used": False})
    if (OUT / "validation_metrics.json").exists():
        assert CHECKPOINT.exists(), "Validation results exist without checkpoint"
        print("Training already complete; no retraining.", flush=True)
        evaluate_test_once()
        return
    assert not CHECKPOINT.exists(), "Checkpoint already exists without final results; refusing overwrite"
    metadata = extract_features(parts, classes)
    train_x, train_y = load_features("train", parts["train"], classes)
    val_x, val_y = load_features("val", parts["val"], classes)
    atomic_json(OUT / "status.json", {"stage": "training_heads", "train_images": len(train_y), "validation_images": len(val_y), "test_images_processed": 0})
    heads, trial, ce_weights, pos_weights = train_heads(train_x, train_y, val_x, val_y, classes)
    CHECKPOINT.parent.mkdir(exist_ok=True)
    with CHECKPOINT.open("xb") as handle:
        torch.save({"state_dict": heads.state_dict(), "classes": classes, "feature_dim": train_x.shape[1],
                    "extractor_metadata": metadata, "protocol": protocol,
                    "best_hyperparameters": {k: v for k, v in trial.items() if k != "history"},
                    "ce_class_weights": ce_weights, "bce_positive_weights": pos_weights}, handle)
    # Verify serialization gives the same final probabilities before reporting.
    restored = FusedHeads(train_x.shape[1], len(classes))
    restored.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"])
    with torch.inference_mode():
        torch.testing.assert_close(heads.probabilities(val_x)["averaged"], restored.probabilities(val_x)["averaged"], rtol=0, atol=0)
    result = save_validation(heads, val_x, val_y, parts["val"], classes, trial)
    atomic_json(OUT / "status.json", {"stage": "validation_checkpoint_complete", "checkpoint": str(CHECKPOINT),
                                   "checkpoint_sha256": digest(CHECKPOINT), "validation_metrics": result,
                                   "test_images_processed": 0})
    print("Model and validation selection frozen. Starting authorized final test.", flush=True)
    evaluate_test_once()


if __name__ == "__main__":
    main()
