"""Steps 3 and 4: freeze on dev, evaluate cached validation once, then STOP.

python -m training.validate_frozen_thresholds
No model imports/inference. No final-test access. Existing thresholds/results
are verified and reused, never overwritten or adjusted based on validation.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from training.dev_frozen_auc import METHODS, OUT, ROOT, SEED, RESAMPLES, bootstrap_auc, dev_fusion


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scores(split):
    path = OUT / f"{split}_scores.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest = ROOT / "data/external/NIH_development/manifests" / f"{split}.csv"
    with manifest.open(newline="") as handle:
        expected = list(csv.DictReader(handle))
    assert [(r["image_id"], r["true_label"]) for r in rows] == [(r["image_id"], r["true_label"]) for r in expected]
    assert len(rows) == len({r["image_id"] for r in rows}) == 200
    assert all(r["siglip_status"] == r["medgemma_status"] == "ok" for r in rows)
    labels = np.array([int(r["true_label"] == "Pneumonia") for r in rows])
    scores = np.array([[float(r["siglip_score"]), float(r["medgemma_score"])] for r in rows])
    assert labels.sum() == 100
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores present; no image may be silently dropped")
    return rows, labels, scores


def choose_threshold(labels, scores, sensitivity=0.80):
    """Use >=; exact observed cutoffs include score ties as a group."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores")
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if not positives or not negatives:
        raise ValueError("Both classes required")
    options = []
    for cutoff in np.unique(scores):
        predictions = scores >= cutoff
        tp = int((predictions & (labels == 1)).sum())
        tn = int((~predictions & (labels == 0)).sum())
        if tp >= math.ceil(sensitivity * positives):
            options.append((tn, float(cutoff), tp))
    tn, cutoff, tp = max(options, key=lambda row: (row[0], row[1]))
    return {"threshold": cutoff, "dev_counts": {"TP": tp, "FN": positives - tp,
                                                "TN": tn, "FP": negatives - tn},
            "dev_sensitivity": tp / positives, "dev_specificity": tn / negatives}


def wilson(successes, total):
    if not total:
        raise ValueError("Empty denominator")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def exact_mcnemar(truth, predictions, baseline):
    correct = np.asarray(predictions) == truth
    baseline_correct = np.asarray(baseline) == truth
    gains = int((correct & ~baseline_correct).sum())
    losses = int((~correct & baseline_correct).sum())
    n = gains + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2 ** n) if n else 1.0
    return {"method_correct_baseline_wrong": gains, "method_wrong_baseline_correct": losses,
            "exact_two_sided_p": p}


def binary_metrics(truth, predictions):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    tp = int(((truth == 1) & (predictions == 1)).sum())
    fn = int(((truth == 1) & (predictions != 1)).sum())
    tn = int(((truth == 0) & (predictions == 0)).sum())
    fp = int(((truth == 0) & (predictions != 0)).sum())
    return {"n": len(truth), "counts": {"TP": tp, "FN": fn, "TN": tn, "FP": fp},
            "matrix_order": ["Normal", "Pneumonia"], "confusion_matrix": [[tn, fp], [fn, tp]],
            "sensitivity": tp / (tp + fn), "sensitivity_ci95": wilson(tp, tp + fn),
            "specificity": tn / (tn + fp), "specificity_ci95": wilson(tn, tn + fp),
            "accuracy": float((truth == predictions).mean()),
            "Other": int(((predictions != 0) & (predictions != 1)).sum())}


def freeze_thresholds():
    protocol = json.loads((OUT / "step1_protocol.json").read_text())
    for name, expected in protocol["source_hashes"].items():
        assert checksum(ROOT / name) == expected, f"Scoring source changed: {name}"
    step2 = json.loads((OUT / "step2_dev_auc.json").read_text())
    assert checksum(OUT / "dev_scores.csv") == step2["dev_scores_sha256"]
    _, truth, raw = read_scores("dev")
    scores, scaling = dev_fusion(raw[:, 0], raw[:, 1])
    assert scaling == step2["scaling"]
    frozen = {"stage": 3, "positive_class": "Pneumonia", "prediction_rule": "score >= threshold",
              "minimum_dev_sensitivity": 0.80,
              "threshold_selection": "Maximum specificity subject to sensitivity >=0.80; ties use highest threshold",
              "validation_selection": "Highest specificity subject to sensitivity >=0.75; ties use method definition order",
              "method_order": list(METHODS), "dev_zscore": scaling,
              "dev_scores_sha256": checksum(OUT / "dev_scores.csv"),
              "step2_sha256": checksum(OUT / "step2_dev_auc.json"),
              "evaluator_source_sha256": checksum(Path(__file__)),
              "methods": {name: choose_threshold(truth, scores[:, j]) for j, name in enumerate(METHODS)}}
    path = OUT / "thresholds.json"
    if path.exists():
        assert json.loads(path.read_text()) == frozen, "Existing frozen thresholds differ; refusing retuning"
    else:
        with path.open("x") as handle:
            json.dump(frozen, handle, indent=2)
    print("FROZEN DEV THRESHOLDS:", json.dumps(frozen["methods"]), flush=True)
    return frozen


def main():
    frozen = freeze_thresholds()  # Written before validation scores are opened.
    frozen_hash = checksum(OUT / "thresholds.json")
    output = OUT / "step4_validation.json"
    if output.exists():
        existing = json.loads(output.read_text())
        assert existing["thresholds_sha256"] == frozen_hash
        assert existing["validation_scores_sha256"] == checksum(OUT / "validation_scores.csv")
        assert existing["predictions_sha256"] == checksum(OUT / "validation_predictions.csv")
        print("Reusing completed validation; no reevaluation.")
        print(json.dumps(existing, indent=2))
        return
    rows, truth, raw = read_scores("validation")
    scaling = frozen["dev_zscore"]
    fused = ((raw - np.array(scaling["means"])) / np.array(scaling["stds"])).mean(axis=1)
    scores = np.column_stack([raw, fused])
    baseline = np.array([1 if r["medgemma_greedy_answer"] == "Pneumonia" else
                         0 if r["medgemma_greedy_answer"] == "Normal" else -1 for r in rows])
    baseline_metrics = binary_metrics(truth, baseline)
    baseline_metrics["parse_failures_count_as_errors"] = int((baseline == -1).sum())
    boot = bootstrap_auc(truth, scores, resamples=RESAMPLES, seed=SEED)
    bounds = np.quantile(boot, [0.025, 0.975], axis=0)
    methods, predictions = {}, {}
    for j, name in enumerate(METHODS):
        predictions[name] = (scores[:, j] >= frozen["methods"][name]["threshold"]).astype(int)
        result = binary_metrics(truth, predictions[name])
        result.update(threshold=frozen["methods"][name]["threshold"],
                      auc=float(roc_auc_score(truth, scores[:, j])), auc_ci95=bounds[:, j].tolist(),
                      mcnemar_vs_greedy=exact_mcnemar(truth, predictions[name], baseline))
        methods[name] = result
    eligible = [name for name in METHODS if methods[name]["sensitivity"] >= 0.75]
    winner = max(eligible, key=lambda name: methods[name]["specificity"]) if eligible else None
    records = []
    for index, row in enumerate(rows):
        item = dict(row, fused_score=float(fused[index]))
        for name in METHODS:
            item[name.lower() + "_prediction"] = "Pneumonia" if predictions[name][index] else "Normal"
            item[name.lower() + "_correct"] = bool(predictions[name][index] == truth[index])
        records.append(item)
    with (OUT / "validation_predictions.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(OUT / "validation_auc_bootstrap.npz", aucs=boot, methods=np.array(METHODS), seed=SEED)
    report = {"stage": "Step 4 checkpoint", "split": "validation", "seed": SEED,
              "bootstrap_resamples": RESAMPLES, "methods": methods, "greedy_baseline": baseline_metrics,
              "winner": winner, "selection_rule": frozen["validation_selection"],
              "eligible_methods": eligible, "thresholds_sha256": frozen_hash,
              "validation_scores_sha256": checksum(OUT / "validation_scores.csv"),
              "predictions_sha256": checksum(OUT / "validation_predictions.csv"),
              "nonfinite_scores": int((~np.isfinite(scores)).sum()),
              "final_test_inference": False, "retuning": False}
    assert checksum(OUT / "thresholds.json") == frozen_hash
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print("STOP AT STEP 4 CHECKPOINT. Final test not run. No retuning.", flush=True)


if __name__ == "__main__":
    main()
