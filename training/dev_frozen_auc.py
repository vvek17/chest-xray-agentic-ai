"""Step 2 only: dev AUC and paired, stratified-bootstrap intervals. Stop afterwards.

Run: python -m training.dev_frozen_auc
Does not load models, validation scores, final-test images, or thresholds.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/frozen_nih_threshold"
SEED = 42
RESAMPLES = 2000
METHODS = ("MedSigLIP", "MedGemma", "Fusion")


def dev_fusion(siglip, medgemma):
    scores = np.column_stack([siglip, medgemma]).astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite dev scores; no images may be silently dropped")
    means, stds = scores.mean(axis=0), scores.std(axis=0, ddof=0)
    if (stds <= 0).any():
        raise ValueError("Cannot z-score a constant score vector")
    fused = ((scores - means) / stds).mean(axis=1)
    return np.column_stack([scores, fused]), {"means": means.tolist(), "stds": stds.tolist(),
                                             "ddof": 0, "fitted_on": "dev only",
                                             "order": ["siglip_score", "medgemma_score"]}


def bootstrap_auc(labels, scores, resamples=RESAMPLES, seed=SEED):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores")
    groups = [np.flatnonzero(labels == value) for value in [0, 1]]
    if not all(len(group) for group in groups):
        raise ValueError("Both classes required")
    rng = np.random.default_rng(seed)
    aucs = np.empty((resamples, scores.shape[1]))
    for i in range(resamples):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        for j in range(scores.shape[1]):
            aucs[i, j] = roc_auc_score(labels[indices], scores[indices, j])
    return aucs


def main():
    path = OUT / "dev_scores.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 200 and len({r["image_id"] for r in rows}) == 200
    assert all(r["siglip_status"] == r["medgemma_status"] == "ok" for r in rows), "Scoring errors exist"
    labels = np.array([int(r["true_label"] == "Pneumonia") for r in rows])
    assert labels.sum() == 100
    siglip = np.array([float(r["siglip_score"]) for r in rows])
    medgemma = np.array([float(r["medgemma_score"]) for r in rows])
    scores, scaling = dev_fusion(siglip, medgemma)
    point = np.array([roc_auc_score(labels, scores[:, j]) for j in range(3)])
    boot = bootstrap_auc(labels, scores)
    bounds = np.quantile(boot, [0.025, 0.975], axis=0)
    best = int(np.argmax(point[:2]))
    differences = boot[:, 2] - boot[:, best]
    delta_ci = np.quantile(differences, [0.025, 0.975])
    result = {
        "stage": "Step 2 checkpoint", "split": "dev", "n": len(rows),
        "class_counts": {"Normal": 100, "Pneumonia": 100}, "positive_class": "Pneumonia",
        "seed": SEED, "bootstrap_resamples": RESAMPLES,
        "bootstrap": "Class-stratified percentile 95% CI; identical resampled cases for all methods; dev scaling held fixed",
        "scaling": scaling,
        "methods": {name: {"auc": float(point[j]), "ci95": bounds[:, j].tolist(),
                           "ci_includes_0_5": bool(bounds[0, j] <= 0.5 <= bounds[1, j])}
                    for j, name in enumerate(METHODS)},
        "fusion_comparison": {"better_single": METHODS[best], "auc_difference": float(point[2] - point[best]),
                              "paired_bootstrap_ci95": delta_ci.tolist(),
                              "ci_entirely_above_zero": bool(delta_ci[0] > 0)},
        "parse_failures": sum(r["parse_failure"] == "True" for r in rows),
        "score_sign_greedy_agreement": {"agree": sum(r["sign_agrees_greedy"] == "True" for r in rows),
                                         "denominator": len(rows)},
        "dev_scores_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "thresholds_selected": False, "validation_performance_evaluated": False, "final_test_inference": False,
    }
    np.savez_compressed(OUT / "dev_auc_bootstrap.npz", aucs=boot, methods=np.array(METHODS),
                        fusion_minus_best_single=differences, seed=SEED)
    with (OUT / "dev_scores_with_fusion.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0], "fused_score"])
        writer.writeheader()
        for row, value in zip(rows, scores[:, 2]):
            writer.writerow(dict(row, fused_score=value))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    colors = ["#2563eb", "#d97706", "#059669"]
    for selected, filename in [(range(3), "dev_roc.png"), *[([j], f"dev_roc_{METHODS[j].lower()}.png") for j in range(3)]]:
        fig, ax = plt.subplots(figsize=(7, 6))
        for j in selected:
            fpr, tpr, _ = roc_curve(labels, scores[:, j])
            ax.plot(fpr, tpr, color=colors[j], linewidth=2,
                    label=f"{METHODS[j]}: AUC {point[j]:.3f} [{bounds[0,j]:.3f}, {bounds[1,j]:.3f}]")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set(xlabel="False-positive rate (1 - specificity)", ylabel="Sensitivity",
               title="Frozen NIH dev ROC (100 Normal / 100 Pneumonia)", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)
    (OUT / "step2_dev_auc.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if all(result["methods"][name]["ci_includes_0_5"] for name in METHODS[:2]):
        print("BOTH single-model AUC intervals include 0.5. STOP; do not tune anything.")
    print("STEP 2 CHECKPOINT. STOP. No thresholds, validation evaluation or final-test inference.")


if __name__ == "__main__":
    main()
