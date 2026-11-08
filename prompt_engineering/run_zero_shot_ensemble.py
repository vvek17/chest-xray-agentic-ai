"""Run 20 development / 40 validation / 140 test photos, without weight training.

python -m prompt_engineering.run_zero_shot_ensemble

Two fixed prompt candidates are evaluated on development and validation.
Select highest validation accuracy, then development accuracy, then definition order.
Lock the choice before opening test images. Logs are resumable; final metrics
are never overwritten. Previous experiment outputs and Streamlit are untouched.
"""

import csv
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from backend import config
from backend.clean_image import clean_xray
from backend.medgemma import MEDGEMMA_NAME, load_medgemma
from backend.zero_shot_ensemble import (
    CLASSES, MODEL_NAME, PROMPTS, TEXTS, ensemble_answer, load_zero_shot, zero_shot_scores,
)

OUT = config.PROJECT_ROOT / "outputs" / "zero_shot_ensemble"
COUNTS = {"development": 20, "validation": 40, "test": 140}
SOURCE = {"development": "Training", "validation": "Validation", "test": "Testing"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg"}


def patient_key(path):
    # This dataset exposes patient identifiers only in the pneumonia person filenames.
    match = re.match(r"person(\d+)_", path.name)
    return "pneumonia_person_" + match.group(1) if match else None


def experiment_spec():
    sources = [Path(__file__), config.PROJECT_ROOT / "backend/zero_shot_ensemble.py",
               config.PROJECT_ROOT / "backend/medgemma.py", config.PROJECT_ROOT / "backend/clean_image.py"]
    return {"seed": config.SEED, "counts": COUNTS, "classes": list(CLASSES),
            "models": [MODEL_NAME, MEDGEMMA_NAME], "texts": list(TEXTS), "prompts": PROMPTS,
            "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}


def prepare_manifest():
    spec = experiment_spec()
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["spec"] != spec:
            raise RuntimeError("Experiment definition changed; refusing to mix results.")
        return manifest
    rng = random.Random(config.SEED)
    # Keep all previously logged evaluation photos out of the new final test.
    previous = set()
    for log in (config.PROJECT_ROOT / "outputs/prompt_logs").glob("*.csv"):
        with log.open() as handle:
            previous.update(str(Path(row["path"]).resolve()) for row in csv.DictReader(handle))
    parts, seen_hashes, seen_patients = {}, set(), set()
    for part, total in COUNTS.items():
        per_class = []
        for label in CLASSES:
            paths = sorted(p for p in (config.CLEAN_DIR / SOURCE[part] / label).iterdir()
                           if p.suffix.lower() in IMAGE_TYPES)
            rng.shuffle(paths)
            chosen = []
            for path in paths:
                if part == "test" and str(path.resolve()) in previous:
                    continue
                patient = patient_key(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen_hashes or (patient and patient in seen_patients):
                    continue
                chosen.append({"path": str(path.resolve()), "true": label, "sha256": digest})
                seen_hashes.add(digest)
                if patient:
                    seen_patients.add(patient)
                if len(chosen) == total // 2:
                    break
            if len(chosen) != total // 2:
                raise RuntimeError(f"Insufficient distinct {label} cases for {part}")
            per_class.append(chosen)
        parts[part] = [item for pair in zip(*per_class) for item in pair]
    manifest = {"spec": spec, "parts": parts,
                "limitations": "Patient separation verified only where person IDs exist; pretrained model data overlap is unknown."}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def evaluate_rows(rows):
    truth = [r["true"] for r in rows]
    predicted = [r["predicted"] for r in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(CLASSES), zero_division=0)
    return {"n": len(rows), "correct": sum(a == b for a, b in zip(truth, predicted)),
            "accuracy": accuracy_score(truth, predicted),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()), "invalid_answers": predicted.count("Other"),
            "matrix_labels": [*CLASSES, "Other"],
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[*CLASSES, "Other"]).tolist(),
            "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                   "f1": float(f1[i]), "support": int(support[i])}
                          for i, label in enumerate(CLASSES)}}


def run_part(part, versions, cases, zero_shot, medgemma):
    logs, rows_by_version, metrics = {}, {}, {}
    for version in versions:
        log = OUT / f"{part}_{version}.csv"
        rows = list(csv.DictReader(log.open())) if log.exists() else []
        expected = cases[:len(rows)]
        if [(r["path"], r["true"]) for r in rows] != [(r["path"], r["true"]) for r in expected]:
            raise RuntimeError("Log does not match the manifest")
        rows_by_version[version], logs[version] = rows, log
    for number, case in enumerate(cases):
        pending = [v for v in versions if len(rows_by_version[v]) <= number]
        if not pending:
            continue
        path = Path(case["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise RuntimeError(f"Image changed: {path}")
        with Image.open(path) as original:
            image = clean_xray(original.convert("RGB"))
        scores = zero_shot_scores(zero_shot, image)
        for version in pending:
            answer, predicted = ensemble_answer(medgemma, image, scores, version)
            row = {"path": case["path"], "true": case["true"], "answer": answer,
                   "predicted": predicted, "medsiglip_predicted": max(scores, key=scores.get),
                   "normal_similarity": scores["Normal"], "pneumonia_similarity": scores["Pneumonia"]}
            rows_by_version[version].append(row)
            log = logs[version]
            new = not log.exists()
            with log.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            print(f"{part} {version} {number + 1}/{len(cases)}: {predicted}", flush=True)
    for version, rows in rows_by_version.items():
        metrics[version] = evaluate_rows(rows)
        baseline = [dict(r, predicted=r["medsiglip_predicted"]) for r in rows]
        metrics[version]["medsiglip_only"] = evaluate_rows(baseline)
        print(part, version, json.dumps(metrics[version]), flush=True)
    return metrics


def main():
    manifest = prepare_manifest()
    if (OUT / "final_metrics.json").exists():
        print("Final test already completed; existing results:")
        print((OUT / "final_metrics.json").read_text())
        return
    print("Device:", config.DEVICE, "| No probe; no weight training.", flush=True)
    zero_shot = load_zero_shot()
    medgemma = load_medgemma()
    medgemma["model"].requires_grad_(False)
    lock_path = OUT / "selection.json"
    if lock_path.exists():
        selection = json.loads(lock_path.read_text())
    else:
        development = run_part("development", list(PROMPTS), manifest["parts"]["development"], zero_shot, medgemma)
        validation = run_part("validation", list(PROMPTS), manifest["parts"]["validation"], zero_shot, medgemma)
        selected = max(PROMPTS, key=lambda v: (validation[v]["accuracy"], development[v]["accuracy"]))
        selection = {"selected": selected, "rule": "validation accuracy; development accuracy; definition order",
                     "development": development, "validation": validation}
        lock_path.write_text(json.dumps(selection, indent=2))
    print("LOCKED before test:", selection["selected"], flush=True)
    test = run_part("test", [selection["selected"]], manifest["parts"]["test"], zero_shot, medgemma)
    final = {"selected": selection["selected"], "development": selection["development"],
             "validation": selection["validation"], "test": test,
             "scope": "Internal dataset, Normal vs Pneumonia; not an NIH evaluation.",
             "limitations": manifest["limitations"]}
    with (OUT / "final_metrics.json").open("x") as handle:
        json.dump(final, handle, indent=2)
    print("Saved:", OUT / "final_metrics.json", flush=True)


if __name__ == "__main__":
    main()
