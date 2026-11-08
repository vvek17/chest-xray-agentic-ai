"""Compare criteria prompts on 20 dev / 40 validation cases, then NIH 200.

Run: python -m prompt_engineering.run_criteria_ensemble
No fine-tuning. Resume from per-case CSV logs; lock selection before NIH inference.
The previously reviewed 40 NIH cases are reported separately from the other 160.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from backend import config
from backend.clean_image import clean_xray
from backend.medgemma import ask_medgemma, load_medgemma
from backend.zero_shot_ensemble import (
    PROMPTS_V2, PROMPTS_V2_SOFTENED, ensemble_answer, load_zero_shot, parse_label, zero_shot_scores,
)
from prompt_engineering.prompts import PROMPTS as LEGACY_PROMPTS
from prompt_engineering.run_zero_shot_ensemble import evaluate_rows

OUT = config.PROJECT_ROOT / "outputs/criteria_ensemble_v2_softened"
OLD = config.PROJECT_ROOT / "outputs/zero_shot_ensemble"
BASELINE = "baseline_nohint"
CANDIDATES = PROMPTS_V2 | PROMPTS_V2_SOFTENED


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def summarize(rows):
    result = evaluate_rows(rows)
    result["sensitivity"] = result["per_class"]["Pneumonia"]["recall"]
    result["specificity"] = result["per_class"]["Normal"]["recall"]
    n, p, z = result["n"], result["accuracy"], 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    result["accuracy_95pct_wilson"] = [center - half, center + half]
    return result


def prepare_manifest():
    previous = json.loads((OLD / "manifest.json").read_text())
    parts = {part: previous["parts"][part] for part in ["development", "validation"]}
    nih = config.PROJECT_ROOT / "data/external/NIH"
    reviewed = read_csv(config.PROJECT_ROOT / "outputs/nih_zero_shot_ensemble_20_each/truth_table.csv")
    reviewed_ids = {row["image_id"] for row in reviewed}
    parts["nih"] = []
    for row in read_csv(nih / "manifest.csv"):
        path = nih / row["class"] / (row["image_id"] + ".png")
        parts["nih"].append({"path": str(path.resolve()), "true": row["class"],
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "reviewed_before_prompt_design": row["image_id"] in reviewed_ids})
    assert len(parts["nih"]) == 200
    assert sum(row["reviewed_before_prompt_design"] for row in parts["nih"]) == 40
    sources = [Path(__file__), config.PROJECT_ROOT / "backend/zero_shot_ensemble.py",
               config.PROJECT_ROOT / "backend/medgemma.py", config.PROJECT_ROOT / "backend/clean_image.py"]
    manifest = {"prompts": CANDIDATES, "baseline": LEGACY_PROMPTS["2c_v1"],
                "parts": parts, "device": config.DEVICE,
                "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "limitations": "NIH labels are from the manifest. Prompts were designed after reviewing 40 NIH cases; the full 200 are not an untouched test set."}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError("Experiment changed; refusing to mix existing results.")
    else:
        path.write_text(json.dumps(manifest, indent=2))
    return manifest


def reuse_unchanged_logs(manifest):
    """Reuse only exact-prompt, exact-case answers from the interrupted first run."""
    previous_dir = config.PROJECT_ROOT / "outputs/criteria_ensemble_v2"
    source = previous_dir / "manifest.json"
    if not source.exists():
        return
    previous = json.loads(source.read_text())
    for part in ["development", "validation"]:
        if previous["parts"][part] != manifest["parts"][part]:
            raise RuntimeError("Cannot reuse answers for different cases")
        for version in [BASELINE, *PROMPTS_V2]:
            old_prompt = previous["baseline"] if version == BASELINE else previous["prompts"][version]
            new_prompt = manifest["baseline"] if version == BASELINE else manifest["prompts"][version]
            if old_prompt != new_prompt:
                raise RuntimeError("Cannot reuse answers for changed prompts")
            old_path = previous_dir / f"{part}_{version}.csv"
            target = OUT / old_path.name
            if old_path.exists() and not target.exists():
                rows = read_csv(old_path)
                assert all(parse_label(r["answer"]) == r["predicted"] for r in rows)
                target.write_bytes(old_path.read_bytes())
                print("Reused unchanged", part, version, len(rows), "answers", flush=True)


def run_cases(part, versions, cases, models):
    logs = {v: OUT / f"{part}_{v}.csv" for v in versions}
    records = {v: read_csv(logs[v]) for v in versions}
    for version, rows in records.items():
        if [(r["path"], r["true"]) for r in rows] != [(r["path"], r["true"]) for r in cases[:len(rows)]]:
            raise RuntimeError("Existing rows do not match manifest")
    for i, case in enumerate(cases):
        pending = [v for v in versions if len(records[v]) <= i]
        if not pending:
            continue
        path = Path(case["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise RuntimeError(f"Image changed: {path}")
        with Image.open(path) as original:
            image = clean_xray(original.convert("RGB"))
        scores = zero_shot_scores(models["zero_shot"], image)
        for version in pending:
            if version == BASELINE:
                answer = ask_medgemma(models["medgemma"], image, LEGACY_PROMPTS["2c_v1"])
                label = parse_label(answer)
            else:
                answer, label = ensemble_answer(models["medgemma"], image, scores, version)
            row = {"path": str(path), "true": case["true"], "answer": answer, "predicted": label,
                   "medsiglip_predicted": max(scores, key=scores.get),
                   "normal_similarity": scores["Normal"], "pneumonia_similarity": scores["Pneumonia"]}
            new = not logs[version].exists()
            with logs[version].open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            records[version].append(row)
            print(f"{part} {version} {i + 1}/{len(cases)}: {label}", flush=True)
    result = {v: summarize(rows) for v, rows in records.items()}
    for version, metrics in result.items():
        print(part, version, "correct", metrics["correct"], "/", metrics["n"],
              "sensitivity", metrics["sensitivity"], "specificity", metrics["specificity"],
              "Other", metrics["invalid_answers"], flush=True)
    return result


def main():
    manifest = prepare_manifest()
    if (OUT / "final_metrics.json").exists():
        print((OUT / "final_metrics.json").read_text())
        return
    reuse_unchanged_logs(manifest)
    models = {"zero_shot": load_zero_shot(), "medgemma": load_medgemma()}
    models["medgemma"]["model"].requires_grad_(False)
    lock = OUT / "selection.json"
    if lock.exists():
        selection = json.loads(lock.read_text())
    else:
        versions = [BASELINE, *CANDIDATES]
        development = run_cases("development", versions, manifest["parts"]["development"], models)
        validation = run_cases("validation", versions, manifest["parts"]["validation"], models)
        # Sensitivity guard is decided in advance, not from NIH outcomes.
        eligible = [v for v in versions if validation[v]["sensitivity"] >= 0.75]
        candidates = eligible or versions
        selected = max(candidates, key=lambda v: (validation[v]["accuracy"], development[v]["accuracy"]))
        old_comparisons = {}
        for part in ["development", "validation"]:
            old_comparisons[part] = {}
            for version in ["label_hint", "similarity_hint"]:
                rows = read_csv(OLD / f"{part}_{version}.csv")
                if [r["path"] for r in rows] != [r["path"] for r in manifest["parts"][part]]:
                    raise RuntimeError("Old comparison used different images")
                old_comparisons[part][version] = summarize(rows)
        selection = {"selected": selected, "rule": "Prefer validation sensitivity >=75%; then validation accuracy, development accuracy, definition order.",
                     "sensitivity_guard_met": bool(eligible), "development": development,
                     "validation": validation, "old_comparisons": old_comparisons}
        lock.write_text(json.dumps(selection, indent=2))
    print("LOCKED before NIH:", selection["selected"], flush=True)
    nih = run_cases("nih", [selection["selected"]], manifest["parts"]["nih"], models)
    rows = read_csv(OUT / f"nih_{selection['selected']}.csv")
    groups = {}
    for reviewed, name in [(True, "previously_reviewed_40"), (False, "remaining_160")]:
        groups[name] = summarize([row for case, row in zip(manifest["parts"]["nih"], rows)
                                  if case["reviewed_before_prompt_design"] == reviewed])
    truth = [dict(row, reviewed_before_prompt_design=case["reviewed_before_prompt_design"])
             for case, row in zip(manifest["parts"]["nih"], rows)]
    with (OUT / "nih_truth_table.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(truth[0]))
        writer.writeheader()
        writer.writerows(truth)
    final = {"selection": selection, "nih": nih, "nih_groups": groups,
             "limitations": manifest["limitations"]}
    with (OUT / "final_metrics.json").open("x") as handle:
        json.dump(final, handle, indent=2)
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
