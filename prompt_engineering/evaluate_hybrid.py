"""Evaluate a frozen four-class probe hint followed by frozen MedGemma.

How to run (from chest-xray-pneumonia):
    python -m prompt_engineering.evaluate_hybrid --stage prepare
    python -m prompt_engineering.evaluate_hybrid --stage dev
    python -m prompt_engineering.evaluate_hybrid --stage val
    python -m prompt_engineering.evaluate_hybrid --stage freeze
    python -m prompt_engineering.evaluate_hybrid --stage test

No inference runs during prepare. Test inference requires a frozen configuration.
No function needs manual replacement: get_hint uses backend.predict.predict.
Patient independence is unverified. Research only; not a medical diagnosis.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import try_to_load_from_cache

from backend import config
from backend.clean_image import clean_xray
from backend.medgemma import MEDGEMMA_NAME, ask_medgemma, load_medgemma, read_answer
from backend.predict import MEDSIGLIP_NAME, PROBE_4_CLASSES, load_classifier, predict
from prompt_engineering.evaluate_medgemma import metrics
from prompt_engineering.photos import LOG_DIR, get_examples, load_split
from prompt_engineering.prompts import PROMPTS

CLASSES = ["Normal", "Pneumonia"]
SIZES = {"dev": 20, "val": 40, "test": 70}
OUT = config.PROJECT_ROOT / "outputs" / "hybrid_original_testing_20_40_70"
SOURCE = config.RAW_DIR / "Testing"
IMAGE_TYPES = [".png", ".jpg", ".jpeg"]
HYBRIDS = ["hybrid_confidence_v1", "hybrid_label_v1", "hybrid_verify_v1"]
ARMS = ["medsiglip", "medgemma_zero_shot", "2c_fs1"] + HYBRIDS
PROMPT_VERSIONS = ["2c_v1", "2c_fs1"] + HYBRIDS
FIELDS = ["path", "sha256", "true", "arm", "answer", "predicted", "hint_label",
          "hint_score", "normal_score", "pneumonia_score", "four_class_scores"]


# Hash file contents so renamed copies are still recognized as the same image.
def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Save readable JSON so the exact experiment can be inspected later.
def save_json(path, value):
    path.write_text(json.dumps(value, indent=2))


# Read saved records without changing any previous experiment files.
def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# Find original counterparts because older logs may name processed copies.
def original_path(path):
    path = Path(path).resolve()
    if path.is_relative_to(config.CLEAN_DIR.resolve()):
        relative = path.relative_to(config.CLEAN_DIR.resolve())
        return config.RAW_DIR / relative
    return path


# Find image paths inside old manifests, including few-shot examples and reserved cases.
def manifest_images(value):
    paths = []
    if isinstance(value, dict):
        for item in value.values():
            paths.extend(manifest_images(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(manifest_images(item))
    elif isinstance(value, str):
        if Path(value).suffix.lower() in IMAGE_TYPES:
            paths.append(Path(value))
    return paths


# Scan actual answer logs, not split inventories that list unused images.
def previous_image_paths():
    paths = set()
    sources = []
    outputs = config.PROJECT_ROOT / "outputs"
    for log in sorted(outputs.rglob("*.csv")):
        if log.is_relative_to(OUT):
            continue
        rows = read_csv(log)
        if not rows or "path" not in rows[0]:
            continue
        has_answers = False
        for column in rows[0]:
            if "answer" in column or "medgemma" in column:
                has_answers = True
        if not has_answers and log.parent != LOG_DIR:
            continue
        sources.append(log)
        for row in rows:
            paths.add(Path(row["path"]))
    # Conservatively exclude reserved old manifest cases as well as completed cases.
    for manifest in sorted(outputs.rglob("manifest.json")):
        if manifest.is_relative_to(OUT):
            continue
        sources.append(manifest)
        paths.update(manifest_images(json.loads(manifest.read_text())))
    # Legacy few-shot images were not always saved in the per-image answer logs.
    for path, label in get_examples(CLASSES, 1):
        paths.add(path)
    # Read the existing split helpers to map legacy paths, never to repartition data.
    for part in ["train", "val", "test"]:
        for path, label in load_split(part):
            if path in paths:
                paths.add(original_path(path))
    return paths, sources


# Exclude every recoverable historical image and fail if an old input is missing.
def history_hashes():
    paths, sources = previous_image_paths()
    hashes = set()
    missing = []
    for path in sorted(paths):
        original = original_path(path)
        if not path.exists() and not original.exists():
            missing.append(str(path))
            continue
        if path.exists():
            hashes.add(file_hash(path))
        if original.exists():
            hashes.add(file_hash(original))
    if missing:
        raise RuntimeError("Cannot hash earlier inputs; restore these files: " + str(missing[:10]))
    return hashes, sources


# Also exclude Training/Validation copies so a Testing duplicate cannot leak probe data.
def fitted_data_hashes():
    hashes = set()
    for part in ["Training", "Validation"]:
        for label in config.CLASS_NAMES:
            for path in sorted((config.RAW_DIR / part / label).iterdir()):
                if path.suffix.lower() in IMAGE_TYPES:
                    hashes.add(file_hash(path))
    return hashes


# Pin local cached model commits rather than silently downloading a changed main revision.
def cached_revision(name):
    # Read a cached config file; optional repository docs need not be downloaded.
    cached = try_to_load_from_cache(name, "config.json")
    if not isinstance(cached, str):
        raise RuntimeError("Model config is not cached locally: " + name)
    return Path(cached).parent.name


# Record prompts, code, weights and settings; changed experiments must use a new OUT.
def experiment_spec():
    prompt_settings = {}
    for version in PROMPT_VERSIONS:
        prompt_settings[version] = PROMPTS[version]
    source_names = ["prompt_engineering/evaluate_hybrid.py", "prompt_engineering/photos.py",
                    "prompt_engineering/evaluate_medgemma.py", "backend/predict.py",
                    "backend/medgemma.py", "backend/clean_image.py", "backend/config.py"]
    source_hashes = {}
    for name in source_names:
        source_hashes[name] = file_hash(config.PROJECT_ROOT / name)
    versions = {}
    for package in ["torch", "transformers", "numpy", "scikit-learn", "pillow"]:
        versions[package] = importlib.metadata.version(package)
    return {"classes": CLASSES, "sizes_per_class": SIZES, "seed": config.SEED,
            "source": str(SOURCE.resolve()), "prompts": prompt_settings, "arms": ARMS,
            "probe_path": str(PROBE_4_CLASSES.resolve()), "probe_sha256": file_hash(PROBE_4_CLASSES),
            "medsiglip_revision": cached_revision(MEDSIGLIP_NAME),
            "medgemma_revision": cached_revision(MEDGEMMA_NAME), "device": config.DEVICE,
            "packages": versions, "source_hashes": source_hashes,
            "generation": {"do_sample": False, "max_tokens": 10},
            "hint_rule": "argmax of Normal and Pneumonia sigmoid scores; no renormalization",
            "hybrid_examples": "same two examples as 2c_fs1; baseline questions precede hints",
            "selection_rule": "validation macro F1, then pneumonia recall, then specificity, then listed hybrid order",
            "limitations": "Patient independence and pretraining overlap are unverified. Research only."}


# Keep enough unused original Testing images for every split and two context examples.
def eligible_images(blocked):
    eligible = []
    seen = set(blocked)
    rng = random.Random(config.SEED)
    for label in CLASSES:
        paths = []
        for path in sorted((SOURCE / label).iterdir()):
            if path.suffix.lower() in IMAGE_TYPES:
                paths.append(path)
        rng.shuffle(paths)
        for path in paths:
            digest = file_hash(path)
            if digest in seen:
                continue
            seen.add(digest)
            eligible.append((path, label))
    return eligible


# Store labels and hashes for an auditable, stable selection, without running a model.
def image_record(path, label):
    return {"path": str(path.resolve()), "true": label, "sha256": file_hash(path)}


# Select disjoint fresh context examples and dev/val/test cases from original Testing only.
def create_manifest():
    excluded, sources = history_hashes()
    fitted = fitted_data_hashes()
    eligible = eligible_images(excluded | fitted)
    examples = get_examples(CLASSES, 1, photos=eligible)
    used = set()
    example_records = []
    for path, label in examples:
        used.add(file_hash(path))
        example_records.append(image_record(path, label))
    parts = {}
    for part, count in SIZES.items():
        chosen = []
        for label in CLASSES:
            number = 0
            for path, truth in eligible:
                digest = file_hash(path)
                if truth != label or digest in used:
                    continue
                chosen.append(image_record(path, truth))
                used.add(digest)
                number += 1
                if number == count:
                    break
            if number != count:
                raise RuntimeError(f"Not enough unused original Testing images for {part}/{label}; need {count}, found {number}.")
        parts[part] = chosen
    source_records = {}
    for path in sources:
        source_records[str(path)] = file_hash(path)
    return {"spec": experiment_spec(), "examples": example_records, "splits": parts,
            "excluded_sha256": sorted(excluded), "fitted_data_sha256": sorted(fitted),
            "history_sources": source_records, "test_status": "reserved; no inference during preparation"}


# Refuse a changed image, source location, count, duplicate, or known historical input.
def verify_manifest(manifest):
    blocked = set(manifest["excluded_sha256"] + manifest["fitted_data_sha256"])
    seen = set(blocked)
    groups = dict(manifest["splits"])
    groups["examples"] = manifest["examples"]
    for part, cases in groups.items():
        expected = 1 if part == "examples" else SIZES[part]
        counts = {"Normal": 0, "Pneumonia": 0}
        for case in cases:
            path = Path(case["path"])
            if path.parent != (SOURCE / case["true"]).resolve():
                raise RuntimeError("Case is not from original Testing: " + str(path))
            digest = file_hash(path)
            if digest != case["sha256"] or digest in seen:
                raise RuntimeError("Changed, duplicate, or excluded image: " + str(path))
            seen.add(digest)
            counts[case["true"]] += 1
        if counts != {"Normal": expected, "Pneumonia": expected}:
            raise RuntimeError("Incorrect split counts: " + part)


# Reuse the saved selection only when the entire experiment still matches.
def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["spec"] != experiment_spec():
            raise RuntimeError("Experiment changed. Restore it or choose a new OUT folder.")
    else:
        manifest = create_manifest()
        verify_manifest(manifest)
        save_json(path, manifest)
    verify_manifest(manifest)
    return manifest


# A new external MedGemma run must not quietly consume our reserved test images.
def verify_history(manifest):
    current, sources = history_hashes()
    old = set(manifest["excluded_sha256"])
    newly_used = current - old
    for cases in manifest["splits"].values():
        for case in cases:
            if case["sha256"] in newly_used:
                raise RuntimeError("A reserved image was used by another run: " + case["path"])
    for case in manifest["examples"]:
        if case["sha256"] in newly_used:
            raise RuntimeError("A context image was used by another run: " + case["path"])


# Restrict the trained FOUR-class probe to the same two choices in every comparison.
def get_hint(classifier, path):
    scores = predict(classifier, path)
    for label in CLASSES:
        if label not in scores or not math.isfinite(scores[label]):
            raise ValueError("Missing or non-finite probe score: " + label)
    label = "Normal"
    if scores["Pneumonia"] > scores["Normal"]:
        label = "Pneumonia"
    return {"label": label, "score": scores[label], "scores": scores}


# Format only the target question; few-shot examples retain the unchanged baseline question.
def target_prompt(arm, hint):
    version = "2c_v1" if arm == "medgemma_zero_shot" else arm
    prompt = dict(PROMPTS[version])
    if arm in HYBRIDS:
        prompt["user"] = prompt["user"].format(hint_label=hint["label"], hint_score=hint["score"])
        prompt["example_user"] = PROMPTS["2c_fs1"]["user"]
    return prompt


# Never turn a verbose or ambiguous answer into a valid prediction by substring matching.
def strict_answer(answer):
    if answer.strip() not in CLASSES:
        return "Other"
    return read_answer(answer.strip(), CLASSES)


# Keep resumable logs in exact case/arm order so cached answers cannot be misaligned.
def checked_rows(part, manifest):
    rows = read_csv(OUT / f"{part}_predictions.csv")
    expected = []
    for case in manifest["splits"][part]:
        for arm in ARMS:
            expected.append((case["path"], case["sha256"], case["true"], arm))
    if len(rows) > len(expected):
        raise RuntimeError("Prediction log has too many rows.")
    for index, row in enumerate(rows):
        actual = (row["path"], row["sha256"], row["true"], row["arm"])
        if actual != expected[index]:
            raise RuntimeError("Prediction log does not match the manifest.")
        if row["predicted"] != strict_answer(row["answer"]):
            raise RuntimeError("Saved label does not match the exact raw answer.")
    return rows


# Each saved row includes the same binary hint plus all four original probe scores.
def prediction_row(case, arm, answer, hint):
    return {"path": case["path"], "sha256": case["sha256"], "true": case["true"],
            "arm": arm, "answer": answer, "predicted": strict_answer(answer),
            "hint_label": hint["label"], "hint_score": hint["score"],
            "normal_score": hint["scores"]["Normal"], "pneumonia_score": hint["scores"]["Pneumonia"],
            "four_class_scores": json.dumps(hint["scores"], sort_keys=True)}


# Run all six arms on each case, saving every answer immediately for safe resumption.
def run_split(part, manifest):
    rows = checked_rows(part, manifest)
    total = len(manifest["splits"][part]) * len(ARMS)
    if len(rows) == total:
        return rows
    classifier = load_classifier(PROBE_4_CLASSES, revision=manifest["spec"]["medsiglip_revision"])
    medgemma = load_medgemma(revision=manifest["spec"]["medgemma_revision"])
    medgemma["model"].requires_grad_(False)
    examples = []
    for case in manifest["examples"]:
        # Apply the same cleaner to target images and context examples in every arm.
        examples.append((clean_xray(case["path"]), case["true"]))
    log = OUT / f"{part}_predictions.csv"
    with log.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        position = 0
        for case in manifest["splits"][part]:
            hint = None
            image = None
            for arm in ARMS:
                position += 1
                if position <= len(rows):
                    continue
                if hint is None:
                    hint = get_hint(classifier, case["path"])
                    image = clean_xray(case["path"])
                if arm == "medsiglip":
                    answer = hint["label"]
                else:
                    prompt = target_prompt(arm, hint)
                    context = [] if arm == "medgemma_zero_shot" else examples
                    answer = ask_medgemma(medgemma, image, prompt, context)
                row = prediction_row(case, arm, answer, hint)
                writer.writerow(row)
                handle.flush()
                print(f"{part}: {position}/{total} answers ({arm})", flush=True)
    return checked_rows(part, manifest)


# Use null, rather than an invented zero rate, when a denominator is empty.
def rate(number, total):
    if total == 0:
        return None
    return number / total


# Count whether the hint was followed, corrected, or damaged, including invalid answers.
def hint_metrics(rows):
    followed = 0
    fixed = 0
    broken = 0
    correct_hints = 0
    wrong_hints = 0
    for row in rows:
        if row["predicted"] == row["hint_label"]:
            followed += 1
        if row["hint_label"] == row["true"]:
            correct_hints += 1
            if row["predicted"] != row["true"]:
                broken += 1
        else:
            wrong_hints += 1
            if row["predicted"] == row["true"]:
                fixed += 1
    return {"hint_follow_count": followed, "hint_follow_rate": rate(followed, len(rows)),
            "wrong_hints": wrong_hints, "medsiglip_errors_fixed": fixed,
            "error_fix_rate": rate(fixed, wrong_hints), "correct_hints": correct_hints,
            "correct_hints_broken": broken, "correct_hint_break_rate": rate(broken, correct_hints)}


# Compute the existing classification metrics and add the requested hint diagnostics.
def split_metrics(rows):
    summary = {}
    for arm in ARMS:
        arm_rows = []
        for row in rows:
            if row["arm"] == arm:
                arm_rows.append(row)
        result = metrics(arm_rows)
        result["invalid_outputs"] = result["unparsed_answers"]
        result.update(hint_metrics(arm_rows))
        result["hint_was_shown"] = arm in HYBRIDS
        summary[arm] = result
    return summary


# Require every arm to finish on every image before reporting or selecting a winner.
def complete_metrics(part, manifest):
    rows = checked_rows(part, manifest)
    expected = len(manifest["splits"][part]) * len(ARMS)
    if len(rows) != expected:
        raise RuntimeError("Complete this split before proceeding: " + part)
    return split_metrics(rows)


# Make one six-panel confusion figure per split, including an invalid-output column.
def save_confusion(part, summary):
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for index, arm in enumerate(ARMS):
        axis = axes.flat[index]
        matrix = summary[arm]["confusion_matrix"]
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(arm)
        axis.set_xticks(range(3), ["Normal", "Pneumonia", "Other"])
        axis.set_yticks(range(3), ["Normal", "Pneumonia", "Other"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                axis.text(j, i, str(matrix[i][j]), ha="center", va="center")
    figure.suptitle(f"Hybrid comparison: {part}")
    figure.tight_layout()
    figure.savefig(OUT / f"{part}_confusion.png", dpi=160)
    plt.close(figure)


# Update the readable report using completed splits only; test is never inferred here.
def write_report(manifest):
    lines = ["# Frozen hybrid evaluation", "", "Research only. Patient independence and pretraining overlap are unverified.",
             "Original Testing images only. Exact SHA-256 duplicates and recorded earlier MedGemma inputs are excluded.",
             "Exclusion is limited to available logs/manifests; unrecorded historical use cannot be established.",
             "All three hybrids share the same two few-shot examples as 2c_fs1. The probe hint is binary argmax, not the four-class winner.",
             "Hint confidence means the uncalibrated winning sigmoid score, without renormalizing the two scores.",
             "Invalid outputs count as errors and as broken correct hints where applicable.",
             "For no-hint baselines, hint-follow metrics mean agreement only, not exposure to a hint.",
             "Rates use all cases for hint following, wrong hints for fixes, and correct hints for breaks. Null means no eligible cases.",
             "ROC-AUC/calibration are not reported for label-only MedGemma answers.", ""]
    lock = OUT / "selection.json"
    if lock.exists():
        lines.append("Frozen primary hybrid: " + json.loads(lock.read_text())["selected_hybrid"])
        lines.append("All six test comparisons were predeclared; do not choose a new winner using test results.")
    for part in SIZES:
        rows = checked_rows(part, manifest)
        if len(rows) != len(manifest["splits"][part]) * len(ARMS):
            continue
        summary = split_metrics(rows)
        save_json(OUT / f"{part}_metrics.json", summary)
        save_confusion(part, summary)
        lines += ["", "## " + part, "", "| Arm | Accuracy | Macro F1 | MCC | Invalid | Hint follow | Fixed | Broken |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm, result in summary.items():
            lines.append(f"| {arm} | {result['accuracy']:.4f} | {result['macro_f1']:.4f} | {result['mcc']:.4f} | {result['invalid_outputs']} | {result['hint_follow_rate']:.4f} | {result['medsiglip_errors_fixed']} | {result['correct_hints_broken']} |")
        for arm, result in summary.items():
            lines += ["", "### " + arm, "", "| Class | Precision | Recall | Specificity | F1 | Support |",
                      "|---|---:|---:|---:|---:|---:|"]
            for label, values in result["per_class"].items():
                lines.append(f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | {values['specificity']:.4f} | {values['f1-score']:.4f} | {values['support']:.0f} |")
            lines += ["", "Rows = true; columns = predicted. Order: Normal, Pneumonia, Other.", "", "```", json.dumps(result["confusion_matrix"]), "```"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


# Freeze the primary hybrid using validation only, before any test inference exists.
def freeze(manifest):
    complete_metrics("dev", manifest)
    validation = complete_metrics("val", manifest)
    if read_csv(OUT / "test_predictions.csv"):
        raise RuntimeError("Test inference has already started; cannot select a new configuration.")
    best = HYBRIDS[0]
    best_key = None
    for arm in HYBRIDS:
        result = validation[arm]
        pneumonia = result["per_class"]["Pneumonia"]
        key = (result["macro_f1"], pneumonia["recall"], pneumonia["specificity"])
        if best_key is None or key > best_key:
            best = arm
            best_key = key
    selection = {"selected_hybrid": best, "manifest_sha256": file_hash(OUT / "manifest.json"),
                 "dev_predictions_sha256": file_hash(OUT / "dev_predictions.csv"),
                 "val_predictions_sha256": file_hash(OUT / "val_predictions.csv"),
                 "all_test_arms_predeclared": ARMS, "selection_rule": manifest["spec"]["selection_rule"]}
    path = OUT / "selection.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise RuntimeError("A different configuration is already frozen.")
    save_json(path, selection)


# Test is inaccessible unless selection, prompts, probe, code, images and prior results match.
def require_frozen(manifest):
    path = OUT / "selection.json"
    if not path.exists():
        raise RuntimeError("Run --stage freeze before --stage test.")
    selection = json.loads(path.read_text())
    if selection["manifest_sha256"] != file_hash(OUT / "manifest.json"):
        raise RuntimeError("Manifest changed after freezing.")
    for part in ["dev", "val"]:
        if selection[part + "_predictions_sha256"] != file_hash(OUT / f"{part}_predictions.csv"):
            raise RuntimeError("Development/validation predictions changed after freezing.")
        complete_metrics(part, manifest)
    if selection["selected_hybrid"] not in HYBRIDS or selection["all_test_arms_predeclared"] != ARMS:
        raise RuntimeError("Frozen arm configuration changed.")


# Expose explicit stages so preparing data can never accidentally evaluate the test set.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "dev", "val", "freeze", "test", "report"], default="prepare")
    args = parser.parse_args()
    manifest = prepare()
    verify_history(manifest)
    if args.stage in ["dev", "val"]:
        if (OUT / "selection.json").exists():
            raise RuntimeError("Configuration frozen; dev/val inference is closed.")
        if args.stage == "val":
            complete_metrics("dev", manifest)
        run_split(args.stage, manifest)
    elif args.stage == "freeze":
        freeze(manifest)
    elif args.stage == "test":
        require_frozen(manifest)
        run_split("test", manifest)
    write_report(manifest)
    print("Stage complete: " + args.stage)
    print("Outputs: " + str(OUT))


if __name__ == "__main__":
    main()
