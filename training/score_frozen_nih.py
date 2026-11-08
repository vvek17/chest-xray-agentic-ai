"""Step 1 only: score cached, patient-disjoint NIH dev and validation images.

Run: python -m training.score_frozen_nih
Never loads a final-test image into either model. No prompts or weights change.
"""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image
import torch

from backend import config
from backend.clean_image import clean_xray
from backend.frozen_scores import PROMPT, label_token_ids, medgemma_score_and_answer
from backend.medgemma import MEDGEMMA_NAME, load_medgemma
from backend.zero_shot_ensemble import MODEL_NAME, TEXTS, load_zero_shot, zero_shot_scores

ROOT = config.PROJECT_ROOT
MANIFESTS = ROOT / "data/external/NIH_development/manifests"
OUT = ROOT / "outputs/frozen_nih_threshold"
SPLITS = ("dev", "validation")
CSV_FIELDS = [
    "image_id", "patient_id", "path", "true_label", "image_sha256",
    "sim_normal", "sim_pneumonia", "siglip_score",
    "medgemma_score", "normal_first_token_logit", "pneumonia_first_token_logit",
    "normal_first_token_logprob", "pneumonia_first_token_logprob",
    "normal_first_token_id", "pneumonia_first_token_id", "raw_argmax_token_id", "raw_argmax_token_text",
    "medgemma_raw_answer", "medgemma_greedy_answer", "medgemma_score_sign_label", "sign_agrees_greedy",
    "parse_failure", "siglip_nonfinite", "medgemma_nonfinite", "nonfinite_vocab_logits",
    "siglip_status", "medgemma_status", "siglip_error", "medgemma_error",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Preserve non-finite values explicitly; never silently remove a case.
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def is_finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_cases():
    # Validate metadata without opening the final images or importing their old scores.
    plan = json.loads((MANIFESTS / "split_plan.json").read_text())
    final_rows = read_csv(MANIFESTS / "final_test.csv")
    assert final_rows == plan["splits"]["final_test"]
    final_ids = {r["image_id"] for r in final_rows}
    final_patients = {int(r["patient_id"]) for r in final_rows}
    parts = {}
    seen_ids, seen_patients = set(), set()
    for split in SPLITS:
        rows = read_csv(MANIFESTS / f"{split}.csv")
        assert rows == plan["splits"][split]
        assert Counter(r["true_label"] for r in rows) == {"Normal": 100, "Pneumonia": 100}
        ids = {r["image_id"] for r in rows}
        patients = {int(r["patient_id"]) for r in rows}
        assert len(ids) == len(rows) and not ids & (seen_ids | final_ids)
        assert not patients & (seen_patients | final_patients)
        seen_ids |= ids
        seen_patients |= patients
        parts[split] = rows
    return parts


def cache_record(split, case, model):
    path = OUT / "cache" / split / f"{case['image_id']}_{model}.json"
    if not path.exists():
        return path, None
    record = json.loads(path.read_text())
    assert record["image_id"] == case["image_id"] and record["image_sha256"] == digest(Path(case["path"]))
    return path, record


def merged_row(case, siglip, medgemma, tokens):
    row = dict(case)
    row.update({"image_sha256": siglip["image_sha256"]})
    row.update({k: v for k, v in siglip.items() if k not in ["image_id", "status", "error"]})
    row.update({k: v for k, v in medgemma.items() if k not in ["image_id", "status", "error"]})
    row["siglip_status"], row["medgemma_status"] = siglip["status"], medgemma["status"]
    row["siglip_error"], row["medgemma_error"] = siglip.get("error", ""), medgemma.get("error", "")
    for label, token in tokens["first_token_ids"].items():
        row[label.lower() + "_first_token_id"] = token
    row["siglip_nonfinite"] = not is_finite(row.get("siglip_score"))
    row["medgemma_nonfinite"] = not is_finite(row.get("medgemma_score"))
    row["parse_failure"] = row.get("medgemma_greedy_answer") not in ["Normal", "Pneumonia"]
    score = float(row.get("medgemma_score", "nan"))
    row["medgemma_score_sign_label"] = ("Pneumonia" if score > 0 else "Normal" if score < 0
                                         else "Tie" if score == 0 else "Nonfinite")
    row["sign_agrees_greedy"] = row["medgemma_score_sign_label"] == row.get("medgemma_greedy_answer")
    return {key: row.get(key, "") for key in CSV_FIELDS}


def export_scores(parts, tokens):
    summary = {}
    for split, cases in parts.items():
        rows = []
        for case in cases:
            _, a = cache_record(split, case, "siglip")
            _, b = cache_record(split, case, "medgemma")
            if a is not None and b is not None:
                rows.append(merged_row(case, a, b, tokens))
        path = OUT / f"{split}_scores.csv"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
        summary[split] = {"expected": len(cases), "completed": len(rows),
                          "class_counts": dict(Counter(r["true_label"] for r in rows)),
                          "parse_failures": sum(r["parse_failure"] for r in rows),
                          "siglip_nonfinite": sum(r["siglip_nonfinite"] for r in rows),
                          "medgemma_nonfinite": sum(r["medgemma_nonfinite"] for r in rows),
                          "score_ties": sum(r["medgemma_score_sign_label"] == "Tie" for r in rows),
                          "sign_agrees_greedy": sum(r["sign_agrees_greedy"] for r in rows),
                          "agreement_denominator": len(rows),
                          "errors": sum(r["siglip_status"] != "ok" or r["medgemma_status"] != "ok" for r in rows)}
    atomic_json(OUT / "step1_summary.json", summary)
    return summary


def main():
    parts = load_cases()
    OUT.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), ROOT / "backend/frozen_scores.py", ROOT / "backend/medgemma.py",
                    ROOT / "backend/zero_shot_ensemble.py", ROOT / "backend/clean_image.py"]
    spec = {"step": 1, "seed": 42, "prompt": PROMPT, "device": config.DEVICE,
            "models": [MODEL_NAME, MEDGEMMA_NAME], "medsiglip_texts": list(TEXTS),
            "split_hashes": {split: digest(MANIFESTS / f"{split}.csv") for split in SPLITS},
            "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in source_files},
            "score_definition": "Full-vocabulary log_softmax at last prompt position, Pneumonia first token minus Normal first token",
            "capture": "First model forward hook inside existing deterministic ask_medgemma; same prefill for score and greedy output",
            "final_test_inference": False}
    frozen = OUT / "step1_protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != spec:
            raise RuntimeError("Frozen scoring definition changed; refusing to mix cached results")
    else:
        atomic_json(frozen, spec)
    token_path = OUT / "label_tokens.json"
    saved_tokens = json.loads(token_path.read_text()) if token_path.exists() else None
    if saved_tokens:
        summary = export_scores(parts, saved_tokens)
        if all(s["completed"] == s["expected"] for s in summary.values()):
            print("All scores cached; no model loaded.", json.dumps(summary), flush=True)
            return
    torch.manual_seed(42)
    zero_shot, medgemma = load_zero_shot(), load_medgemma()
    medgemma["model"].requires_grad_(False)
    tokens = label_token_ids(medgemma["processor"].tokenizer)
    print("LABEL TOKENS:", json.dumps(tokens), flush=True)
    if saved_tokens:
        assert tokens == saved_tokens, "Tokenizer changed"
    else:
        atomic_json(token_path, tokens)
    runtime = {"device": config.DEVICE,
               "medsiglip_revision": getattr(zero_shot["model"].config, "_commit_hash", None),
               "medgemma_revision": getattr(medgemma["model"].config, "_commit_hash", None),
               "medsiglip_dtype": str(zero_shot["dtype"]), "medgemma_dtype": str(medgemma["model"].dtype)}
    runtime_path = OUT / "model_runtime.json"
    if runtime_path.exists():
        assert json.loads(runtime_path.read_text()) == runtime, "Model runtime/revision changed"
    else:
        atomic_json(runtime_path, runtime)
    for split, cases in parts.items():
        for index, case in enumerate(cases, 1):
            paths_and_records = {name: cache_record(split, case, name) for name in ["siglip", "medgemma"]}
            if all(record is not None for path, record in paths_and_records.values()):
                continue
            with Image.open(case["path"]) as original:
                cleaned = clean_xray(original.convert("RGB"))
            image_hash = digest(Path(case["path"]))
            for name, (path, record) in paths_and_records.items():
                if record is not None:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                record = {"image_id": case["image_id"], "image_sha256": image_hash}
                try:
                    if name == "siglip":
                        similarities = zero_shot_scores(zero_shot, cleaned)
                        record.update(sim_normal=similarities["Normal"], sim_pneumonia=similarities["Pneumonia"],
                                      siglip_score=similarities["Pneumonia"] - similarities["Normal"])
                    else:
                        record.update(medgemma_score_and_answer(medgemma, cleaned, tokens))
                    record["status"] = "ok"
                except Exception as error:
                    record.update(status="error", error=f"{type(error).__name__}: {error}")
                    atomic_json(path, json_safe(record))
                    export_scores(parts, tokens)
                    raise
                atomic_json(path, json_safe(record))
            print(f"{split}: {index}/{len(cases)} scored and cached", flush=True)
            if index % 10 == 0:
                export_scores(parts, tokens)
    summary = export_scores(parts, tokens)
    print("STEP 1 COMPLETE:", json.dumps(summary), flush=True)
    print("No final-test inference. No thresholds or validation performance metrics computed.", flush=True)


if __name__ == "__main__":
    main()
