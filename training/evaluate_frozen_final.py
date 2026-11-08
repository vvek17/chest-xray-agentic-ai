"""Authorized Step 6 only. Score the protected final set once and report all methods.

python -m training.evaluate_frozen_final
Uses frozen dev thresholds/scaling and existing scoring functions unchanged.
Successful per-model results are cached; completed final results are never overwritten.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score

from backend import config
from backend.clean_image import clean_xray
from backend.frozen_scores import PROMPT, label_token_ids, medgemma_score_and_answer
from backend.medgemma import load_medgemma
from backend.zero_shot_ensemble import load_zero_shot, zero_shot_scores
from training.assert_nih_splits import check_splits
from training.dev_frozen_auc import METHODS, RESAMPLES, SEED
from training.score_frozen_nih import CSV_FIELDS, MANIFESTS, OUT, ROOT, atomic_json, cache_record, json_safe, merged_row
from training.validate_frozen_thresholds import binary_metrics, bootstrap_auc, checksum, exact_mcnemar


def export_final_scores(cases, tokens):
    rows = []
    for case in cases:
        _, siglip = cache_record("final_test", case, "siglip")
        _, medgemma = cache_record("final_test", case, "medgemma")
        if siglip is not None and medgemma is not None:
            rows.append(merged_row(case, siglip, medgemma, tokens))
    target = OUT / "final_test_scores.csv"
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return rows


def main():
    final_path = OUT / "step6_final_test.json"
    thresholds_path = OUT / "thresholds.json"
    threshold_hash = checksum(thresholds_path)
    if final_path.exists():
        existing = json.loads(final_path.read_text())
        assert existing["thresholds_sha256"] == threshold_hash
        assert existing["scores_sha256"] == checksum(OUT / "final_test_scores.csv")
        assert existing["predictions_sha256"] == checksum(OUT / "final_test_predictions.csv")
        print("Final evaluation already complete; no model loaded, no reevaluation.")
        print(json.dumps(existing, indent=2))
        return
    check_splits()
    score_protocol = json.loads((OUT / "step1_protocol.json").read_text())
    for name, digest in score_protocol["source_hashes"].items():
        assert checksum(ROOT / name) == digest, f"Frozen scoring code changed: {name}"
    assert score_protocol["prompt"] == PROMPT
    assert config.DEVICE == score_protocol["device"], "Device changed from cached dev/validation scoring"
    frozen = json.loads(thresholds_path.read_text())
    assert checksum(ROOT / "training/validate_frozen_thresholds.py") == frozen["evaluator_source_sha256"]
    validation = json.loads((OUT / "step4_validation.json").read_text())
    assert validation["thresholds_sha256"] == threshold_hash
    assert checksum(OUT / "dev_scores.csv") == frozen["dev_scores_sha256"]
    assert checksum(OUT / "validation_scores.csv") == validation["validation_scores_sha256"]
    with (MANIFESTS / "final_test.csv").open(newline="") as handle:
        cases = list(csv.DictReader(handle))
    assert len(cases) == len({r["image_id"] for r in cases}) == 200
    assert Counter(r["true_label"] for r in cases) == {"Normal": 100, "Pneumonia": 100}
    tokens = json.loads((OUT / "label_tokens.json").read_text())
    runtime = json.loads((OUT / "model_runtime.json").read_text())
    final_protocol = {"step": 6, "methods": list(METHODS), "positive_class": "Pneumonia",
                      "validation_selected_winner": validation["winner"],
                      "thresholds_sha256": threshold_hash, "thresholds": frozen["methods"],
                      "dev_zscore": frozen["dev_zscore"], "prompt": PROMPT,
                      "final_manifest_sha256": checksum(MANIFESTS / "final_test.csv"),
                      "step1_protocol_sha256": checksum(OUT / "step1_protocol.json"),
                      "validation_report_sha256": checksum(OUT / "step4_validation.json"),
                      "runner_sha256": checksum(Path(__file__)), "runtime": runtime,
                      "seed": SEED, "bootstrap_resamples": RESAMPLES,
                      "limitation": "This set was used in earlier project experiments; Step 6 uses frozen decisions without retuning."}
    protocol_path = OUT / "step6_protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == final_protocol, "Final protocol changed"
    else:
        with protocol_path.open("x") as handle:
            json.dump(final_protocol, handle, indent=2)
    pending = any(cache_record("final_test", case, name)[1] is None
                  for case in cases for name in ["siglip", "medgemma"])
    if pending:
        torch.manual_seed(SEED)
        zero_shot, medgemma = load_zero_shot(), load_medgemma()
        medgemma["model"].requires_grad_(False)
        assert label_token_ids(medgemma["processor"].tokenizer) == tokens
        actual_runtime = {"device": config.DEVICE,
                          "medsiglip_revision": getattr(zero_shot["model"].config, "_commit_hash", None),
                          "medgemma_revision": getattr(medgemma["model"].config, "_commit_hash", None),
                          "medsiglip_dtype": str(zero_shot["dtype"]),
                          "medgemma_dtype": str(medgemma["model"].dtype)}
        assert actual_runtime == runtime, "Runtime differs from frozen dev/validation scoring"
        for index, case in enumerate(cases, 1):
            records = {name: cache_record("final_test", case, name) for name in ["siglip", "medgemma"]}
            if all(record is not None for path, record in records.values()):
                continue
            with Image.open(case["path"]) as original:
                cleaned = clean_xray(original.convert("RGB"))
            image_hash = checksum(Path(case["path"]))
            for name, (path, record) in records.items():
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
                    export_final_scores(cases, tokens)
                    raise
                atomic_json(path, json_safe(record))
            print(f"Final test: {index}/200 scored and cached", flush=True)
            if index % 10 == 0:
                export_final_scores(cases, tokens)
    rows = export_final_scores(cases, tokens)
    assert len(rows) == 200
    bad = sum(r["siglip_status"] != "ok" or r["medgemma_status"] != "ok" for r in rows)
    truth = np.array([int(r["true_label"] == "Pneumonia") for r in rows])
    raw = np.array([[float(r["siglip_score"]), float(r["medgemma_score"])] for r in rows])
    nonfinite = int((~np.isfinite(raw)).sum())
    print("Final score audit:", {"errors": bad, "nonfinite": nonfinite,
                                  "parse_failures": sum(r["parse_failure"] for r in rows)}, flush=True)
    if bad or nonfinite:
        raise RuntimeError("Final scoring errors/nonfinite values saved; no images dropped and no retuning")
    scaling = frozen["dev_zscore"]
    fused = ((raw - np.array(scaling["means"])) / np.array(scaling["stds"])).mean(axis=1)
    scores = np.column_stack([raw, fused])
    baseline = np.array([1 if r["medgemma_greedy_answer"] == "Pneumonia" else
                         0 if r["medgemma_greedy_answer"] == "Normal" else -1 for r in rows])
    boot = bootstrap_auc(truth, scores, resamples=RESAMPLES, seed=SEED)
    bounds = np.quantile(boot, [0.025, 0.975], axis=0)
    methods, predictions = {}, {}
    for j, name in enumerate(METHODS):
        threshold = frozen["methods"][name]["threshold"]
        predictions[name] = (scores[:, j] >= threshold).astype(int)
        result = binary_metrics(truth, predictions[name])
        result.update(threshold=threshold, auc=float(roc_auc_score(truth, scores[:, j])),
                      auc_ci95=bounds[:, j].tolist(),
                      mcnemar_vs_greedy=exact_mcnemar(truth, predictions[name], baseline))
        methods[name] = result
    predictions_path = OUT / "final_test_predictions.csv"
    enriched = []
    for index, row in enumerate(rows):
        item = dict(row, fused_score=float(fused[index]))
        for name in METHODS:
            item[name.lower() + "_prediction"] = "Pneumonia" if predictions[name][index] else "Normal"
            item[name.lower() + "_correct"] = bool(predictions[name][index] == truth[index])
        enriched.append(item)
    with predictions_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched[0]))
        writer.writeheader()
        writer.writerows(enriched)
    np.savez_compressed(OUT / "final_test_auc_bootstrap.npz", aucs=boot, methods=np.array(METHODS), seed=SEED)
    report = {"stage": 6, "split": "final_test", "n": 200, "methods": methods,
              "validation_selected_winner": validation["winner"],
              "greedy_baseline": binary_metrics(truth, baseline),
              "seed": SEED, "bootstrap_resamples": RESAMPLES, "nonfinite_scores": nonfinite,
              "parse_failures": sum(r["parse_failure"] for r in rows),
              "score_sign_greedy_agreement": {"agree": sum(r["sign_agrees_greedy"] for r in rows),
                                               "n": 200, "ties": sum(r["medgemma_score_sign_label"] == "Tie" for r in rows)},
              "thresholds_sha256": threshold_hash, "scores_sha256": checksum(OUT / "final_test_scores.csv"),
              "predictions_sha256": checksum(predictions_path),
              "retuning": False, "winner_reselected": False, "limitations": final_protocol["limitation"]}
    assert checksum(thresholds_path) == threshold_hash
    for name, digest in score_protocol["source_hashes"].items():
        assert checksum(ROOT / name) == digest, f"Scoring source changed during run: {name}"
    with final_path.open("x") as handle:
        json.dump(report, handle, indent=2)
    print("FINAL RESULTS:", json.dumps(report, indent=2), flush=True)
    print("FINAL TEST COMPLETE. No thresholds, prompts or winner changed. STOP.", flush=True)


if __name__ == "__main__":
    main()
