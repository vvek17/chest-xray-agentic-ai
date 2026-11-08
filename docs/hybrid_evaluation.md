# Frozen probe-hint evaluation

This experiment evaluates the existing trained **four-class MedSigLIP probe**,
not the older image-text similarity matcher. MedGemma weights stay frozen.
Patient independence and pretraining overlap are unverified. Research only.

## How to run

From `/Users/vivek/Desktop/Alzaimer-AgenticAI/chest-xray-pneumonia`:

```bash
source ../.venv/bin/activate
python -m prompt_engineering.evaluate_hybrid --stage prepare
python -m prompt_engineering.evaluate_hybrid --stage dev
python -m prompt_engineering.evaluate_hybrid --stage val
python -m prompt_engineering.evaluate_hybrid --stage freeze
python -m prompt_engineering.evaluate_hybrid --stage test
python -m prompt_engineering.evaluate_hybrid --stage report
```

Preparation hashes/selects images but never runs inference. The default stage is
`prepare`. Stop before `freeze` if reviewing development and validation results.
Run an interrupted stage again to resume its exact recorded configuration.
Never delete predictions or the freeze file to change a finished experiment.

## What to change yourself

**Nothing is required for this checkout.** `get_hint` already calls
`backend.predict.predict` using `PROBE_4_CLASSES`. It compares only Normal and
Pneumonia scores; a higher COVID-19 or Tuberculosis score is retained in the log
but does not enter this binary comparison. Equal scores choose Normal.

For a future experiment, settings are at the top of `evaluate_hybrid.py`:
`CLASSES`, `SIZES`, `OUT`, `SOURCE`, `HYBRIDS`, and `ARMS`. Add prompts as NEW versions
in `prompts.py`, never overwrite a measured version. Change `OUT` for a new
configuration; do not edit a saved manifest to bypass the checks. If the original
dataset moves, update `RAW_DIR` in `backend/config.py` before preparing a new run.

## Data and exclusions

All 260 target images AND the two labeled context examples come from the original
`config.RAW_DIR / "Testing"` folder. Context examples are separate from dev, val,
and test. All variants use the exact same selected targets.

The preparation step scans answer CSVs recursively under `outputs/`, legacy
`LOG_DIR`, and saved manifests. It hashes the recorded files and their matching
original counterparts, including prior few-shot examples. Old reserved manifest
cases are conservatively excluded, even if inference was interrupted. Missing
historical inputs cause an error instead of silently weakening the exclusion.
Raw Training/Validation hashes are also blocked, preventing exact copies from
entering this Testing selection. Distinct SHA-256 hashes do not prove different
patients or exclude near-duplicates. Unrecorded historical usage is unknown.

Do not run another MedGemma experiment against these reserved images. Every stage
rescans external history and refuses newly consumed reserved inputs.

## Six comparisons

1. MedSigLIP binary argmax using its trained four-class probe.
2. MedGemma zero-shot, existing `2c_v1`.
3. MedGemma few-shot, existing `2c_fs1`.
4. Few-shot plus `hybrid_confidence_v1`.
5. Few-shot plus `hybrid_label_v1`.
6. Few-shot plus `hybrid_verify_v1`.

All few-shot arms share one fresh Normal and one fresh Pneumonia example.
Example questions stay equal to the baseline question: target hints are never
copied into the context examples. Targets and examples use the existing image
cleaner. Zero-shot has no context examples. No training or prompt selection uses
true target labels at inference.

The confidence variant presents the winning **uncalibrated sigmoid score**.
It is neither a calibrated disease probability nor a two-class normalized score.
All prompts request exactly `Normal` or `Pneumonia`. Unexpected model answers are
saved verbatim and marked `Other`; they are never silently forced to a class.

## Freeze and test

The primary hybrid is selected using validation macro F1, then Pneumonia recall,
then Pneumonia specificity. A complete tie uses the order in `HYBRIDS`.
`selection.json` locks that choice and the manifest/dev/val log hashes before test
inference. The manifest records prompts, code hashes, probe checksum, cached
Hugging Face model revisions, dependency versions, generation settings and device.
The six test comparisons are predeclared; do not select another winner on test.

## Outputs and definitions

Folder: `outputs/hybrid_original_testing_20_40_70/`.

- `manifest.json`: exact examples, cases, hashes, historical sources and settings.
- `dev_predictions.csv`, `val_predictions.csv`, `test_predictions.csv`: long format,
  six rows per image, raw answers, labels, hints and all four probe scores.
- `*_metrics.json`: metrics for every arm, per-class precision/recall/specificity/F1,
  macro/weighted F1, accuracy, balanced accuracy, MCC, invalid answers, hint metrics.
- `*_confusion.png`: six panels; rows are true classes, columns predicted classes.
- `report.md`: completed split results, including per-class metrics and matrices.
- `selection.json`: frozen primary hybrid and result checksums.

Hint follow rate = matching hint labels / all cases. For no-hint baselines this is
agreement only, not hint exposure. Errors fixed = incorrect hints whose final
answer matches truth. Fix rate divides by incorrect hints. Correct hints broken =
correct hints whose final answer is wrong OR invalid. Break rate divides by correct
hints. Empty denominators produce JSON `null`. Invalid outputs remain in all
classification denominators. Probability-based MedGemma metrics are unavailable
from label-only generation.

These staged inference runs may take substantial time: each image has five
MedGemma generations. No performance numbers exist until those runs finish.
