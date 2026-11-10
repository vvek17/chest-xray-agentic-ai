# Prompt 4: A CLAIM-style report for every prompt version

Why this prompt exists: the current prompt logs in `logs/prompt_runs/` show 10
images per prompt and no accuracy. `docs/medgemma_fresh_results.md` shows
accuracy for one prompt only, with no confidence interval, no per-source
breakdown and no failure analysis. The gold standard (CLAIM 2024) asks for
diagnostic accuracy with precision (item 36), a statement of uncertainty
(item 29), and a failure analysis of wrong cases (item 37).

Run prompt 3 first. This prompt needs `outputs/prompt_split_grouped/`.

Give the prompt below to Codex.

```text
Write one script: chest-xray-pneumonia/prompt_engineering/report_prompts.py

Goal: evaluate every binary prompt version on the same patient-grouped split
and write one CLAIM-style report per prompt.

Reuse the existing backend without changing it: backend.clean_image,
backend.predict, backend.medgemma, backend.xray_check, and the prompt text in
prompt_engineering/prompts.py. Real model calls only. Do not change prompt
wording, model logic or thresholds.

Input split: outputs/prompt_split_grouped/{train,val,test}.csv from
build_prompt_split.py. Default to train. --split val and --split test must be
explicit, so the test split is not touched by accident.
--per-class N limits how many images per class are used (default 20).
Sample images with random.Random(config.SEED) after sorting, so runs repeat.

Few-shot examples: take them from the TRAIN split only, never from the split
being evaluated, and never from a patient that appears in the evaluated set.
Refuse to run (exit 1) if an example patient appears in the evaluated set.

For every prompt in PROMPT_NAMES and every image:
  - clean the image, get the four sigmoid scores, run the scope check
  - build the prompt exactly as the app builds it
  - call MedGemma with do_sample=False and keep the raw answer
  - read the answer with backend.medgemma.read_answer; an answer that is not
    Normal or Pneumonia counts as "Other", never as a correct answer
  - record patient_id and source from the split CSV

Write outputs/prompt_reports/<prompt_name>.json with every per-image row
(path, patient_id, source, true label, raw answer, parsed label, scores,
stage timings), and outputs/prompt_reports/<prompt_name>.md with:

  1. Header: prompt name, the full system and user text, the split, N images,
     N patients, the seed, the model name and revision, and the date.
  2. Accuracy, balanced accuracy, macro F1, MCC.
  3. Per class: precision, recall/sensitivity, specificity, F1, support.
  4. A Wilson 95% confidence interval for accuracy, sensitivity and
     specificity. Implement Wilson by hand with the math module; do not add a
     dependency. Print it as point estimate [low, high].
  5. The confusion matrix, rows = true, columns = predicted, order
     Normal, Pneumonia, Other, and the count of unparsed answers.
  6. Per-source accuracy table (source x true class x N x accuracy), with one
     line saying which classes are present in only one source.
  7. Failure analysis: every wrong image, with its raw answer, its classifier
     scores, its source, and whether the scope check accepted it. Group the
     wrong cases by error type (Normal called Pneumonia, Pneumonia called
     Normal, unparsed).
  8. A "Limitations" section with these exact facts, not softened:
     scores are uncalibrated sigmoid scores, not probabilities; ROC-AUC and
     calibration cannot be computed from label-only answers; MedGemma 1.5 is
     documented as not evaluated for multi-image or multi-turn use, so any
     few-shot prompt is outside its evaluated scope; overlap between these
     public datasets and MedGemma pretraining is not established; Pneumonia
     comes from one paediatric source.

Then write outputs/prompt_reports/summary.md: one row per prompt with N,
accuracy with its Wilson interval, balanced accuracy, macro F1, unparsed
count, and total runtime. Sort by accuracy, highest first. Add one line below
the table saying whether the top two intervals overlap, and if they do, state
that the difference is not established at this sample size.

Rules:
  - if one image fails, log the error in the JSON and keep going
  - checkpoint the JSON after every image, so an interrupted run keeps its rows
  - json indent=2, no NaN
  - no new dependencies, no emojis
  - do not print or save the test-split rows unless --split test was passed

Also write tests/test_report_prompts.py with pytest tests, using fake models
and a tiny fake split, for:
  - the Wilson interval against two hand-computed cases
  - an unparsed answer counts as Other and never as correct
  - the script exits non-zero when a few-shot example patient is also in the
    evaluated set
  - per-source accuracy adds up to the overall accuracy
  - metrics are computed from the per-image rows, not recomputed differently
    in the markdown writer
```
