# Prompt 3: A leakage-free, source-tagged split for prompt evaluation

Why this prompt exists: the current prompt split in `outputs/prompt_split/` is
random per image. The same patient appears in several splits (253 of 335
pneumonia patients in `train.csv` also appear in `test.csv`). The gold standard
(CLAIM 2024, item "partition level") asks for splits at the patient level, not
the image level. The split is also confounded by source: every Pneumonia image
comes from one paediatric dataset (`personN_bacteria_*`, `personN_virus_*`),
while Normal mixes several sources. Any accuracy from the current split can
come from the source, not from the disease.

Give the prompt below to Codex.

```text
Write one script: chest-xray-pneumonia/prompt_engineering/build_prompt_split.py

Goal: replace the image-level prompt split with a patient-grouped,
source-tagged split, so prompt evaluation cannot leak patients between splits.

Do not change model code, prompt text, thresholds, or any training script.
Do not move or rewrite image files. Write only new CSV files.

Input: data/processed/{Training,Validation,Testing}/{Normal,Pneumonia}/*
Reuse backend.config for PROJECT_ROOT, CLEAN_DIR, SEED and SPLITS.

Step 1 - give every image a patient id and a source id, from its file name:
  - personN_bacteria_*, personN_virus_*  -> source "guangzhou_paediatric", patient "person<N>"
  - NORMALn-IM-*, IM-*                   -> source "guangzhou_paediatric", patient = full stem
  - CHNCXR_*                             -> source "shenzhen",   patient = full stem
  - MCUCXR_*                             -> source "montgomery", patient = full stem
  - Normal-N.png                         -> source "tb_database", patient = full stem
  - anything else                         -> source "unknown",    patient = full stem
  Keep the rules in one dictionary at the top of the file, so a new source
  can be added in one line. If a file matches no rule, count it and report it.

Step 2 - group by patient. Every image of one patient must land in exactly one
split. Shuffle patients with random.Random(config.SEED), then fill
train 10% / val 20% / test 70% of the patients of each class, class by class.

Step 3 - write outputs/prompt_split_grouped/{train,val,test}.csv with the
columns: path, disease, patient_id, source.

Step 4 - write outputs/prompt_split_grouped/report.md with:
  - images and patients per split and per class
  - images and patients per source per split and per class
  - the patient overlap between every pair of splits (must be 0)
  - the same overlap numbers for the old outputs/prompt_split/*.csv, as the
    before/after comparison
  - a short plain-English warning that Pneumonia comes from one source only,
    so a per-source reading of any result is required

Rules:
  - fail loudly (exit code 1) if any pair of splits shares a patient
  - fail loudly if a class ends with 0 patients in any split
  - deterministic: two runs give byte-identical CSVs
  - no new dependencies, no emojis, no accuracy calculation in this script

Also write tests/test_prompt_split_grouped.py with pytest tests for:
  - the file-name to patient/source rules, one case per rule
  - zero patient overlap between all three splits
  - determinism: two builds give the same rows
  - every input image appears exactly once across the three CSVs
```
