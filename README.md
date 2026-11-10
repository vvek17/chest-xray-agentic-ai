# Chest X-Ray Classification for an Agentic AI System

Classify chest X-rays as **Covid-19**, **Normal**, **Pneumonia**, or **Tuberculosis**,
as the X-ray "tool" of a planned agentic AI system.

Two tools, both from Google, neither fine-tuned:

1. **Classifier**: MedSigLIP (frozen) + a linear probe. Gives a label and a confidence.
   **95.6%** test accuracy on 4 classes; **98.4%** on Normal vs Pneumonia.
2. **MedGemma** with prompt engineering. Answers in words; nothing is trained.
   **90.71%** on the fresh Normal vs Pneumonia test (127 / 140);
   the earlier test achieved 87.9% (123 / 140). These use different images and examples.

## Metrics results for review

The [metrics results and senior review guide](metrics_results/README.md) includes saved metric CSVs, prediction and prompt logs, curve images, confusion matrices, and run-status notes. Start there to review completed results and identify the main limitations.

## Project structure

```
chest-xray-pneumonia/
  frontend/            the web page (Streamlit), user interface only
    app.py
  backend/             the models and the logic the web page calls
    config.py            all settings: paths, classes, model names, limits
    clean_image.py       clean an uploaded photo (gray + cut empty borders)
    predict.py           MedSigLIP + linear probe: one X-ray in, label + confidence out
    medgemma.py          load MedGemma, ask it about an X-ray, read its answer
    xray_check.py        safety checks (is it a chest X-ray? supported label? confident?)
  prompt_engineering/  MedGemma prompt engineering
    prompts.py           all prompt versions, each with a note on its result
    run_prompts.py       run a prompt on try / validate / test photos and print the accuracy
    photos.py            photo split (10 / 20 / 70), few-shot examples, answer logs
  training/            make the classifier
    data_check.py        clean the dataset (removes broken and duplicate images)
    train_probe.py       frozen model (DINOv2, RAD-DINO or MedSigLIP) + linear probe, 4 classes
    train_2class_probe.py  MedSigLIP + linear probe, Normal vs Pneumonia only
  old_experiments/     earlier fine-tuning of DenseNet121 / ResNet18 (kept for reference only)
  tests/               fast rule tests and slow tests with the real models
  notebooks/           data exploration
  docs/                project guide (PDF)
```

Files the code makes (not in the repository, because they are large):
`data/processed/` (clean images), `models/` (saved probes), `outputs/` (features, logs, prompt answers).

## Setup

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MedSigLIP and MedGemma are gated models on Hugging Face: accept their terms on the
model pages, then log in with `hf auth login`.

The dataset is not in this repository. Put it next to this folder as `../Chest_X-ray/`
(Training / Validation / Testing, each with Covid-19 / Normal / Pneumonia / Tuberculosis).

## How to run

Run every command from this folder.

| Command | What it does |
|---|---|
| `python -m training.data_check` | clean the dataset and copy it to `data/processed/` |
| `python -m training.train_probe` | train the 4-class probe (model chosen by `DINOV2_NAME` in `backend/config.py`) |
| `python -m training.train_2class_probe` | train the Normal vs Pneumonia probe |
| `python -m prompt_engineering.run_prompts` | prompt engineering with MedGemma (`MODE` = try / validate / test) |
| `python -m backend.predict photo.png` | classify one X-ray in the terminal |
| `streamlit run frontend/app.py --server.port 8502` | start the web page, then open http://localhost:8502 |
| `python -m pytest` | run all tests (`-m "not slow"` for the fast ones only) |

## 1. Classifier: frozen model + linear probe

The pretrained model is **frozen**. It turns each X-ray into numbers (an embedding),
and only a small **linear probe** learns from them (10 epochs, the best epoch on
Validation is kept).

Test results, 4 classes (2,791 X-rays):

| Model | Learned from | Covid-19 | Normal | Pneumonia | Tuberculosis | All |
|---|---|---|---|---|---|---|
| RAD-DINO (Microsoft) | chest X-rays | 84.4% | 75.4% | 96.7% | 95.3% | 87.9% |
| DINOv2 (Meta) | normal photos | 85.6% | 79.6% | 94.9% | 95.1% | 88.8% |
| **MedSigLIP (Google)** | medical images + text | **93.6%** | **92.1%** | **98.1%** | **98.7%** | **95.6%** |

Other ways to use MedSigLIP's numbers: zero-shot (no training) 48.6%, k-NN 84.9%.
Normal vs Pneumonia only (2-class probe): **98.35%**.

## 2. Prompt engineering with MedGemma

MedGemma (`google/medgemma-1.5-4b-it`) looks at an X-ray and answers in words.
We do not train it; only the prompt changes.

- **Photos per class:** 10 to try (read the answers, improve the prompt),
  20 to validate (choose the best prompt), 70 to test (final, used once).
- **Loop:** write a prompt in `prompts.py`, run `run_prompts.py` with `MODE = "try"`,
  read the answers, change one thing, repeat; then `"validate"`; at the end `"test"` once.

| Task | Prompt | Try | Validate | Test |
|---|---|---|---|---|
| 4 classes | v1: one-word answer | 5 / 10 | - | - |
| 4 classes | v2: describe first, then answer | 5 / 10 | - | - |
| 4 classes | v3: step-by-step rules | 1 / 10 | - | - |
| Normal vs Pneumonia | 2c_v1: one-word answer | 17 / 20 | 32 / 40 (80%) | - |
| Normal vs Pneumonia | **2c_fs1: 2c_v1 + 1 solved example per class (few-shot)** | 17 / 20 | 35 / 40 (87.5%) | **123 / 140 (87.9%)** |

Final test errors: Normal 65 / 70 correct, Pneumonia 58 / 70 correct
(12 of the 17 errors are pneumonia called Normal).

Lessons: short, clear prompts work better with this small model; naming a sign
in the prompt (for example "hazy") can make the model "see" it in healthy lungs.

## 3. Web page

`frontend/app.py` displays the trained four-class MedSigLIP probe's scores and
compares Normal versus Pneumonia using their rescaled scores. Frozen MedGemma
then answers with the selected prompt. The sidebar offers Hint + score, Hint only,
Double-check, No hint, and No hint + 2 examples. `CURRENT_VERSION` selects the
default; the plain labels live in `PROMPT_LABELS` in the app.

The app has Analyze X-ray, Evaluation results, and About tabs. Evaluation results
reads existing files from `outputs/hybrid_siglip_medgemma/` and shows a friendly
empty state when they are absent. The staged evaluator writes a separate folder,
`outputs/hybrid_original_testing_20_40_70/`; change `RESULTS_DIR` explicitly to
inspect it. Interactive rescaled hints and evaluation raw-score hints are different
configurations; do not attribute one configuration's results to another.

See [the complete app source and setup guide](docs/hybrid_streamlit_app.md).
Both models load on demand and stay cached. Scores and agreement are not calibrated
diagnostic certainty. Research prototype; not for clinical use.


## Dataset

15,600 chest X-rays, already split into Training / Validation / Testing.
`data_check.py` found no empty or broken images and removed 38 duplicates
(a picture in both Training and Testing stays only in Training).

| | Covid-19 | Normal | Pneumonia | Tuberculosis | Total |
|---|---|---|---|---|---|
| Training | 2,699 | 2,697 | 2,688 | 2,693 | 10,777 |
| Validation | 500 | 500 | 494 | 500 | 1,994 |
| Testing | 700 | 700 | 693 | 698 | 2,791 |

## Known limits

- **Dataset shortcut.** Each class comes from different sources: all Pneumonia images are
  young children from one hospital; Covid-19 and Tuberculosis are mostly adults from other
  hospitals. The probe can learn the source instead of the disease. On 2 pneumonia X-rays
  from outside the dataset (an adult, an older child), the classifier was wrong on both
  (Normal 47%, Tuberculosis 81%); MedGemma was right on both. More outside images are
  needed to measure this properly.
- MedGemma gives no confidence number and takes about 5 seconds per X-ray.
- The same patient may have X-rays in more than one split (file names do not show patients).
- Not clinically validated. Results are for research and learning only.

## Next steps

- Test both tools on more X-rays from outside the dataset.
- Build the agent: it calls the classifier and MedGemma as tools and asks a doctor
  when they disagree or are unsure.

## Fresh pretrained MedGemma evaluation

Run `python -m prompt_engineering.evaluate_medgemma` from this folder.
This uses the frozen `2c_fs1` prompt without weight fine-tuning, with 20 Normal
and 20 Pneumonia images for development, 40 each for validation, and 70 each
for testing. It excludes image contents recorded in previous prompt logs and
the interrupted initial evaluation, and uses two separate, fresh few-shot examples.
The manifests preserve the exact selected paths and image SHA-256 hashes.

Results are saved incrementally under `outputs/medgemma_fresh_20_40_70/`:
per-image prediction CSVs, per-split metric JSONs, confusion-matrix PNGs,
and `report.md` when all three splits finish. Metrics include accuracy,
balanced accuracy, per-class precision, recall/sensitivity, specificity, F1,
macro/weighted F1, MCC, support, and unparsed-answer counts. Unparsed answers
count as errors. Probability-based metrics are unavailable from label-only output.
Patient independence and overlap with MedGemma pretraining are not established.

## Frozen MedSigLIP-probe + MedGemma hybrid

See [the staged hybrid evaluation guide](docs/hybrid_evaluation.md). Run
`python -m prompt_engineering.evaluate_hybrid --stage prepare` to reserve fresh
original Testing images, then run `dev`, `val`, `freeze`, and `test` separately.
The test stage requires a frozen configuration. All six arms share the same
cases; prompts are versioned in `prompt_engineering/prompts.py`.

## Saved evaluation summary

The aggregate [fresh standalone MedGemma report](docs/medgemma_fresh_results.md)
records the completed 20/40/70-per-class evaluation. Hybrid preparation and unit
tests do not establish hybrid accuracy; its staged inference must be run separately.

Dataset images, model checkpoints, generated prediction logs, local manifests,
and credentials are excluded from Git. Source code, tests, guides and project
presentations are included. The test command for checks without real models is
`python -m pytest -m "not slow"`.
