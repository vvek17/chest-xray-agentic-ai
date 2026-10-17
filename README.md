# Chest X-Ray Classification for an Agentic AI System

Classify chest X-rays as **Covid-19**, **Normal**, **Pneumonia**, or **Tuberculosis**
(4-class image classification), as the X-ray "tool" of a planned agentic AI system.

Two parts:

1. **Classifier**: a frozen pretrained model + a linear probe (no fine-tuning).
   Best: **MedSigLIP + linear probe, 95.6% test accuracy**.
2. **Prompt engineering** (first step of the agent): ask MedGemma about an X-ray
   with a prompt, read its answers, and improve the prompt step by step.

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

Run from this folder (`python -m src.<name>`), or use the VS Code Run button.

| Command | What it does |
|---|---|
| `python -m src.data_check` | removes empty, broken and duplicate images; copies the clean data to `data/processed/` |
| `python -m src.dinov2_classify` | frozen model + linear probe; the model is chosen by `DINOV2_NAME` in `config.py` |
| `python -m src.prompt_engineering` | prompt engineering with MedGemma (see below) |

## 1. Classifier: frozen model + linear probe

The pretrained model is **frozen** (it never changes). It turns each X-ray into numbers
(an embedding), and only a small **linear probe** learns from them (10 epochs, the best
epoch on Validation is kept).

Test results (2,791 X-rays):

| Model | Learned from | Covid-19 | Normal | Pneumonia | Tuberculosis | All |
|---|---|---|---|---|---|---|
| RAD-DINO (Microsoft) | chest X-rays | 84.4% | 75.4% | 96.7% | 95.3% | 87.9% |
| DINOv2 (Meta) | normal photos | 85.6% | 79.6% | 94.9% | 95.1% | 88.8% |
| **MedSigLIP (Google)** | medical images + text | **93.6%** | **92.1%** | **98.1%** | **98.7%** | **95.6%** |

Other ways to use MedSigLIP's numbers: zero-shot (no training) 48.6%, k-NN 84.9%.

## 2. Prompt engineering with MedGemma

MedGemma (`google/medgemma-1.5-4b-it`) looks at an X-ray and answers in words.
We do not train it; only the prompt changes.

- **Data:** 100 balanced photos: 10 training (read the answers, improve the prompt),
  20 validation (accuracy), 70 testing (final, used once).
- **Files:**
  - `src/prompts.py`: all prompt versions, each with a note on its result
  - `src/prompt_engineering.py`: settings (`MODE`, `CLASSES`) and the simple main steps
  - `src/medgemma_helper.py`: technical code (photos, MedGemma, logs)
- **Loop:** write a prompt, run `MODE = "try"`, read the answers, change one thing,
  repeat; then `MODE = "validate"`; at the end `MODE = "test"` once.

Progress:

| Task | Prompt | Training (10) | Validation (20) |
|---|---|---|---|
| 4 classes | v1: one-word answer | 5 / 10 | - |
| 4 classes | v2: describe the classes, describe first, then answer | 5 / 10 | - |
| 4 classes | v3: step-by-step rules | 1 / 10 | - |
| 2 classes (Normal vs Pneumonia) | 2c_v1: one-word answer | 9 / 10 | 17 / 20 (85%) |

Lessons so far: short, clear prompts work better with this small model; naming a sign
in the prompt (for example "hazy") can make the model "see" it in healthy lungs.

## Dataset

15,600 chest X-rays, already split into Training / Validation / Testing.
`data_check.py` found no empty or broken images and removed 38 duplicates
(a picture in both Training and Testing stays only in Training).

| | Covid-19 | Normal | Pneumonia | Tuberculosis | Total |
|---|---|---|---|---|---|
| Training | 2,699 | 2,697 | 2,688 | 2,693 | 10,777 |
| Validation | 500 | 500 | 494 | 500 | 1,994 |
| Testing | 700 | 700 | 693 | 698 | 2,791 |

## Project structure

| Path | Purpose |
|---|---|
| `src/config.py` | all settings: paths, classes, model names, training settings, device |
| `src/data_check.py` | clean the dataset and copy it to `data/processed/` |
| `src/dinov2_classify.py` | frozen model (DINOv2, RAD-DINO or MedSigLIP) + k-NN, linear probe, zero-shot |
| `src/prompts.py`, `src/prompt_engineering.py`, `src/medgemma_helper.py` | prompt engineering with MedGemma |
| `src/preprocessing.py`, `src/data_loader.py`, `src/model.py`, `src/train.py`, `src/evaluate.py` | earlier fine-tuning experiments (ResNet18 / DenseNet121), kept for reference |
| `src/utils.py` | helpers: `set_seed`, `save_model`, `load_model` |
| `notebooks/01_data_exploration.ipynb` | image counts, sample images, sizes and colors |

## Known limits

- Mild Covid-19 and healthy adult lungs are the hardest pair; most errors are between them.
- Classes partly come from different hospitals and age groups, so a model may partly
  learn the source instead of the disease.
- The same patient may have X-rays in more than one split (file names do not show patients).
- Not clinically validated. Results are for research and learning only.

## Next steps

- Finish prompt engineering (more versions, validation, final test).
- `predict.py`: one X-ray in, class and confidence out (MedSigLIP + probe).
- Build the agent: it calls the classifier and MedGemma as tools and asks a doctor when unsure.
