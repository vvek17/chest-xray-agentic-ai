"""Reproducible pretrained MedGemma evaluation; no weight updates.

Run: python -m prompt_engineering.evaluate_medgemma
Uses the existing frozen 2c_fs1 prompt and original split manifests.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef

from backend import config
from backend.medgemma import ask_medgemma, load_medgemma, read_answer, MEDGEMMA_NAME
from prompt_engineering.photos import get_photos, get_examples, load_split, LOG_DIR
from prompt_engineering.prompts import PROMPTS

CLASSES = ['Normal', 'Pneumonia']
SIZES = {'train': 20, 'val': 40, 'test': 70}
OUT = config.PROJECT_ROOT / 'outputs' / 'medgemma_fresh_20_40_70'
VERSION = '2c_fs1'


def metrics(rows):
    truth = [r['true'] for r in rows]
    predicted = [r['predicted'] for r in rows]
    labels = CLASSES + ['Other']
    cm = confusion_matrix(truth, predicted, labels=labels)
    report = classification_report(truth, predicted, labels=CLASSES,
                                   output_dict=True, zero_division=0)
    per_class = {}
    for i, label in enumerate(CLASSES):
        tp = int(cm[i, i]); fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp); tn = len(rows) - tp - fn - fp
        per_class[label] = dict(report[label], specificity=tn/(tn+fp) if tn+fp else None,
                               tp=tp, fn=fn, fp=fp, tn=tn)
    return {'n': len(rows), 'accuracy': accuracy_score(truth, predicted),
            'balanced_accuracy': float(np.mean([report[c]['recall'] for c in CLASSES])),
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'mcc': matthews_corrcoef(truth, predicted),
            'unparsed_answers': predicted.count('Other'),
            'per_class': per_class, 'confusion_labels': labels,
            'confusion_matrix': cm.tolist()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prompt = PROMPTS[VERSION]
    excluded = set()
    sources = list(LOG_DIR.glob('*.csv')) + list((config.PROJECT_ROOT / 'outputs' / 'medgemma_20_40_70').glob('*_predictions.csv'))
    for source in sources:
        with source.open() as f:
            for row in csv.DictReader(f):
                path = Path(row['path'])
                if path.exists():
                    excluded.add(hashlib.sha256(path.read_bytes()).hexdigest())
    # Previous solved examples are also excluded from this run.
    for path, _ in get_examples(CLASSES, 1):
        excluded.add(hashlib.sha256(path.read_bytes()).hexdigest())
    selected = {}
    examples = []
    used = set(excluded)
    for part, count in SIZES.items():
        chosen = []
        for label in CLASSES:
            candidates = []
            for path, truth in load_split(part):
                if truth != label:
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in used:
                    continue
                candidates.append((path, truth))
                used.add(digest)
                if len(candidates) == count + (1 if part == 'train' else 0):
                    break
            needed = count + (1 if part == 'train' else 0)
            if len(candidates) < needed:
                raise ValueError(f'Insufficient unused images in {part}/{label}')
            chosen.extend(candidates[:count])
            if part == 'train':
                examples.extend(candidates[count:])
        selected[part] = chosen
    # Require disjoint actual file contents, including few-shot context examples.
    seen = {}
    for part, photos in list(selected.items()) + [('examples', examples)]:
        for path, label in photos:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError(f'Duplicate image across selected data: {path} and {seen[digest]}')
            seen[digest] = str(path)
    manifest = {'model': MEDGEMMA_NAME, 'prompt_version': VERSION, 'prompt': prompt,
                'weight_training': False, 'test_previously_evaluated': False, 'excluded_image_count': len(excluded), 'excluded_log_sources': [str(p) for p in sources],
                'examples': [(str(p), c) for p, c in examples],
                'splits': {k: [(str(p), c) for p, c in v] for k, v in selected.items()},
                'sha256_to_path': seen}
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing run manifest differs; use a separate output directory.')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    model = None
    summary = {}
    for part, photos in selected.items():
        log = OUT / f'{part}_predictions.csv'
        cached = {}
        # Reuse only answers from this same frozen prompt, not other models.
        for source in ([log] if log.exists() else []):
            with source.open() as f:
                for row in csv.DictReader(f):
                    cached[(row['path'], row['true'])] = row
        rows = []
        with log.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'true', 'answer', 'predicted'])
            writer.writeheader()
            for index, (path, truth) in enumerate(photos, 1):
                row = cached.get((str(path), truth))
                if row is None:
                    if model is None:
                        model = load_medgemma()
                    answer = ask_medgemma(model, path, prompt, examples)
                    row = dict(path=str(path), true=truth, answer=answer,
                               predicted=read_answer(answer, CLASSES))
                rows.append(row); writer.writerow(row); f.flush()
                print(f'{part}: {index}/{len(photos)}', flush=True)
        result = metrics(rows)
        summary[part] = result
        (OUT / f'{part}_metrics.json').write_text(json.dumps(result, indent=2))
        cm = np.array(result['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap='Blues')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=result['confusion_labels'],
               yticklabels=result['confusion_labels'], xlabel='Predicted class', ylabel='True class',
               title=f'MedGemma {part} (n={len(rows)})')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center')
        fig.tight_layout(); fig.savefig(OUT / f'{part}_confusion.png', dpi=160); plt.close(fig)
    lines = ['# Pretrained MedGemma: Normal vs Pneumonia', '',
             'Frozen 2c_fs1 prompt; no model weights were trained. Development uses 20 images per class, validation 40, test 70.',
             'Images in previous prompt logs and the interrupted run were excluded by SHA-256. Patient independence and pretraining overlap have not been established.',
             'All predictions are generated for this fresh run with new few-shot examples; only this run’s checkpoints are reused on resume.',
             'ROC-AUC, PR-AUC, log loss and calibration are unavailable because these runs produce labels, not class probabilities.', '',
             '| Split | N | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | MCC |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for part, r in summary.items():
        lines.append(f"| {part} | {r['n']} | {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | {r['mcc']:.4f} |")
    for part, r in summary.items():
        lines += ['', f'## {part}', '', '| Class | Precision | Recall / sensitivity | Specificity | F1 | Support |', '|---|---:|---:|---:|---:|---:|']
        for label, c in r['per_class'].items():
            lines.append(f"| {label} | {c['precision']:.4f} | {c['recall']:.4f} | {c['specificity']:.4f} | {c['f1-score']:.4f} | {c['support']} |")
        lines += ['', f"Unparsed answers: {r['unparsed_answers']}. Confusion matrix rows = true class; columns = predicted class; order: Normal, Pneumonia, Other.", '', '```', str(np.array(r['confusion_matrix'])), '```', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    print(f'Report: {OUT / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
