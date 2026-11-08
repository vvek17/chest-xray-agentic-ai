"""
Helper code for run_prompts.py.

YOU DO NOT NEED TO CHANGE THIS FILE.
It does the technical work:
  - choose the photos (the 10% / 20% / 70% split and the small balanced sets)
  - choose the solved examples for few-shot prompts
  - save every answer to a log file
"""

import csv
import random
import sys
from pathlib import Path

# Let the VS Code Run button find the project folders
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config

SPLIT_DIR = config.PROJECT_ROOT / "outputs" / "prompt_split"
LOG_DIR = config.PROJECT_ROOT / "outputs" / "prompt_logs"
SPLIT_SIZES = {"train": 0.10, "val": 0.20, "test": 0.70}
IMAGE_TYPES = [".png", ".jpg", ".jpeg"]


# ------------------------------------------------------------
# Photos
# ------------------------------------------------------------
def make_split():
    """Split all clean X-rays into 10% / 20% / 70% (each class separately) and save the lists."""
    all_images = []
    for split in config.SPLITS:
        for disease in config.CLASS_NAMES:
            for path in sorted((config.CLEAN_DIR / split / disease).iterdir()):
                if path.suffix.lower() in IMAGE_TYPES:
                    all_images.append((str(path), disease))

    shuffler = random.Random(config.SEED)   # same split every time
    parts = {"train": [], "val": [], "test": []}

    for disease in config.CLASS_NAMES:
        disease_images = [item for item in all_images if item[1] == disease]
        shuffler.shuffle(disease_images)
        number_train = int(len(disease_images) * SPLIT_SIZES["train"])
        number_val = int(len(disease_images) * SPLIT_SIZES["val"])
        parts["train"] += disease_images[:number_train]
        parts["val"] += disease_images[number_train:number_train + number_val]
        parts["test"] += disease_images[number_train + number_val:]

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, items in parts.items():
        shuffler.shuffle(items)
        with open(SPLIT_DIR / f"{name}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "disease"])
            writer.writerows(items)


def load_split(name):
    """Read one part of the split ("train", "val" or "test")."""
    if not (SPLIT_DIR / "train.csv").exists():
        make_split()
    with open(SPLIT_DIR / f"{name}.csv") as f:
        rows = list(csv.DictReader(f))
    return [(Path(row["path"]), row["disease"]) for row in rows]


def get_photos(part, number_of_photos, classes):
    """
    Return a small, balanced list of (photo path, true class).
    The photos are taken in turn from each class: Normal, Pneumonia, Normal, ...
    The same photos come back every time.
    """
    per_class = {}
    for disease in classes:
        per_class[disease] = []
    for path, disease in load_split(part):
        if disease in classes:
            per_class[disease].append((path, disease))

    chosen = []
    position = 0
    while len(chosen) < number_of_photos:
        for disease in classes:
            if len(chosen) < number_of_photos:
                chosen.append(per_class[disease][position])
        position += 1
    return chosen


def get_examples(classes, examples_per_class, photos=None):
    """
    Example photos for few-shot prompts, with their correct class.
    They come from the training part, but from far down the list (position 100+),
    so they are never one of the photos we test on.
    """
    # A caller may supply a fresh, already screened pool for a new experiment.
    # Keep the original behavior for every existing caller.
    EXAMPLE_START = 100
    if photos is None:
        photos = load_split("train")
    else:
        EXAMPLE_START = 0
    per_class = {}
    for disease in classes:
        per_class[disease] = []
    for path, disease in photos:
        if disease in classes:
            per_class[disease].append((path, disease))

    examples = []
    for i in range(examples_per_class):
        for disease in classes:
            examples.append(per_class[disease][EXAMPLE_START + i])
    return examples


# ------------------------------------------------------------
# Log
# ------------------------------------------------------------
def save_log(results, version, classes, part):
    """Save every answer to outputs/prompt_logs/ (for example 2c_v1_2class_val_20cases.csv)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{version}_{len(classes)}class_{part}_{len(results)}cases.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true", "answer", "predicted"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved all answers: {log_path}")


if __name__ == "__main__":
    print("This file is only a helper. It is not meant to be run.")
    print("Run prompt_engineering/run_prompts.py instead.")
