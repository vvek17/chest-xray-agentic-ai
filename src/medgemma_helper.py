"""
Helper code for prompt_engineering.py.

YOU DO NOT NEED TO CHANGE THIS FILE.
It does the technical work:
  - choose the photos (the 10% / 20% / 70% split and the small balanced sets)
  - load MedGemma and ask it about one X-ray
  - save every answer to a log file
"""

import csv
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from src import config

MEDGEMMA_NAME = "google/medgemma-1.5-4b-it"
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


# ------------------------------------------------------------
# MedGemma
# ------------------------------------------------------------
def load_medgemma():
    """Load MedGemma (the first time it downloads about 8 GB)."""
    print(f"Loading {MEDGEMMA_NAME} ...")
    processor = AutoProcessor.from_pretrained(MEDGEMMA_NAME)
    model = AutoModelForImageTextToText.from_pretrained(MEDGEMMA_NAME, dtype=torch.bfloat16)
    model = model.to(config.DEVICE)
    model.eval()
    return {"processor": processor, "model": model}


def ask_medgemma(medgemma, photo_path, prompt):
    """Show MedGemma one X-ray with the prompt. Return its answer as text."""
    processor = medgemma["processor"]
    model = medgemma["model"]

    image = Image.open(photo_path).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt["user"]},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(config.DEVICE, dtype=torch.bfloat16)

    input_length = inputs["input_ids"].shape[-1]
    max_tokens = prompt.get("max_tokens", 20)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    new_tokens = output[0][input_length:]    # only the words MedGemma wrote
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


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
