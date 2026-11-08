"""
Train the classifier for ONLY 2 classes: Normal vs Pneumonia.

The classifier = MedSigLIP (frozen) + a linear probe (the only part that learns).

Steps:
  1. MedSigLIP looks at every Normal and Pneumonia photo and turns it into
     1152 numbers (about 20 minutes). The numbers are saved in
     outputs/features/medsiglip-448_2class_<split>.pt
  2. Train the linear probe for 10 epochs, keep the best epoch on Validation
  3. Test it on the Testing X-rays
  4. Save it as models/probe_medsiglip-448_2class.pth

Run it from the chest-xray-pneumonia folder (or with the Run button):
    python -m training.train_2class_probe
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the project folders
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor

from backend import config

# ============================================================
# SETTINGS
# ============================================================
CLASSES = ["Normal", "Pneumonia"]
MAKE_NUMBERS_FROM_PHOTOS = False  # True  = step 1 runs: MedSigLIP looks at all photos (about 20 min)
                                  # False = reuse the numbers saved by your last run (seconds)
                                  # MedSigLIP is frozen, so the saved numbers never change.
                                  # Only set it back to True after the photos change.
MEDSIGLIP_NAME = "google/medsiglip-448"
SAVE_PATH = config.MODEL_DIR / "probe_medsiglip-448_2class.pth"
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
PHOTOS_AT_ONCE = 32               # how many photos MedSigLIP looks at together
IMAGE_TYPES = [".png", ".jpg", ".jpeg"]


# ============================================================
# STEP 1: MedSigLIP turns every photo into 1152 numbers
# ============================================================
def load_medsiglip():
    """Load MedSigLIP. It is frozen: it is only used, never trained."""
    print(f"Loading {MEDSIGLIP_NAME} ...")
    if config.DEVICE == "cpu":
        number_type = torch.float32
    else:
        number_type = torch.float16      # half precision on the Mac GPU: faster
    processor = AutoProcessor.from_pretrained(MEDSIGLIP_NAME)
    model = AutoModel.from_pretrained(MEDSIGLIP_NAME, dtype=number_type)
    model = model.to(config.DEVICE)
    model.eval()
    return processor, model, number_type


def list_photos(split):
    """All photos of our 2 classes in one split, with their label (0 = Normal, 1 = Pneumonia)."""
    photos = []
    for label in range(len(CLASSES)):
        folder = config.CLEAN_DIR / split / CLASSES[label]
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_TYPES:
                photos.append((path, label))
    return photos


def make_numbers(split, processor, model, number_type):
    """Give every photo of one split to MedSigLIP. Return (numbers, labels)."""
    photos = list_photos(split)
    all_numbers = []
    all_labels = []

    with torch.no_grad():   # no learning
        for start in range(0, len(photos), PHOTOS_AT_ONCE):
            group = photos[start:start + PHOTOS_AT_ONCE]

            images = []
            for path, label in group:
                images.append(Image.open(path).convert("RGB"))
                all_labels.append(label)

            inputs = processor(images=images, return_tensors="pt").to(config.DEVICE, number_type)
            numbers = model.get_image_features(**inputs).pooler_output   # 1152 numbers per photo
            all_numbers.append(numbers.float().cpu())

            done = min(start + PHOTOS_AT_ONCE, len(photos))
            print(f"  {split}: {done} / {len(photos)} photos", end="\r")
    print()

    return torch.cat(all_numbers), torch.tensor(all_labels)


def get_numbers(split, medsiglip):
    """Make the numbers from the photos (and save them), or reuse the saved ones."""
    save_path = config.FEATURES_DIR / f"medsiglip-448_2class_{split}.pt"

    if MAKE_NUMBERS_FROM_PHOTOS or not save_path.exists():
        if medsiglip is None:
            medsiglip = load_medsiglip()
        processor, model, number_type = medsiglip
        numbers, labels = make_numbers(split, processor, model, number_type)
        config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": numbers, "labels": labels, "classes": CLASSES}, save_path)
    else:
        print(f"Using saved numbers: {save_path.name}")
        saved = torch.load(save_path)
        numbers, labels = saved["embeddings"], saved["labels"]

    # Give every row the same length (the same as in the 4-class training)
    numbers = torch.nn.functional.normalize(numbers, dim=1)
    return numbers, labels, medsiglip


# ============================================================
# STEP 2: Train the linear probe
# ============================================================
def train_probe(train_X, train_y, val_X, val_y):
    torch.manual_seed(config.SEED)        # same result every run

    # Put all 1152 numbers on the same scale (using the Training numbers only)
    mean = train_X.mean(dim=0)
    std = train_X.std(dim=0) + 1e-6
    train_X = (train_X - mean) / std
    val_X = (val_X - mean) / std

    # Sigmoid needs one YES/NO answer per class, not one class number.
    # Normal (class 0) becomes [1, 0]; Pneumonia (class 1) becomes [0, 1].
    train_y_per_class = torch.zeros(len(train_y), len(CLASSES))
    train_y_per_class[torch.arange(len(train_y)), train_y] = 1.0

    # The linear probe: 1152 numbers in -> 2 scores out (Normal, Pneumonia)
    probe = nn.Linear(train_X.shape[1], len(CLASSES))
    loss_function = nn.BCEWithLogitsLoss()
    # The old way (softmax). CrossEntropyLoss does softmax inside itself, so the 2 scores
    # were forced to add up to 1 and the probe could never say "none of these".
    # loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0
    best_epoch = 0
    best_weights = None

    for epoch in range(EPOCHS):
        # ---- Learn ----
        probe.train()
        order = torch.randperm(len(train_X))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()                          # 1. forget the old fixes
            guesses = probe(train_X[batch])                # 2. the probe guesses
            loss = loss_function(guesses, train_y_per_class[batch])  # 3. how wrong?
            loss.backward()                                # 4. find how to fix it
            optimizer.step()                               # 5. fix it a little

        # ---- Check on Validation ----
        # The highest score still wins, exactly as before: sigmoid does not change the
        # order of the scores, so the accuracy can be compared with the old runs.
        probe.eval()
        with torch.no_grad():
            val_scores = probe(val_X)
            predicted = val_scores.argmax(dim=1)
            # New with sigmoid: how often does the probe say "I do not know"?
            # (both classes below 50%). With softmax this was impossible.
            unsure = (torch.sigmoid(val_scores).max(dim=1).values < 0.5).float().mean().item()
        accuracy = (predicted == val_y).float().mean().item()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(probe.state_dict())
            print(f"Epoch {epoch + 1}: validation accuracy = {accuracy:.2%} "
                  f"(unsure: {unsure:.1%})  <- best so far")
        else:
            print(f"Epoch {epoch + 1}: validation accuracy = {accuracy:.2%} "
                  f"(unsure: {unsure:.1%})")

    probe.load_state_dict(best_weights)
    print(f"Best epoch: {best_epoch} ({best_accuracy:.2%})")
    return probe, mean, std, best_epoch


# ============================================================
# STEP 3: Test
# ============================================================
def test_probe(probe, mean, std, test_X, test_y):
    with torch.no_grad():
        predicted = probe((test_X - mean) / std).argmax(dim=1)

    print()
    print("TEST RESULTS")
    total_correct = 0
    for i in range(len(CLASSES)):
        correct = ((predicted == i) & (test_y == i)).sum().item()
        total = (test_y == i).sum().item()
        total_correct += correct
        print(f"  {CLASSES[i]:<10} {correct} / {total} correct ({correct / total:.1%})")
    print(f"  {'ALL':<10} {total_correct / len(test_y):.2%}")

    print()
    print("  Confusion matrix (rows = TRUE class, columns = PREDICTED class)")
    print(f"  {'':<10}" + "".join(f"{name:>12}" for name in CLASSES))
    for i in range(len(CLASSES)):
        row = f"  {CLASSES[i]:<10}"
        for j in range(len(CLASSES)):
            count = ((test_y == i) & (predicted == j)).sum().item()
            row += f"{count:>12}"
        print(row)


# ============================================================
# Run everything
# ============================================================
def main():
    print("Classes:", CLASSES)

    # ---- STEP 1: numbers for every photo ----
    medsiglip = None          # loaded only if we need it
    train_X, train_y, medsiglip = get_numbers("Training", medsiglip)
    val_X, val_y, medsiglip = get_numbers("Validation", medsiglip)
    test_X, test_y, medsiglip = get_numbers("Testing", medsiglip)
    print(f"Training: {len(train_y)} X-rays | Validation: {len(val_y)} | Testing: {len(test_y)}")
    print()

    # ---- STEP 2 and 3: train and test the probe ----

    probe, mean, std, best_epoch = train_probe(train_X, train_y, val_X, val_y)
    test_probe(probe, mean, std, test_X, test_y)

    # ---- STEP 4: Save ----
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": probe.state_dict(), "mean": mean, "std": std,
                "classes": CLASSES, "model_name": "google/medsiglip-448",
                "best_epoch": best_epoch}, SAVE_PATH)
    print()
    print("Saved:", SAVE_PATH)


if __name__ == "__main__":
    main()
