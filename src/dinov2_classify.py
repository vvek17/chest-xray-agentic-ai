"""
Classify chest X-rays with a FROZEN pretrained model (no fine-tuning).
Works with DINOv2, RAD-DINO and MedSigLIP (choose DINOV2_NAME in config.py).

How it works:
  Step 1: The model looks at every X-ray and turns it into a list of numbers
          (an "embedding" that describes the image):
          768 numbers for DINOv2 / RAD-DINO, 1152 for MedSigLIP.
          The model itself never changes.
  Step 2: Two simple ways to choose the disease from those numbers:
          - k-NN: find the k most similar Training X-rays and take
                  the disease that most of them have.
          - Linear probe: a tiny classifier (logistic regression)
                  that learns from the numbers in a few seconds.
          - Linear probe with epochs: the same tiny layer, trained like
                  train.py (10 epochs, keep the best epoch on Validation).
                  The best one is saved in models/probe_<model name>.pth
          - Zero-shot (MedSigLIP only): no training at all. Compare the
                  X-ray with one sentence per disease and pick the closest.
  Step 3: Test all of them on the Testing X-rays.

The embeddings are saved in outputs/features/, so step 1 runs only once.

Run it from the chest-xray-pneumonia folder:
    python -m src.dinov2_classify
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the "src" folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy

import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

from src import config

IMAGE_TYPES = [".png", ".jpg", ".jpeg"]
BATCH_SIZE = 32


def is_medsiglip():
    """True if the chosen model is MedSigLIP (it works a little differently)."""
    return "siglip" in config.DINOV2_NAME.lower()


def number_type():
    """MedSigLIP on the Mac GPU uses half precision: 2x faster, almost the same numbers."""
    if is_medsiglip() and config.DEVICE != "cpu":
        return torch.float16
    return torch.float32


# ============================================================
# STEP 1: Turn every X-ray into a list of numbers
# ============================================================
def load_dinov2():
    """Download (first time only) and load the model. It is frozen: no learning."""
    print(f"Loading {config.DINOV2_NAME} ...")
    if is_medsiglip():
        processor = AutoProcessor.from_pretrained(config.DINOV2_NAME)       # images AND text
    else:
        processor = AutoImageProcessor.from_pretrained(config.DINOV2_NAME)  # images only
    model = AutoModel.from_pretrained(config.DINOV2_NAME, dtype=number_type())
    model = model.to(config.DEVICE)
    model.eval()    # "exam mode": the model is only used, never trained
    return processor, model


def list_images(folder):
    """Return a list of (path, label number) for one split folder."""
    images = []
    for label, disease in enumerate(config.CLASS_NAMES):
        for path in sorted((folder / disease).iterdir()):
            if path.suffix.lower() in IMAGE_TYPES:
                images.append((path, label))
    return images


def make_embeddings(folder, processor, model):
    """Give every X-ray in a folder to DINOv2. Return the numbers and the labels."""
    images = list_images(folder)
    all_embeddings = []
    all_labels = []

    with torch.no_grad():   # no learning
        for start in range(0, len(images), BATCH_SIZE):
            batch = images[start:start + BATCH_SIZE]

            photos = []
            for path, label in batch:
                photos.append(Image.open(path).convert("RGB"))
                all_labels.append(label)

            inputs = processor(images=photos, return_tensors="pt").to(config.DEVICE, number_type())

            if is_medsiglip():
                # MedSigLIP gives one summary of the image: 1152 numbers
                embeddings = model.get_image_features(**inputs).pooler_output
            else:
                # DINOv2 / RAD-DINO: the first token ("CLS") summarizes the image: 768 numbers
                embeddings = model(**inputs).last_hidden_state[:, 0]
            all_embeddings.append(embeddings.float().cpu())

            print(f"  {folder.name}: {min(start + BATCH_SIZE, len(images))} / {len(images)}", end="\r")

    print()
    return torch.cat(all_embeddings), torch.tensor(all_labels)


def get_embeddings(split, processor, model):
    """Load saved embeddings if we have them, else make them and save them."""
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # The file name includes the model name, so each model has its own saved numbers.
    # Example: "microsoft/rad-dino" -> "rad-dino_Training.pt"
    model_short_name = config.DINOV2_NAME.split("/")[-1]
    save_path = config.FEATURES_DIR / f"{model_short_name}_{split}.pt"

    if save_path.exists():
        print(f"Using saved embeddings: {save_path.name}")
        saved = torch.load(save_path)
        return saved["embeddings"], saved["labels"]

    print(f"Making embeddings for {split} ...")
    embeddings, labels = make_embeddings(config.CLEAN_DIR / split, processor, model)
    torch.save({"embeddings": embeddings, "labels": labels}, save_path)
    return embeddings, labels


# ============================================================
# STEP 3: Print the results
# ============================================================
def print_results(title, true_labels, predicted_labels):
    """Print the accuracy for each disease and the confusion matrix."""
    n = config.NUM_CLASSES
    names = config.CLASS_NAMES

    # confusion[true][predicted] = how many photos
    confusion = []
    for i in range(n):
        confusion.append([0] * n)
    for true, predicted in zip(true_labels, predicted_labels):
        confusion[int(true)][int(predicted)] += 1

    print()
    print(title)
    total_correct = 0
    total_photos = 0
    for i in range(n):
        correct = confusion[i][i]
        total = sum(confusion[i])
        total_correct += correct
        total_photos += total
        print(f"  {names[i]:<13} {correct:>4} / {total:<4} correct ({correct / total:.1%})")
    print(f"  {'ALL':<13} {total_correct / total_photos:.2%}")

    print()
    print("  Confusion matrix (rows = TRUE disease, columns = PREDICTED disease)")
    header = f"  {'':<13}"
    for name in names:
        header += f"{name:>14}"
    print(header)
    for i in range(n):
        row = f"  {names[i]:<13}"
        for j in range(n):
            row += f"{confusion[i][j]:>14}"
        print(row)


# ============================================================
# STEP 2c: Linear probe trained with epochs (like train.py)
# Only this small layer learns. The big model stays frozen.
# ============================================================
def train_probe_with_epochs(train_X, train_y, val_X, val_y):
    """Train a linear layer for EPOCHS epochs and keep the best epoch on Validation."""
    # From numpy to PyTorch
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    val_X = torch.tensor(val_X, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.long)

    # Put all 768 numbers on the same scale (using the Training numbers only)
    mean = train_X.mean(dim=0)
    std = train_X.std(dim=0) + 1e-6
    train_X = (train_X - mean) / std
    val_X = (val_X - mean) / std

    # The linear probe: 768 numbers in -> 4 scores out (one per disease)
    probe = nn.Linear(train_X.shape[1], config.NUM_CLASSES)
    loss_function = nn.CrossEntropyLoss()                            # how wrong the guesses are
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)       # fixes the probe a little

    best_accuracy = 0
    best_epoch = 0
    best_weights = None

    print()
    print(f"Linear probe with epochs: training for {config.EPOCHS} epochs")
    for epoch in range(config.EPOCHS):
        # ---- Learn: go through all Training numbers in a new random order ----
        probe.train()
        order = torch.randperm(len(train_X))
        for start in range(0, len(order), 64):           # batches of 64
            batch = order[start:start + 64]
            optimizer.zero_grad()                         # 1. forget the old fixes
            guesses = probe(train_X[batch])               # 2. the probe guesses
            loss = loss_function(guesses, train_y[batch]) # 3. how wrong?
            loss.backward()                               # 4. find how to fix it
            optimizer.step()                              # 5. fix it a little

        # ---- Check on Validation ----
        probe.eval()
        with torch.no_grad():
            predicted = probe(val_X).argmax(dim=1)
        val_accuracy = (predicted == val_y).float().mean().item()

        print(f"  Epoch {epoch + 1}: validation accuracy = {val_accuracy:.2%}", end="")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(probe.state_dict())   # remember the best epoch
            print("  <- best so far")
        else:
            print()

    # Go back to the best epoch
    probe.load_state_dict(best_weights)
    print(f"  Best epoch: {best_epoch} ({best_accuracy:.2%})")

    # Save the best probe (and the scale numbers) for later, e.g. predict.py
    model_short_name = config.DINOV2_NAME.split("/")[-1]
    save_path = config.MODEL_DIR / f"probe_{model_short_name}.pth"
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": probe.state_dict(), "mean": mean, "std": std,
                "model_name": config.DINOV2_NAME, "best_epoch": best_epoch}, save_path)
    print(f"  Saved: {save_path}")

    return probe, mean, std


def probe_predict(probe, mean, std, X):
    """Use the trained probe to pick a disease for each row of numbers."""
    X = torch.tensor(X, dtype=torch.float32)
    X = (X - mean) / std
    with torch.no_grad():
        return probe(X).argmax(dim=1).numpy()


# ============================================================
# STEP 2d: Zero-shot (MedSigLIP only) - no training at all
# ============================================================
def zero_shot_predict(processor, model, image_X):
    """Compare each X-ray with one sentence per disease, and pick the closest sentence."""
    sentences = []
    for disease in config.CLASS_NAMES:
        sentences.append(config.ZERO_SHOT_PROMPTS[disease])

    print()
    print("Zero-shot sentences:")
    for disease, sentence in zip(config.CLASS_NAMES, sentences):
        print(f"  {disease:<13} \"{sentence}\"")

    # Turn the 4 sentences into numbers (the same kind of numbers as the images)
    text_inputs = processor(text=sentences, padding="max_length", return_tensors="pt").to(config.DEVICE)
    with torch.no_grad():
        text_X = model.get_text_features(**text_inputs).pooler_output.float().cpu()
    text_X = torch.nn.functional.normalize(text_X, dim=1)

    # How similar is each X-ray to each sentence? (higher = more similar)
    image_X = torch.tensor(image_X, dtype=torch.float32)     # already normalized in main()
    similarity = image_X @ text_X.T                          # one row per X-ray, one column per disease
    return similarity.argmax(dim=1).numpy()                  # the most similar sentence wins


def accuracy(true_labels, predicted_labels):
    correct = 0
    for true, predicted in zip(true_labels, predicted_labels):
        if int(true) == int(predicted):
            correct += 1
    return correct / len(true_labels)


# ============================================================
# Run everything
# ============================================================
def main():
    torch.manual_seed(config.SEED)

    # ---- Step 1: embeddings for the 3 parts ----
    processor, model = load_dinov2()
    train_X, train_y = get_embeddings("Training", processor, model)
    val_X, val_y = get_embeddings("Validation", processor, model)
    test_X, test_y = get_embeddings("Testing", processor, model)
    print(f"Each X-ray is now {train_X.shape[1]} numbers.")

    # Make every embedding the same "length", so k-NN compares directions
    train_X = torch.nn.functional.normalize(train_X, dim=1).numpy()
    val_X = torch.nn.functional.normalize(val_X, dim=1).numpy()
    test_X = torch.nn.functional.normalize(test_X, dim=1).numpy()
    train_y = train_y.numpy()
    val_y = val_y.numpy()
    test_y = test_y.numpy()

    # ---- Step 2a: k-NN ----
    # Try a few values of k on Validation, and keep the best one.
    print()
    print("k-NN: choosing k on the Validation X-rays")
    best_k = None
    best_val_accuracy = 0
    for k in [1, 5, 10, 20, 50]:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(train_X, train_y)
        val_accuracy = accuracy(val_y, knn.predict(val_X))
        print(f"  k = {k:<3} validation accuracy = {val_accuracy:.2%}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_k = k
    print(f"  Best k = {best_k}")

    knn = KNeighborsClassifier(n_neighbors=best_k, metric="cosine")
    knn.fit(train_X, train_y)
    print_results(f"k-NN (k = {best_k}): TEST RESULTS", test_y, knn.predict(test_X))

    # ---- Step 2b: Linear probe (logistic regression) ----
    scaler = StandardScaler()                      # put all 768 numbers on the same scale
    train_scaled = scaler.fit_transform(train_X)
    val_scaled = scaler.transform(val_X)
    test_scaled = scaler.transform(test_X)

    probe = LogisticRegression(max_iter=3000)
    probe.fit(train_scaled, train_y)

    print()
    print(f"Linear probe: validation accuracy = {accuracy(val_y, probe.predict(val_scaled)):.2%}")
    print_results("Linear probe: TEST RESULTS", test_y, probe.predict(test_scaled))

    # ---- Step 2c: Linear probe with epochs, keep the best epoch ----
    epoch_probe, mean, std = train_probe_with_epochs(train_X, train_y, val_X, val_y)
    print_results("Linear probe with epochs (best epoch): TEST RESULTS",
                  test_y, probe_predict(epoch_probe, mean, std, test_X))

    # ---- Step 2d: Zero-shot (only MedSigLIP understands words) ----
    if is_medsiglip():
        print_results("Zero-shot (no training at all): TEST RESULTS",
                      test_y, zero_shot_predict(processor, model, test_X))


if __name__ == "__main__":
    main()
