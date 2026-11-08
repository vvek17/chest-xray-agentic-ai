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
    python -m training.train_probe

For the NEW frozen MedSigLIP + MedGemma model and NEW dataset splits:
    python -m training.train_probe --new-model
It caches features once, runs the baselines, trains the two linear heads for
10 epochs, saves the best validation epoch, and tests the frozen results once.
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the project folders
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy

import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

from backend import config

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

    # Sigmoid needs one YES/NO answer per class, not one class number.
    # Covid-19 (class 0) becomes [1, 0, 0, 0]; Normal (class 1) becomes [0, 1, 0, 0].
    train_y_per_class = torch.zeros(len(train_y), config.NUM_CLASSES)
    train_y_per_class[torch.arange(len(train_y)), train_y] = 1.0

    # The linear probe: 768 numbers in -> 4 scores out (one per disease)
    probe = nn.Linear(train_X.shape[1], config.NUM_CLASSES)
    loss_function = nn.BCEWithLogitsLoss()                           # how wrong the guesses are
    # The old way (softmax). CrossEntropyLoss does softmax inside itself, so the 4 scores
    # were forced to add up to 1 and the probe could never say "none of these".
    # loss_function = nn.CrossEntropyLoss()
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
            loss = loss_function(guesses, train_y_per_class[batch])  # 3. how wrong?
            loss.backward()                               # 4. find how to fix it
            optimizer.step()                              # 5. fix it a little

        # ---- Check on Validation ----
        # The highest score still wins, exactly as before: sigmoid does not change the
        # order of the scores, so the accuracy can be compared with the old runs.
        probe.eval()
        with torch.no_grad():
            val_scores = probe(val_X)
            predicted = val_scores.argmax(dim=1)
            # New with sigmoid: how often does the probe say "I do not know"?
            # (every class below 50%). With softmax this was impossible.
            unsure = (torch.sigmoid(val_scores).max(dim=1).values < 0.5).float().mean().item()
        val_accuracy = (predicted == val_y).float().mean().item()

        print(f"  Epoch {epoch + 1}: validation accuracy = {val_accuracy:.2%} "
              f"(unsure: {unsure:.1%})", end="")
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
# NEW MODEL: MedSigLIP + MedGemma, with softmax and sigmoid heads
# Only the tiny heads learn. Both large models stay frozen.
# ============================================================
def train_new_probe_with_epochs(train_X, train_y, val_X, val_y, classes):
    """Train for exactly 10 epochs and keep the best validation epoch."""
    from backend.dual_backbone import FusedHeads

    torch.manual_seed(42)
    order_generator = torch.Generator().manual_seed(42)
    number_of_classes = len(classes)

    # The two linear heads share the same combined image features.
    probe = FusedHeads(train_X.shape[1], number_of_classes)
    probe.fit_scaler(train_X)  # learn mean/std from Training only

    # Give each class a fair contribution to the two losses.
    counts = torch.bincount(train_y, minlength=number_of_classes).float()
    class_weights = len(train_y) / (number_of_classes * counts)
    positive_weights = (len(train_y) - counts) / counts
    softmax_loss = nn.CrossEntropyLoss(weight=class_weights)
    sigmoid_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    train_targets = nn.functional.one_hot(train_y, number_of_classes).float()
    val_targets = nn.functional.one_hot(val_y, number_of_classes).float()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001, weight_decay=0.0001)

    best_score = None
    best_weights = None
    best_epoch = 0
    history = []

    for epoch in range(1, 11):
        # 1. Learn from the Training features, in batches of 128.
        probe.train()
        order = torch.randperm(len(train_y), generator=order_generator)
        total_loss = 0.0
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            optimizer.zero_grad()
            softmax_scores, sigmoid_scores = probe(train_X[batch])
            loss = softmax_loss(softmax_scores, train_y[batch])
            loss = loss + sigmoid_loss(sigmoid_scores, train_targets[batch])
            if not torch.isfinite(loss):
                raise ValueError("Training loss contains NaN or infinity")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        # 2. Check Validation. Nothing learns from these images.
        probe.eval()
        with torch.no_grad():
            softmax_scores, sigmoid_scores = probe(val_X)
            val_loss = softmax_loss(softmax_scores, val_y)
            val_loss = float(val_loss + sigmoid_loss(sigmoid_scores, val_targets))
            probabilities = probe.probabilities(val_X)["averaged"]
            predicted = probabilities.argmax(dim=1).numpy()

        macro_f1 = float(f1_score(val_y.numpy(), predicted, average="macro", zero_division=0))
        balanced = float(balanced_accuracy_score(val_y.numpy(), predicted))
        score = (macro_f1, balanced, -val_loss)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_y),
                        "val_loss": val_loss, "val_macro_f1": macro_f1,
                        "val_balanced_accuracy": balanced})

        # 3. Remember the best epoch, using Validation only.
        improved = best_score is None or score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_weights = copy.deepcopy(probe.state_dict())
        print(f"Epoch {epoch:2}/10: validation macro-F1 = {macro_f1:.2%}, "
              f"balanced accuracy = {balanced:.2%}" + (" <- best" if improved else ""), flush=True)

    probe.load_state_dict(best_weights)
    details = {"lr": 0.001, "weight_decay": 0.0001, "best_epoch": best_epoch,
               "epochs_run": 10, "validation_macro_f1": best_score[0],
               "validation_balanced_accuracy": best_score[1], "history": history}
    print(f"Best epoch: {best_epoch} of 10", flush=True)
    return probe.eval(), details, class_weights.tolist(), positive_weights.tolist()


def main_new_model():
    """The same three steps as the original experiment, using the new model."""
    from training import run_new_model_baselines as experiment

    print("NEW MODEL: frozen MedSigLIP + frozen MedGemma", flush=True)
    print("New dataset: data/new_distribution", flush=True)
    # The helper handles cached features, saving models and result files.
    # Step 1: get features once. Step 2: train/select. Step 3: test once.
    experiment.main(["--stage", "all"])


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
    if "--new-model" in sys.argv:
        main_new_model()
    else:
        main()
