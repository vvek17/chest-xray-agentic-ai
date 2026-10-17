"""
The final exam: test the saved model on the TESTING photos.
The model has never seen these photos.

Run from the chest-xray-pneumonia folder (after training):
    python -m src.evaluate
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the "src" folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src import config
from src.data_loader import load_photos, make_batches
from src.model import build_model
from src.utils import load_model


def test_each_disease(model, photos):
    """Check every photo, then print the accuracy and the confusion matrix."""
    batches = make_batches(photos, shuffle=False)
    names = photos.classes

    # Confusion matrix: a table that counts, for each TRUE disease,
    # what the model PREDICTED.
    # confusion[true][predicted] = how many photos
    confusion = []
    for i in range(config.NUM_CLASSES):
        confusion.append([0] * config.NUM_CLASSES)

    with torch.no_grad():   # only checking, no learning
        for photo_batch, answers in batches:
            guesses = model(photo_batch.to(config.DEVICE))
            predicted = guesses.argmax(dim=1).cpu()     # the disease with the highest score

            for i in range(len(answers)):
                true_disease = answers[i].item()
                guessed_disease = predicted[i].item()
                confusion[true_disease][guessed_disease] += 1

    # ---- Accuracy for each disease ----
    total_correct = 0
    total_photos = 0
    for i in range(config.NUM_CLASSES):
        correct = confusion[i][i]           # the diagonal = correct answers
        total = sum(confusion[i])
        total_correct += correct
        total_photos += total
        print(f"  {names[i]:<13} {correct:>4} / {total:<4} correct ({correct / total:.1%})")
    print(f"  {'ALL':<13} {total_correct / total_photos:.2%}")

    # ---- Confusion matrix ----
    print()
    print("  Confusion matrix (rows = TRUE disease, columns = PREDICTED disease)")
    header = f"  {'':<13}"
    for name in names:
        header += f"{name:>14}"
    print(header)
    for i in range(config.NUM_CLASSES):
        row = f"  {names[i]:<13}"
        for j in range(config.NUM_CLASSES):
            row += f"{confusion[i][j]:>14}"
        print(row)


def main():
    # STEP 0: Is there a trained model? (train.py makes it)
    if not config.BEST_MODEL_PATH.exists():
        print(f"No trained model found: {config.BEST_MODEL_PATH}")
        print("Run train.py first, then run evaluate.py again.")
        return

    # STEP 1: Build an empty model, then fill it with what we learned
    model = build_model(pretrained=False)
    model = load_model(model, config.BEST_MODEL_PATH, config.DEVICE)   # also switches to "exam mode"
    print(f"Loaded model: {config.BEST_MODEL_PATH}")

    # STEP 2: Practice test (Validation photos)
    print()
    print("VALIDATION RESULTS")
    test_each_disease(model, load_photos(config.VAL_DIR))

    # STEP 3: Final exam (Testing photos)
    print()
    print("TEST RESULTS (final exam)")
    test_each_disease(model, load_photos(config.TEST_DIR))


if __name__ == "__main__":
    main()
