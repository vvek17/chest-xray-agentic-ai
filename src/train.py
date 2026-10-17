"""
Train the model.

Run from the chest-xray-pneumonia folder:
    python -m src.train
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the "src" folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from src import config
from src.data_loader import load_photos, make_batches
from src.model import build_model
from src.utils import save_model, set_seed


def learn_one_epoch(model, train_batches, loss_function, optimizer):
    """The model sees all training photos one time, and learns."""
    model.train()   # "learning mode"

    for photos, answers in train_batches:
        # photos  = 32 X-rays (as numbers)
        # answers = the 32 correct diseases (0 = Covid-19, 1 = Normal, 2 = Pneumonia)
        photos = photos.to(config.DEVICE)
        answers = answers.to(config.DEVICE)

        optimizer.zero_grad()                       # 1. forget the old fixes

        guesses = model(photos)                     # 2. the model guesses
        loss = loss_function(guesses, answers)      # 3. how wrong are the guesses?

        loss.backward()                             # 4. find how to fix the model
        optimizer.step()                            # 5. fix the model a little (= learning!)


def check_accuracy(model, batches):
    """Return the % of photos the model gets right. No learning here."""
    model.eval()    # "exam mode"

    number_correct = 0
    number_total = 0

    with torch.no_grad():   # only checking, no learning -> faster
        for photos, answers in batches:
            photos = photos.to(config.DEVICE)
            answers = answers.to(config.DEVICE)

            guesses = model(photos)                         # a score for each disease
            predicted = guesses.argmax(dim=1)               # pick the disease with the highest score

            correct_now = (predicted == answers).sum()      # count how many are right
            number_correct = number_correct + correct_now.item()
            number_total = number_total + len(answers)

    return number_correct / number_total


def main():
    # STEP 1: Same "random" every run, so results can be compared
    set_seed(config.SEED)

    # STEP 2: Load the photos
    train_photos = load_photos(config.TRAIN_DIR)
    val_photos = load_photos(config.VAL_DIR)

    print("Diseases:", train_photos.classes)
    print("Training photos:", len(train_photos))
    print("Validation photos:", len(val_photos))
    print("Device:", config.DEVICE)

    train_batches = make_batches(train_photos, shuffle=True)
    val_batches = make_batches(val_photos, shuffle=False)

    # STEP 3: Build the model and move it to the GPU
    model = build_model(pretrained=True)
    model = model.to(config.DEVICE)

    # STEP 4: Tools for learning
    loss_function = nn.CrossEntropyLoss()       # measures HOW WRONG the guesses are
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)  # fixes the model

    # STEP 5: Train, and keep the BEST model
    best_accuracy = 0

    for epoch in range(config.EPOCHS):
        learn_one_epoch(model, train_batches, loss_function, optimizer)
        accuracy = check_accuracy(model, val_batches)

        print(f"Epoch {epoch + 1}: validation accuracy = {accuracy:.2%}", end="")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, config.BEST_MODEL_PATH)
            print("  <- best so far, saved")
        else:
            print()

    print()
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")
    print("Next: python -m src.evaluate")


if __name__ == "__main__":
    main()
