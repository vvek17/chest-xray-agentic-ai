"""
Train the COMBINED model: the softmax probe and the sigmoid probe together.

The idea: we already have two probes. They are both MedSigLIP (frozen) plus a
small linear layer. One was trained with softmax, one with sigmoid. This script
lets both of them vote, and trains one tiny new layer to decide who to believe.

  one X-ray
    -> MedSigLIP (frozen)                 1152 numbers
    -> softmax probe (frozen)             4 softmax numbers  ┐
    -> sigmoid probe (frozen)             4 sigmoid numbers  ┘  put together: 8 numbers
    -> the NEW layer (8 numbers in, 4 scores out)   <- the only part that learns
    -> softmax                            4 confidences, they add up to 1

Only 36 numbers are learned (8 x 4 weights + 4 biases). The two old probes are
never changed: they are loaded read-only.

What you need first:
  models/probe_medsiglip-448_softmax.pth
  models/probe_medsiglip-448_sigmoid.pth
  outputs/features/medsiglip-448_{Training,Validation,Testing}.pt

What it saves:
  models/probe_medsiglip-448_combined.pth
  outputs/curves_combined_model.png

Run it from the chest-xray-pneumonia folder (NOT from Alzaimer-AgenticAI):
    cd ~/Desktop/Alzaimer-AgenticAI/chest-xray-pneumonia
    python -m training.train_combined_probe
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the project folders
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy

import torch
import torch.nn as nn

from backend import config

# ============================================================
# SETTINGS (the same ones as the two old probes, so the test is fair)
# ============================================================
SOFTMAX_PATH = config.MODEL_DIR / "probe_medsiglip-448_softmax.pth"
SIGMOID_PATH = config.MODEL_DIR / "probe_medsiglip-448_sigmoid.pth"
SAVE_PATH = config.MODEL_DIR / "probe_medsiglip-448_combined.pth"
CURVES_PATH = config.PROJECT_ROOT / "outputs" / "curves_combined_model.png"

MAX_EPOCHS = 200          # it stops early, so this is only an upper limit
PATIENCE = 20             # stop if the validation loss did not improve for 20 epochs
LEARNING_RATE = 0.001
BATCH_SIZE = 64


# ============================================================
# STEP 1: load the saved numbers and the two old probes
# ============================================================
def get_numbers(split):
    """The 1152 numbers of every X-ray in one split (made by train_probe.py)."""
    path = config.FEATURES_DIR / f"medsiglip-448_{split}.pt"
    if not path.exists():
        print("Missing:", path)
        print("Run  python -m training.train_probe  first to make the numbers.")
        sys.exit()
    saved = torch.load(path)
    # Same length for every row, exactly like the other training scripts
    numbers = torch.nn.functional.normalize(saved["embeddings"], dim=1)
    return numbers, saved["labels"].long()


def load_old_probe(path):
    """Load one old probe. It is FROZEN: it only votes, it never learns again."""
    if not path.exists():
        print("Missing:", path)
        print("This file is made by the softmax / sigmoid comparison run.")
        sys.exit()
    saved = torch.load(path)
    probe = nn.Linear(1152, config.NUM_CLASSES)
    probe.load_state_dict(saved["weights"])
    probe.eval()
    for weight in probe.parameters():
        weight.requires_grad = False
    return probe, saved["mean"], saved["std"]


def ask_old_probe(probe_parts, numbers, activation):
    """
    Ask one old probe what it thinks. Each probe keeps its OWN activation:
    the softmax probe answers with softmax, the sigmoid probe with sigmoid.
    """
    probe, mean, std = probe_parts
    with torch.no_grad():
        scores = probe((numbers - mean) / std)
    if activation == "softmax":
        return torch.softmax(scores, dim=1)
    return torch.sigmoid(scores)


def make_votes(numbers, softmax_parts, sigmoid_parts):
    """Put the two answers side by side: 4 + 4 = 8 numbers per X-ray."""
    return torch.cat([ask_old_probe(softmax_parts, numbers, "softmax"),
                      ask_old_probe(sigmoid_parts, numbers, "sigmoid")], dim=1)


# ============================================================
# STEP 2: train the new layer, and stop early
# ============================================================
def train_combined(train_votes, train_y, val_votes, val_y):
    torch.manual_seed(config.SEED)        # same result every run

    # 8 numbers in -> 4 scores out. Only these 36 numbers learn.
    combined = nn.Linear(train_votes.shape[1], config.NUM_CLASSES)
    loss_function = nn.CrossEntropyLoss()           # softmax is inside it
    optimizer = torch.optim.Adam(combined.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    epochs_without_progress = 0

    print(f"Training the new layer (stops early after {PATIENCE} epochs with no progress)")
    for epoch in range(1, MAX_EPOCHS + 1):
        # ---- Learn ----
        combined.train()
        order = torch.randperm(len(train_votes))
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()                           # 1. forget the old fixes
            guesses = combined(train_votes[batch])          # 2. the layer guesses
            loss = loss_function(guesses, train_y[batch])   # 3. how wrong?
            loss.backward()                                 # 4. find how to fix it
            optimizer.step()                                # 5. fix it a little
            total_loss += loss.item() * len(batch)

        # ---- Check on Validation ----
        combined.eval()
        with torch.no_grad():
            train_guesses = combined(train_votes)
            val_guesses = combined(val_votes)
            val_loss = loss_function(val_guesses, val_y).item()

        history["train_loss"].append(total_loss / len(train_votes))
        history["val_loss"].append(val_loss)
        history["train_acc"].append((train_guesses.argmax(dim=1) == train_y).float().mean().item())
        history["val_acc"].append((val_guesses.argmax(dim=1) == val_y).float().mean().item())

        # ---- Early stopping: keep the epoch with the LOWEST validation loss ----
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(combined.state_dict())
            epochs_without_progress = 0
        else:
            epochs_without_progress += 1
            if epochs_without_progress >= PATIENCE:
                print(f"  stopped early at epoch {epoch}")
                break

        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}: validation loss = {val_loss:.4f}, "
                  f"accuracy = {history['val_acc'][-1]:.2%}")

    combined.load_state_dict(best_weights)
    print(f"  Best epoch: {best_epoch} (validation loss {best_loss:.4f})")
    return combined, best_epoch, history


# ============================================================
# STEP 3: test, and draw the curves
# ============================================================
def test_combined(combined, test_votes, test_y):
    with torch.no_grad():
        predicted = combined(test_votes).argmax(dim=1)

    names = config.CLASS_NAMES
    print()
    print("TEST RESULTS")
    correct_total = 0
    for i, name in enumerate(names):
        correct = ((predicted == i) & (test_y == i)).sum().item()
        total = (test_y == i).sum().item()
        correct_total += correct
        print(f"  {name:<14} {correct:>4} / {total:<4} correct ({correct / total:.1%})")
    print(f"  {'ALL':<14} {correct_total / len(test_y):.2%}")

    print()
    print("  Confusion matrix (rows = TRUE disease, columns = PREDICTED disease)")
    print(f"  {'':<14}" + "".join(f"{n:>14}" for n in names))
    for i, name in enumerate(names):
        row = f"  {name:<14}"
        for j in range(len(names)):
            row += f"{((test_y == i) & (predicted == j)).sum().item():>14}"
        print(row)


def draw_curves(history, best_epoch):
    """Save the loss and accuracy pictures, so they can go in the report."""
    import matplotlib
    matplotlib.use("Agg")                 # save to a file, do not open a window
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, charts = plt.subplots(1, 2, figsize=(11, 4.2))

    for chart, key, title in [(charts[0], "loss", "loss"), (charts[1], "acc", "accuracy")]:
        chart.plot(epochs, history["train_" + key], label="train")
        chart.plot(epochs, history["val_" + key], linestyle="--", label="validation")
        chart.axvline(best_epoch, color="green", linestyle=":", label=f"best epoch {best_epoch}")
        chart.set_title("Combined model: " + title)
        chart.set_xlabel("epoch")
        chart.set_ylabel(title)
        chart.grid(alpha=0.3)
        chart.legend(fontsize=8)

    figure.tight_layout()
    CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CURVES_PATH, dpi=140)
    plt.close(figure)
    print("Curves saved:", CURVES_PATH)


# ============================================================
# Run everything
# ============================================================
def main():
    print("Loading the two old probes (frozen) ...")
    softmax_parts = load_old_probe(SOFTMAX_PATH)
    sigmoid_parts = load_old_probe(SIGMOID_PATH)

    votes = {}
    labels = {}
    for split in ["Training", "Validation", "Testing"]:
        numbers, labels[split] = get_numbers(split)
        votes[split] = make_votes(numbers, softmax_parts, sigmoid_parts)
        print(f"  {split:<11} {len(labels[split])} X-rays -> {votes[split].shape[1]} votes each")
    print()

    combined, best_epoch, history = train_combined(
        votes["Training"], labels["Training"], votes["Validation"], labels["Validation"])

    test_combined(combined, votes["Testing"], labels["Testing"])

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": combined.state_dict(),
                "input": "concat(softmax_probs, sigmoid_probs)",
                "base_models": [SOFTMAX_PATH.name, SIGMOID_PATH.name],
                "classes": config.CLASS_NAMES,
                "final_activation": "softmax",
                "best_epoch": best_epoch}, SAVE_PATH)
    print()
    print("Saved:", SAVE_PATH)

    draw_curves(history, best_epoch)


if __name__ == "__main__":
    main()
