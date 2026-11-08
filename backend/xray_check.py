"""
Check every uploaded photo, and only give an answer when it is reliably
"Normal" or "Pneumonia" (config.SUPPORTED_LABELS). Everything else is
rejected or marked for review, with a neutral message.

  Check 1: Is it a chest X-ray at all?
           Similarity to the average training X-ray of each class.
           Below config.XRAY_SIMILARITY_LIMIT -> REJECTED.
  Check 2: Is it inside the supported labels?
           The 4-class model gives the probability of classes we do not support.
           At or above config.OUT_OF_SCOPE_LIMIT (or the top class is unsupported)
           -> NEEDS REVIEW.
  Check 3: Is the 2-class model confident?
           Below config.CONFIDENCE_LIMIT -> NEEDS REVIEW.
  All passed -> ACCEPTED, with the label and its confidence.

This is a research prototype, not a medical diagnosis.
Photo files are never changed (cleaning happens in memory only).

Try it from the chest-xray-pneumonia folder:
    python -m backend.xray_check path/to/photo.png
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the project folders
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from backend import config
from backend.clean_image import clean_xray
from backend.predict import PROBE_2_CLASSES, PROBE_4_CLASSES, load_classifier

AVERAGES_PATH = config.MODEL_DIR / "xray_averages_medsiglip-448.pt"

# The 3 possible results
ACCEPTED = "accepted"
REJECTED = "rejected"
NEEDS_REVIEW = "needs_review"

# Neutral messages: they never name any label outside config.SUPPORTED_LABELS
MESSAGE_NOT_XRAY = ("This image does not appear to be a chest X-ray, so it was not classified. "
                    "Please upload a chest X-ray image.")
MESSAGE_OUT_OF_SCOPE = ("This X-ray could not be classified reliably as Normal or Pneumonia. "
                        "It needs review by a qualified professional.")
MESSAGE_LOW_CONFIDENCE = ("The model is not confident enough to classify this X-ray reliably, "
                          "so no result is given. It needs review by a qualified professional.")
MESSAGE_INVALID_FILE = ("This file could not be opened as an image, so it was not classified. "
                        "Please upload a PNG or JPG chest X-ray.")
DISCLAIMER = "Research prototype: this is not a medical diagnosis."


# ============================================================
# The decision (no model inside, so it is easy to test)
# ============================================================
def decide(xray_score, four, two):
    """
    xray_score: similarity to the training X-rays (0 to 1)
    four: class -> probability from the 4-class model (all classes)
    two:  label -> probability from the 2-class model (Normal, Pneumonia)
    Returns a result dictionary. It only ever contains the supported labels.
    """
    supported = config.SUPPORTED_LABELS

    # ---- Check 1: is it a chest X-ray? ----
    if xray_score < config.XRAY_SIMILARITY_LIMIT:
        return {"status": REJECTED, "reason": "not_chest_xray", "message": MESSAGE_NOT_XRAY,
                "xray_score": xray_score}

    # ---- Check 2: inside the supported labels? ----
    out_of_scope_score = 0.0
    for name, probability in four.items():
        if name not in supported:
            out_of_scope_score += probability
    top_four = max(four, key=four.get)
    if top_four not in supported or out_of_scope_score >= config.OUT_OF_SCOPE_LIMIT:
        return {"status": NEEDS_REVIEW, "reason": "outside_supported_labels",
                "message": MESSAGE_OUT_OF_SCOPE, "xray_score": xray_score,
                "out_of_scope_score": out_of_scope_score}

    # ---- Check 3: confident enough? ----
    label = max(two, key=two.get)
    confidence = two[label]
    if label not in supported or confidence < config.CONFIDENCE_LIMIT:
        return {"status": NEEDS_REVIEW, "reason": "low_confidence",
                "message": MESSAGE_LOW_CONFIDENCE, "xray_score": xray_score,
                "out_of_scope_score": out_of_scope_score, "confidence": confidence}

    # ---- All checks passed ----
    return {"status": ACCEPTED, "reason": "all_checks_passed",
            "label": label, "confidence": confidence,
            "message": f"Model output: {label} ({confidence:.1%} confidence). {DISCLAIMER}",
            "probabilities": {name: two[name] for name in supported},
            "xray_score": xray_score, "out_of_scope_score": out_of_scope_score}


# ============================================================
# Loading the models
# ============================================================
def make_averages():
    """The average numbers of the training X-rays, one per class (made once and saved)."""
    saved = torch.load(config.FEATURES_DIR / "medsiglip-448_Training.pt")
    numbers = torch.nn.functional.normalize(saved["embeddings"], dim=1)
    labels = saved["labels"]

    averages = []
    for k in range(len(config.CLASS_NAMES)):
        averages.append(numbers[labels == k].mean(dim=0))
    averages = torch.nn.functional.normalize(torch.stack(averages), dim=1)

    torch.save({"averages": averages, "classes": config.CLASS_NAMES}, AVERAGES_PATH)
    return averages


def load_checker():
    """Load MedSigLIP once, plus both probes and the training averages."""
    four = load_classifier(PROBE_4_CLASSES)        # loads MedSigLIP + the 4-class probe
    two_saved = torch.load(PROBE_2_CLASSES)        # the 2-class probe uses the same MedSigLIP
    two = torch.nn.Linear(1152, len(two_saved["classes"]))
    two.load_state_dict(two_saved["weights"])
    two.eval()

    if AVERAGES_PATH.exists():
        averages = torch.load(AVERAGES_PATH)["averages"]
    else:
        averages = make_averages()

    return {
        "processor": four["processor"], "model": four["model"], "number_type": four["number_type"],
        "four": {"probe": four["probe"], "mean": four["mean"], "std": four["std"], "classes": four["classes"]},
        "two": {"probe": two, "mean": two_saved["mean"], "std": two_saved["std"], "classes": two_saved["classes"]},
        "averages": averages,
    }


# ============================================================
# From a photo to a result
# ============================================================
def photo_to_numbers(checker, image):
    """Clean the photo (in memory), then MedSigLIP turns it into 1152 numbers (same length)."""
    image = clean_xray(image)
    with torch.no_grad():
        inputs = checker["processor"](images=[image], return_tensors="pt")
        inputs = inputs.to(config.DEVICE, checker["number_type"])
        numbers = checker["model"].get_image_features(**inputs).pooler_output.float().cpu()
    return torch.nn.functional.normalize(numbers, dim=1)


def use_probe(probe_info, numbers):
    """
    Apply one probe. Returns class -> score between 0 and 1.

    Sigmoid: every class gets its own score. They do NOT add up to 1.
    So all scores can be low at once, which means "I do not know".
    """
    with torch.no_grad():
        scaled = (numbers - probe_info["mean"]) / probe_info["std"]
        probabilities = torch.sigmoid(probe_info["probe"](scaled))[0]
        # The old way (softmax): the scores were forced to add up to 1.
        # probabilities = torch.softmax(probe_info["probe"](scaled), dim=1)[0]
    result = {}
    for i, name in enumerate(probe_info["classes"]):
        result[name] = probabilities[i].item()
    return result


def check_xray(checker, image):
    """Run all checks on one photo (a path or an open image). Every upload must go through this."""
    numbers = photo_to_numbers(checker, image)
    xray_score = (numbers @ checker["averages"].T).max().item()
    four = use_probe(checker["four"], numbers)
    two = use_probe(checker["two"], numbers)
    return decide(xray_score, four, two)


def analyze_upload(checker, uploaded_file):
    """
    The web page calls this for EVERY upload.
    Returns (result, cleaned image or None). A file that is not a valid image is rejected.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        original = Image.open(uploaded_file)
        original.load()                      # really read the file, so broken files fail here
    except (UnidentifiedImageError, OSError, ValueError):
        return {"status": REJECTED, "reason": "invalid_file", "message": MESSAGE_INVALID_FILE}, None

    cleaned = clean_xray(original)
    return check_xray(checker, cleaned), cleaned


def medgemma_label(answer):
    """
    Read MedGemma's answer. Return "Normal" or "Pneumonia" only if the answer clearly
    says exactly one of them; otherwise None (the answer is then not shown).
    """
    text = answer.lower()
    if "answer:" in text:
        text = text.split("answer:")[-1]

    found = []
    for label in config.SUPPORTED_LABELS:
        if label.lower() in text:
            found.append(label)
    if len(found) == 1:
        return found[0]
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.xray_check path/to/photo.png")
        sys.exit()

    checker = load_checker()
    result = check_xray(checker, sys.argv[1])
    print()
    print("Status :", result["status"])
    print("Message:", result["message"])
