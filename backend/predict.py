"""
Prediction mode: one X-ray in -> disease and confidence out.

Uses our classifier:
  clean the photo -> MedSigLIP (frozen) -> 1152 numbers -> linear probe
  (2 classes: models/probe_medsiglip-448_2class.pth, 4 classes: models/probe_medsiglip-448.pth)

The COMBINED model works the same way, but two old probes vote first and a small
layer decides (models/probe_medsiglip-448_combined.pth, made by
training/train_combined_probe.py).

Run it on one X-ray from the chest-xray-pneumonia folder:
    python -m backend.predict path/to/xray.png              all the safety checks
    python -m backend.predict path/to/xray.png --combined   the combined model only
    python -m backend.predict path/to/xray.png --4class     the 4-class probe only
    python -m backend.predict path/to/xray.png --2class     the 2-class probe only
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the project folders
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor

from backend import config
from backend.clean_image import clean_xray

MEDSIGLIP_NAME = "google/medsiglip-448"

# The two saved probes (both use the same frozen MedSigLIP)
PROBE_4_CLASSES = config.MODEL_DIR / "probe_medsiglip-448.pth"          # Covid-19, Normal, Pneumonia, Tuberculosis
PROBE_2_CLASSES = config.MODEL_DIR / "probe_medsiglip-448_2class.pth"   # Normal, Pneumonia (train_2class_probe.py)
PROBE_COMBINED = config.MODEL_DIR / "probe_medsiglip-448_combined.pth"  # softmax probe + sigmoid probe together
PROBE_PATH = PROBE_4_CLASSES


class CombinedProbe(nn.Module):
    """
    The combined model, wrapped so it looks like a normal probe.

    1152 numbers in -> the two old probes vote (4 + 4 = 8 numbers)
                    -> the small final layer -> 4 scores out

    Each old probe keeps its own scaling numbers and its own activation,
    exactly as it was trained. They are frozen: they never learn again.
    """

    def __init__(self, saved):
        super().__init__()
        self.voters = nn.ModuleList()
        self.means = []
        self.stds = []
        self.activations = []

        for name in saved["base_models"]:
            old = torch.load(config.MODEL_DIR / name)
            voter = nn.Linear(1152, len(saved["classes"]))
            voter.load_state_dict(old["weights"])
            voter.eval()
            self.voters.append(voter)
            self.means.append(old["mean"])
            self.stds.append(old["std"])
            # "softmax" or "sigmoid": how that probe was trained
            self.activations.append(old.get("activation", "sigmoid" if "sigmoid" in name else "softmax"))

        self.final = nn.Linear(len(saved["base_models"]) * len(saved["classes"]),
                               len(saved["classes"]))
        self.final.load_state_dict(saved["weights"])
        self.eval()

    def forward(self, numbers):
        votes = []
        for voter, mean, std, activation in zip(self.voters, self.means,
                                                self.stds, self.activations):
            scores = voter((numbers - mean) / std)
            if activation == "softmax":
                votes.append(torch.softmax(scores, dim=1))
            else:
                votes.append(torch.sigmoid(scores))
        return self.final(torch.cat(votes, dim=1))


def load_classifier(probe_path=PROBE_PATH, revision=None):
    """Load MedSigLIP (frozen) and one saved probe (4-class, 2-class or combined)."""
    print(f"Loading {MEDSIGLIP_NAME} and the linear probe {Path(probe_path).name} ...")

    # Use half precision on the Mac GPU (faster, almost the same numbers)
    if config.DEVICE == "cpu":
        number_type = torch.float32
    else:
        number_type = torch.float16

    processor = AutoProcessor.from_pretrained(MEDSIGLIP_NAME, revision=revision)
    model = AutoModel.from_pretrained(MEDSIGLIP_NAME, revision=revision, dtype=number_type)
    model = model.to(config.DEVICE)
    model.eval()

    # The saved probe: its weights + the numbers used to scale the embeddings
    saved = torch.load(probe_path)
    classes = saved.get("classes", config.CLASS_NAMES)   # the 4-class probe has no "classes" saved

    if "base_models" in saved:
        # The combined model does its own scaling inside, one set per old probe,
        # so here we scale by nothing (take away 0, divide by 1).
        probe = CombinedProbe(saved)
        mean = torch.zeros(1152)
        std = torch.ones(1152)
        activation = "softmax"           # one disease per X-ray, so the scores add up to 1
    else:
        probe = nn.Linear(1152, len(classes))
        probe.load_state_dict(saved["weights"])
        probe.eval()
        mean, std = saved["mean"], saved["std"]
        activation = "sigmoid"           # see the note in predict()

    return {"processor": processor, "model": model, "number_type": number_type,
            "probe": probe, "mean": mean, "std": std, "classes": classes,
            "activation": activation}


def predict(classifier, image):
    """
    image: a photo path, or an image that is already open (any X-ray photo).
    Returns a dictionary: disease -> confidence (0 to 1), highest first.

    Every photo is prepared the same way as the training photos:
      clean (gray + empty borders cut) -> MedSigLIP's processor (448 x 448, normalize)
      -> same length -> same scale (training mean and std) -> probe
    """
    # 0. Clean the photo first, so it looks like the training photos
    image = clean_xray(image)

    with torch.no_grad():
        # 1. MedSigLIP turns the X-ray into 1152 numbers
        inputs = classifier["processor"](images=[image], return_tensors="pt")
        inputs = inputs.to(config.DEVICE, classifier["number_type"])
        numbers = classifier["model"].get_image_features(**inputs).pooler_output.float().cpu()

        # 2. The same steps as in training: same length, then same scale
        numbers = torch.nn.functional.normalize(numbers, dim=1)
        numbers = (numbers - classifier["mean"]) / classifier["std"]

        # 3. The probe gives a score per disease. Sigmoid turns EACH score into its own
        #    number between 0 and 1. They do NOT add up to 1, so all of them can be low
        #    at the same time: that is the model saying "I do not know".
        #    The combined model asks for softmax instead, because its final layer was
        #    trained that way (one disease per X-ray).
        scores = classifier["probe"](numbers)
        if classifier.get("activation", "sigmoid") == "softmax":
            confidences = torch.softmax(scores, dim=1)[0]
        else:
            confidences = torch.sigmoid(scores)[0]

        # The old way for every probe was softmax. It forced the numbers to add up to 1,
        # so the model always had to give one of its classes almost all of the confidence,
        # even for an X-ray it had never seen anything like. That is why it answered
        # "Covid-19, 100% sure" on healthy adult X-rays from the NIH dataset.
        # confidences = torch.softmax(scores, dim=1)[0]

    result = {}
    for i, disease in enumerate(classifier["classes"]):
        result[disease] = confidences[i].item()

    # Highest confidence first
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


ONE_MODEL_ONLY = {"--combined": PROBE_COMBINED,
                  "--4class": PROBE_4_CLASSES,
                  "--2class": PROBE_2_CLASSES}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m backend.predict photo.png              all the safety checks")
        print("  python -m backend.predict photo.png --combined   the combined model only")
        print("  python -m backend.predict photo.png --4class     the 4-class probe only")
        print("  python -m backend.predict photo.png --2class     the 2-class probe only")
        sys.exit()

    photo = sys.argv[1]
    choice = sys.argv[2] if len(sys.argv) > 2 else None

    if choice in ONE_MODEL_ONLY:
        # One model, no safety checks: the raw answer, for testing by hand
        probe_path = ONE_MODEL_ONLY[choice]
        if not probe_path.exists():
            print("Missing:", probe_path)
            print("Train it first (see training/).")
            sys.exit()

        result = predict(load_classifier(probe_path), photo)
        print()
        for disease, confidence in result.items():
            bar = "#" * int(confidence * 40)
            print(f"  {disease:<14} {confidence:>7.1%}  {bar}")
        best = next(iter(result))
        print()
        print(f"Answer: {best} ({result[best]:.1%} confidence)")
        print("Research prototype: this is not a medical diagnosis.")
    else:
        # Use the same checks as the web page (only Normal / Pneumonia, limits from config.py)
        from backend.xray_check import analyze_upload, load_checker

        result, _ = analyze_upload(load_checker(), photo)
        print()
        print("Status :", result["status"])
        print("Message:", result["message"])
