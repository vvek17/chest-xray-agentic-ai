"""Frozen MedSigLIP text matching supplies evidence to frozen MedGemma.

No saved probe is loaded and no model weights are trained.
Similarity scores are cosine similarities, not probabilities or confidence.
"""

import re

import torch
from transformers import AutoModel, AutoProcessor

from backend import config
from backend.medgemma import ask_medgemma

CLASSES = ("Normal", "Pneumonia")
MODEL_NAME = "google/medsiglip-448"
TEXTS = ("a normal chest x-ray", "a chest x-ray showing pneumonia")
PROMPTS = {
    "label_hint": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. Is it Normal or Pneumonia? "
            "A separate frozen image-text model suggests {label}. "
            "That suggestion can be wrong. Make your own assessment of the image. "
            "Answer with exactly Normal or Pneumonia and nothing else."
        ),
        "max_tokens": 10,
    },
    "similarity_hint": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. Is it Normal or Pneumonia? "
            "A separate frozen image-text model gives cosine similarities: "
            "Normal={normal:.4f}, Pneumonia={pneumonia:.4f}. "
            "Higher means a closer text match, not a probability. "
            "Use these as auxiliary evidence; the image determines your answer. "
            "Answer with exactly Normal or Pneumonia and nothing else."
        ),
        "max_tokens": 10,
    },
}

_CRITERIA = (
    "Classify this chest X-ray as Normal or Pneumonia.\n\n"
    "Normal: clear lung fields, sharp costophrenic angles, normal heart "
    "size and mediastinal contour, no focal opacity. Normal findings such "
    "as ribs, vessels, and slight rotation are not pneumonia.\n"
    "Pneumonia: a specific focal or lobar consolidation, air bronchograms, "
    "or patchy or interstitial opacity, sometimes with effusion.\n\n"
)
_FINAL_INSTRUCTION = (
    "Call Pneumonia only if you can point to a specific opacity. "
    "If you cannot, answer Normal.\n\n"
    "In one short sentence, state the key finding. Then end with a final "
    "line in exactly this format:\n"
    "Final answer: Normal\n"
    "or\n"
    "Final answer: Pneumonia"
)
PROMPTS_V2 = {
    "criteria_nohint": {
        "system": "You are an expert radiologist reviewing a frontal chest X-ray.",
        "user": _CRITERIA + _FINAL_INSTRUCTION,
        "max_tokens": 100,
    },
    "criteria_hint": {
        "system": "You are an expert radiologist reviewing a frontal chest X-ray.",
        "user": _CRITERIA + (
            "A weak automated image-text matcher leaned toward {label}. It is "
            "often wrong, so use it only as a tiebreaker when the image is "
            "genuinely unclear. Base your decision on visible findings.\n\n"
        ) + _FINAL_INSTRUCTION,
        "max_tokens": 100,
    },
}

_SOFTENED_INSTRUCTION = (
    "Base your choice on all visible lung findings, not only a focal opacity. "
    "Do not call normal anatomical structures pneumonia. If findings are subtle, "
    "weigh their pattern and distribution before choosing the best-supported label.\n\n"
    "In one short sentence, state the key finding. Then end with a final "
    "line in exactly this format:\n"
    "Final answer: Normal\n"
    "or\n"
    "Final answer: Pneumonia"
)
# Added after the original criteria prompts missed most development pneumonia cases.
# These remain separate versions so the exact user-provided prompts are preserved.
PROMPTS_V2_SOFTENED = {
    name + "_softened": dict(prompt, user=prompt["user"].replace(
        _FINAL_INSTRUCTION, _SOFTENED_INSTRUCTION))
    for name, prompt in PROMPTS_V2.items()
}


def parse_label(answer):
    """Read the last explicit final answer; retain strict legacy one-word support."""
    final = re.findall(r"final answer\s*:\s*(normal|pneumonia)\b", answer, re.I)
    if final:
        return final[-1].capitalize()
    match = re.fullmatch(r"\s*(?:answer\s*:\s*)?(normal|pneumonia)[.!]?\s*", answer, re.I)
    return match.group(1).capitalize() if match else "Other"


def load_zero_shot():
    dtype = torch.float32 if config.DEVICE == "cpu" else torch.float16
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=dtype).to(config.DEVICE)
    model.eval().requires_grad_(False)
    inputs = processor(text=list(TEXTS), padding="max_length", return_tensors="pt").to(config.DEVICE)
    with torch.inference_mode():
        features = model.get_text_features(**inputs).pooler_output.float().cpu()
    return {"processor": processor, "model": model, "dtype": dtype,
            "text_features": torch.nn.functional.normalize(features, dim=1)}


def zero_shot_scores(zero_shot, cleaned_image):
    inputs = zero_shot["processor"](images=[cleaned_image], return_tensors="pt")
    inputs = inputs.to(config.DEVICE, zero_shot["dtype"])
    with torch.inference_mode():
        features = zero_shot["model"].get_image_features(**inputs).pooler_output.float().cpu()
        features = torch.nn.functional.normalize(features, dim=1)
        values = (features @ zero_shot["text_features"].T)[0]
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite MedSigLIP similarities")
    return dict(zip(CLASSES, values.tolist()))


def ensemble_answer(medgemma, cleaned_image, scores, version):
    label = max(scores, key=scores.get)
    prompt = dict((PROMPTS | PROMPTS_V2 | PROMPTS_V2_SOFTENED)[version])
    prompt["user"] = prompt["user"].format(
        label=label, normal=scores["Normal"], pneumonia=scores["Pneumonia"])
    raw = ask_medgemma(medgemma, cleaned_image, prompt)
    return raw, parse_label(raw)
