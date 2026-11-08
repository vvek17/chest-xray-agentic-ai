"""
MedGemma: load it, ask it about one X-ray, and read its answer.

Used by: frontend/app.py and prompt_engineering/run_prompts.py
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the project folders
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from backend import config

MEDGEMMA_NAME = "google/medgemma-1.5-4b-it"


def load_medgemma(revision=None):
    """Load MedGemma (the first time it downloads about 8 GB)."""
    print(f"Loading {MEDGEMMA_NAME} ...")
    processor = AutoProcessor.from_pretrained(MEDGEMMA_NAME, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(MEDGEMMA_NAME, revision=revision, dtype=torch.bfloat16)
    model = model.to(config.DEVICE)
    model.eval()
    return {"processor": processor, "model": model}


def ask_medgemma(medgemma, photo_path, prompt, examples=None):
    """
    Show MedGemma one X-ray with the prompt. Return its answer as text.
    examples (few-shot): a list of (photo path, correct class). Each example is shown
    first as a solved question: the question, then the correct answer.
    """
    processor = medgemma["processor"]
    model = medgemma["model"]

    messages = [{"role": "system", "content": [{"type": "text", "text": prompt["system"]}]}]

    # Few-shot: solved examples first
    if examples:
        for example_path, example_class in examples:
            # Evaluation may pass an already cleaned example to match target preprocessing.
            if isinstance(example_path, Image.Image):
                example_image = example_path.convert("RGB")
            else:
                example_image = Image.open(example_path).convert("RGB")
            messages.append({"role": "user", "content": [
                {"type": "image", "image": example_image},
                # Hybrid hints belong to the target image, never to its examples.
                {"type": "text", "text": prompt.get("example_user", prompt["user"])},
            ]})
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": prompt.get("example_answer", "{class}").format(**{"class": example_class})},
            ]})

    # The real question (photo_path can also be an image that is already open)
    if isinstance(photo_path, Image.Image):
        image = photo_path.convert("RGB")
    else:
        image = Image.open(photo_path).convert("RGB")
    messages.append({"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt["user"]},
    ]})
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


def read_answer(answer, classes):
    """Find which class MedGemma said. If none of our classes, return "Other"."""
    text = answer.lower()

    # If the answer has "Answer: ...", only read the part after it
    if "answer:" in text:
        text = text.split("answer:")[-1]

    # "Covid" first, so "covid-19 pneumonia" counts as Covid-19
    if "covid" in text and "Covid-19" in classes:
        return "Covid-19"
    if "tuberculosis" in text and "Tuberculosis" in classes:
        return "Tuberculosis"
    if "pneumonia" in text and "Pneumonia" in classes:
        return "Pneumonia"
    if "normal" in text and "Normal" in classes:
        return "Normal"
    return "Other"


if __name__ == "__main__":
    print("This file is only a helper. It is not meant to be run.")
    print("Run prompt_engineering/run_prompts.py or the web page instead.")
