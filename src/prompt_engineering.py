"""
Prompt engineering with MedGemma.

What you do:
  1. Write a prompt in src/prompts.py
  2. Choose MODE below and press Run
  3. Read the answers, improve the prompt, and run again

MODE:
  "try"      -> 10 training photos. Shows every answer, so you can improve the prompt.
  "validate" -> 20 validation photos. Shows the accuracy of the prompt.
  "test"     -> 70 testing photos. The final accuracy. Use it ONLY ONCE, at the end.
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the "src" folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import prompts
from src.medgemma_helper import ask_medgemma, get_photos, load_medgemma, save_log


# ============================================================
# SETTINGS: change these
# ============================================================
MODE = "try"                               # "try", "validate" or "test"
CLASSES = ["Normal", "Pneumonia"]          # the classes MedGemma must choose from

PHOTOS_FOR_MODE = {"try": 10, "validate": 20, "test": 70}
PART_FOR_MODE = {"try": "train", "validate": "val", "test": "test"}


# ============================================================
# Turn MedGemma's answer into one class
# ============================================================
def read_answer(answer):
    """Find which class MedGemma said. If none of our classes, return "Other"."""
    text = answer.lower()

    # If the answer has "Answer: ...", only read the part after it
    if "answer:" in text:
        text = text.split("answer:")[-1]

    # "Covid" first, so "covid-19 pneumonia" counts as Covid-19
    if "covid" in text and "Covid-19" in CLASSES:
        return "Covid-19"
    if "tuberculosis" in text and "Tuberculosis" in CLASSES:
        return "Tuberculosis"
    if "pneumonia" in text and "Pneumonia" in CLASSES:
        return "Pneumonia"
    if "normal" in text and "Normal" in CLASSES:
        return "Normal"
    return "Other"


# ============================================================
# The main program
# ============================================================
def main():
    # ---- 1. Which prompt? ----
    version = prompts.CURRENT_VERSION
    prompt = prompts.PROMPTS[version]
    print("Prompt version:", version)
    print("System prompt :", prompt["system"])
    print("User prompt   :", prompt["user"])
    print()

    # ---- 2. Which photos? ----
    part = PART_FOR_MODE[MODE]
    number_of_photos = PHOTOS_FOR_MODE[MODE]
    photos = get_photos(part, number_of_photos, CLASSES)
    print("Mode:", MODE, "|", len(photos), "photos |", "Classes:", CLASSES)

    # ---- 3. Load MedGemma ----
    medgemma = load_medgemma()

    # ---- 4. Ask MedGemma about every photo ----
    results = []
    number_correct = 0

    for number in range(len(photos)):
        photo_path, true_class = photos[number]

        answer = ask_medgemma(medgemma, photo_path, prompt)
        predicted = read_answer(answer)

        if predicted == true_class:
            number_correct = number_correct + 1
            mark = "RIGHT"
        else:
            mark = "WRONG"

        results.append({"path": str(photo_path), "true": true_class,
                        "answer": answer, "predicted": predicted})

        # In "try" mode, show every answer so you can read it
        if MODE == "try":
            print()
            print("Photo", number + 1)
            print("  True class   :", true_class)
            print("  MedGemma said:", answer.replace("\n", " / "))
            print("  Read as      :", predicted, "->", mark)

    # ---- 5. Accuracy ----
    print()
    accuracy = number_correct / len(photos) * 100
    print("Accuracy:", number_correct, "/", len(photos), "=", round(accuracy, 1), "%")

    # ---- 6. Confusion matrix: what did MedGemma say for each true class? ----
    print()
    print("What MedGemma said for each true class:")
    for true_class in CLASSES:
        counts = {}
        for name in CLASSES + ["Other"]:
            counts[name] = 0
        for result in results:
            if result["true"] == true_class:
                counts[result["predicted"]] = counts[result["predicted"]] + 1
        print(" ", true_class, "->", counts)

    # ---- 7. Save every answer ----
    print()
    save_log(results, version, CLASSES, part)


if __name__ == "__main__":
    main()
