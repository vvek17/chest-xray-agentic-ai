"""
Prompt engineering with MedGemma.

What you do:
  1. Write a prompt in prompt_engineering/prompts.py
  2. Choose MODE below and press Run
  3. Read the answers, improve the prompt, and run again

MODE (the photo numbers are PER CLASS):
  "try"      -> 10 training photos per class. Shows every answer, so you can improve the prompt.
  "validate" -> 20 validation photos per class. Shows the accuracy of the prompt.
  "test"     -> 70 testing photos per class. The final accuracy. Use it ONLY ONCE, at the end.
"""

import sys
from pathlib import Path

# Let the VS Code Run button find the project folders
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompt_engineering import prompts
from backend.medgemma import ask_medgemma, load_medgemma, read_answer
from prompt_engineering.photos import get_examples, get_photos, save_log


# ============================================================
# SETTINGS: change these
# ============================================================
MODE = "validate"                               # "try", "validate" or "test"
CLASSES = ["Normal", "Pneumonia"]          # the classes MedGemma must choose from

# WHO answers the questions?
#   "medgemma" -> MedGemma with the prompt from prompts.py (the normal way)
#   "combined" -> our combined model (softmax probe + sigmoid probe), no prompt at all
#   "4class" / "2class" -> one of the older probes
# The photos, the split and the log format stay exactly the same, so the
# answers can be compared photo by photo.
WHO = "combined"

# Photos PER CLASS (for 2 classes: "try" = 10 Normal + 10 Pneumonia = 20 photos)
PHOTOS_PER_CLASS = {"try": 10, "validate": 20, "test": 70}
PART_FOR_MODE = {"try": "train", "validate": "val", "test": "test"}


# ============================================================
# The main program
# ============================================================
def main():
    # ---- 1. Which prompt? ----
    # A probe has no prompt: it only looks at the picture. So this part is
    # printed only when MedGemma is answering.
    version = prompts.CURRENT_VERSION
    prompt = prompts.PROMPTS[version]
    if WHO == "medgemma":
        print("Prompt version:", version)
        print("System prompt :", prompt["system"])
        print("User prompt   :", prompt["user"])
    else:
        version = WHO                       # the log file is named after the model
        print("Model:", WHO, "(no prompt: it only looks at the X-ray)")
    print()

    # ---- 2. Which photos? ----
    part = PART_FOR_MODE[MODE]
    number_of_photos = PHOTOS_PER_CLASS[MODE] * len(CLASSES)   # for example 10 x 2 = 20
    photos = get_photos(part, number_of_photos, CLASSES)
    print("Mode:", MODE, "|", len(photos), "photos |", "Classes:", CLASSES)

    # ---- 2b. Few-shot examples? (only if the prompt asks for them) ----
    examples = []
    examples_per_class = prompt.get("examples_per_class", 0)
    if examples_per_class > 0:
        examples = get_examples(CLASSES, examples_per_class)
        print("Few-shot examples:", len(examples), "solved examples are shown before each question")

    # ---- 3. Load whoever is answering ----
    medgemma = None
    classifier = None
    if WHO == "medgemma":
        medgemma = load_medgemma()
    else:
        from backend.predict import (PROBE_2_CLASSES, PROBE_4_CLASSES,
                                     PROBE_COMBINED, load_classifier, predict)
        probe_path = {"combined": PROBE_COMBINED, "4class": PROBE_4_CLASSES,
                      "2class": PROBE_2_CLASSES}[WHO]
        classifier = load_classifier(probe_path)

    # ---- 4. Ask about every photo ----
    results = []
    number_correct = 0

    for number in range(len(photos)):
        photo_path, true_class = photos[number]

        if WHO == "medgemma":
            answer = ask_medgemma(medgemma, photo_path, prompt, examples)
            predicted = read_answer(answer, CLASSES)
        else:
            # The probe gives a confidence for every class it knows. We keep only
            # the classes of this run (CLASSES) and take the highest one.
            confidences = predict(classifier, photo_path)
            allowed = {name: confidences[name] for name in CLASSES if name in confidences}
            predicted = max(allowed, key=allowed.get)
            # Write the numbers into "answer", so the log looks like MedGemma's
            answer = " ".join(f"{name}={allowed[name]:.4f}" for name in CLASSES)

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
            print(f"  {WHO} said:", answer.replace("\n", " / "))
            print("  Read as      :", predicted, "->", mark)

    # ---- 5. Accuracy ----
    print()
    accuracy = number_correct / len(photos) * 100
    print("Accuracy:", number_correct, "/", len(photos), "=", round(accuracy, 1), "%")

    # ---- 6. Confusion matrix: what did MedGemma say for each true class? ----
    print()
    print(f"What {WHO} said for each true class:")
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
