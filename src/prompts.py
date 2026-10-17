"""
All prompt versions for MedGemma, kept in one place.

How to use:
  1. Write a new version below (v2, v3, ...). Never delete old versions:
     they are the record of your prompt engineering.
  2. Set CURRENT_VERSION to the version you want to try.
  3. Run src/prompt_engineering.py.
"""

CURRENT_VERSION = "2c_v1"

PROMPTS = {
    # ================= 2 classes: Normal vs Pneumonia =================
    # (set CLASSES_TO_USE = ["Normal", "Pneumonia"] in prompt_engineering.py)

    # ---------------- 2c_v1: the simplest 2-class prompt ----------------
    "2c_v1": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. Is it Normal or Pneumonia? "
            "Answer with only one word."
        ),
        "max_tokens": 10,
    },

    # ================= 4 classes: Covid-19, Normal, Pneumonia, Tuberculosis =================
    # (set CLASSES_TO_USE to all 4 classes in prompt_engineering.py)

    # ---------------- v1: the simplest prompt ----------------
    # Result on 10 training photos: 5/10 (50%).
    # Problem: it never said Covid-19 or Tuberculosis. Tuberculosis -> Pneumonia,
    # Covid-19 -> Normal or Pneumonia. With "one word" it does not think.
    "v1": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. "
            "Which one of these is it: Covid-19, Normal, Pneumonia, or Tuberculosis? "
            "Answer with only one word."
        ),
        "max_tokens": 20,
    },

    # ---------------- v2: describe each disease + think first ----------------
    # Changes from v1:
    #   1. Tell the model what each disease typically looks like on an X-ray.
    #   2. Let it describe what it sees first, then give the answer on the last line.
    # Result on the same 10 training photos: 5/10 (50%), but different mistakes.
    #   Better: Covid-19 2/3 (was 0/3). Its descriptions are useful to read.
    #   Worse: Pneumonia 0/2 -> called Covid-19 ("bilateral opacities in lower lungs"
    #   matched our Covid-19 description). Tuberculosis still never chosen, even for
    #   "consolidation in the right upper lobe".
    "v2": {
        "system": (
            "You are an expert thoracic radiologist. "
            "You classify chest X-rays into exactly one of four classes."
        ),
        "user": (
            "Classify this chest X-ray as exactly one of: Covid-19, Normal, Pneumonia, Tuberculosis.\n"
            "Typical signs:\n"
            "- Covid-19: hazy ground-glass opacities, usually in both lungs, often in the outer and lower parts.\n"
            "- Pneumonia: dense consolidation, often in one lobe or on one side.\n"
            "- Tuberculosis: patchy opacities, nodules or cavities, mainly in the upper lobes, sometimes scarring.\n"
            "- Normal: clear lungs with no opacities.\n"
            "First, describe the main findings in one short sentence. "
            "Then, on a new line, write 'Answer:' followed by one class name."
        ),
        "max_tokens": 80,
    },

    # ---------------- v3: step-by-step decision rules ----------------
    # Changes from v2:
    #   1. Steps in a fixed order, so the model follows one clear path.
    #   2. Upper lobes / cavities / nodules / scarring -> Tuberculosis (it was never chosen).
    #   3. Covid-19 vs Pneumonia by DENSITY (hazy vs dense), not only by location.
    #   4. The description must say the location AND the density.
    # Result on the same 10 training photos: 1/10 (10%) - much WORSE.
    #   Normal X-rays were described as "hazy opacity" -> Covid-19 (0/3 Normal).
    #   It did not follow step 2: "right upper lobe ... consolidation" -> Covid-19.
    # Lesson: long rules confuse this small model, and naming signs ("hazy")
    #   makes it "see" them even in healthy lungs. Shorter prompts work better.
    "v3": {
        "system": (
            "You are an expert thoracic radiologist. "
            "You classify chest X-rays into exactly one of four classes."
        ),
        "user": (
            "Classify this chest X-ray as exactly one of: Covid-19, Normal, Pneumonia, Tuberculosis.\n"
            "Follow these steps:\n"
            "1. Are the lungs clear, with no opacities? Then it is Normal.\n"
            "2. Is the main finding in the upper lobes, or are there cavities, nodules or scarring? "
            "Then it is Tuberculosis.\n"
            "3. Otherwise, look at how dense the opacity is:\n"
            "   - faint, hazy ground-glass opacity (you can still see the vessels through it): Covid-19\n"
            "   - dense, white consolidation (it hides the vessels): Pneumonia\n"
            "First, describe the location and the density of the main finding in one short sentence. "
            "Then, on a new line, write 'Answer:' followed by one class name."
        ),
        "max_tokens": 80,
    },
}
