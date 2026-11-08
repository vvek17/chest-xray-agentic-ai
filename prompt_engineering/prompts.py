"""
All prompt versions for MedGemma, kept in one place.

How to use:
  1. Write a new version below (v2, v3, ...). Never delete old versions:
     they are the record of your prompt engineering.
  2. Set CURRENT_VERSION to the version you want to try.
  3. Run prompt_engineering/run_prompts.py.
"""

CURRENT_VERSION = "2c_fs1"

PROMPTS = {
    # Hint + score: show the binary hint and rescaled score to MedGemma.
    # Exact user-supplied wording; confidence here is not calibrated certainty.
    # Result: not evaluated yet.
    "with_score": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. A separate image classifier suggests it is "
            "{label} (confidence {conf:.0%}). That classifier can be wrong, so judge "
            "the image yourself. Is it Normal or Pneumonia? Answer with only one word."
        ),
        "max_tokens": 10,
    },
    # Hint only: show the classifier label without its numerical score.
    # Exact user-supplied wording. Result: not evaluated yet.
    "label_only": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. A separate image classifier suggests it is "
            "{label}. That classifier can be wrong, so judge the image yourself. "
            "Is it Normal or Pneumonia? Answer with only one word."
        ),
        "max_tokens": 10,
    },
    # Double-check: ask for image review before overriding the scored hint.
    # Exact user-supplied wording. Result: not evaluated yet.
    "double_check": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. A separate image classifier suggests it is "
            "{label} (confidence {conf:.0%}). First look at the image yourself. "
            "Only disagree with the classifier if the image clearly shows otherwise. "
            "Is it Normal or Pneumonia? Answer with only one word."
        ),
        "max_tokens": 10,
    },
    # Archive the previous app wording unchanged so earlier versions are retained.
    # App: Hint + score. Same instructions as hybrid_confidence_v1.
    # Hybrid v1: add the four-class probe's binary hint and winning score.
    # Scores are uncalibrated; do not present them as diagnostic probabilities.
    # Result: not evaluated yet; compare on the locked hybrid manifests.
    "with_score_previous_v1": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. Is it Normal or Pneumonia? "
            "A separate four-class image classifier suggests {hint_label} "
            "when comparing Normal and Pneumonia. Its score for that label is "
            "{hint_score:.4f}. This is an uncalibrated model score, not a "
            "probability of disease. The suggestion may be wrong. "
            "Use the image to make your final decision. "
            "Answer with exactly Normal or Pneumonia and nothing else."
        ),
        "max_tokens": 10,
    },
    # App: Hint only. Same instructions as hybrid_label_v1.
    # Hybrid v2: show only the hint label, removing the numeric score.
    # Everything else, including the few-shot examples, stays fixed.
    # Result: not evaluated yet.
    "label_only_previous_v1": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. Is it Normal or Pneumonia? "
            "A separate four-class image classifier suggests {hint_label} "
            "when comparing Normal and Pneumonia. The suggestion may be wrong. "
            "Use the image to make your final decision. "
            "Answer with exactly Normal or Pneumonia and nothing else."
        ),
        "max_tokens": 10,
    },
    # App: Double-check. Review the image before considering the hint.
    # Hybrid v3: change the label-only decision rule to verify, then override.
    # Do not request an explanation; the parser requires a single label.
    # Result: not evaluated yet.
    "double_check_previous_v1": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. Is it Normal or Pneumonia? "
            "A separate four-class image classifier suggests {hint_label} "
            "when comparing Normal and Pneumonia. The suggestion may be wrong. "
            "First assess the image yourself, then double-check the suggestion. Keep the suggested "
            "label unless the image clearly supports the other label. "
            "Answer with exactly Normal or Pneumonia and nothing else."
        ),
        "max_tokens": 10,
    },
    # Legacy versions remain available for earlier evaluation manifests.
    # Hybrid v1: add the four-class probe's binary hint and winning score.
    # Scores are uncalibrated; do not present them as diagnostic probabilities.
    # Result: not evaluated yet; compare on the locked hybrid manifests.
    "hybrid_confidence_v1": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. Is it Normal or Pneumonia? "
            "A separate four-class image classifier suggests {hint_label} "
            "when comparing Normal and Pneumonia. Its score for that label is "
            "{hint_score:.4f}. This is an uncalibrated model score, not a "
            "probability of disease. The suggestion may be wrong. "
            "Use the image to make your final decision. "
            "Answer with exactly Normal or Pneumonia and nothing else."
        ),
        "max_tokens": 10,
    },
    # Hybrid v2: show only the hint label, removing the numeric score.
    # Everything else, including the few-shot examples, stays fixed.
    # Result: not evaluated yet.
    "hybrid_label_v1": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. Is it Normal or Pneumonia? "
            "A separate four-class image classifier suggests {hint_label} "
            "when comparing Normal and Pneumonia. The suggestion may be wrong. "
            "Use the image to make your final decision. "
            "Answer with exactly Normal or Pneumonia and nothing else."
        ),
        "max_tokens": 10,
    },
    # Hybrid v3: change the label-only decision rule to verify, then override.
    # Do not request an explanation; the parser requires a single label.
    # Result: not evaluated yet.
    "hybrid_verify_v1": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. Is it Normal or Pneumonia? "
            "A separate four-class image classifier suggests {hint_label} "
            "when comparing Normal and Pneumonia. The suggestion may be wrong. "
            "First verify the suggestion against the image. Keep the suggested "
            "label unless the image clearly supports the other label. "
            "Answer with exactly Normal or Pneumonia and nothing else."
        ),
        "max_tokens": 10,
    },
    # ================= 2 classes: Normal vs Pneumonia =================
    # (set CLASSES = ["Normal", "Pneumonia"] in run_prompts.py)

    # ---------------- 2c_v1: the simplest 2-class prompt ----------------
    "2c_v1": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. Is it Normal or Pneumonia? "
            "Answer with only one word."
        ),
        "max_tokens": 10,
    },
    # Result (10 + 10 training, 20 + 20 validation photos):
    #   training 17/20 (85%), validation 32/40 (80%).
    #   Mistakes: mostly Normal called Pneumonia (validation: 5 of 20 Normal).

    # ---------------- 2c_fs1: few-shot (2c_v1 + solved examples) ----------------
    # The only change from 2c_v1: before the question, MedGemma sees solved examples
    # (1 Normal + 1 Pneumonia X-ray with the correct answer). Still no training.
    "2c_fs1": {
        "system": "You are an expert radiologist.",
        "user": (
            "Look at this chest X-ray. Is it Normal or Pneumonia? "
            "Answer with only one word."
        ),
        "max_tokens": 10,
        "examples_per_class": 1,            # how many solved examples per class
        "example_answer": "{class}",        # the correct answer shown for each example
    },
    # Result (10 + 10 training, 20 + 20 validation photos):
    #   training 17/20 (85%), validation 35/40 (87.5%) - better than 2c_v1 (80%).
    #   Normal is now almost always right (validation 19/20), but more Pneumonia
    #   is called Normal (validation 4 of 20).

    # ================= 4 classes: Covid-19, Normal, Pneumonia, Tuberculosis =================
    # (set CLASSES to all 4 classes in run_prompts.py)

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
