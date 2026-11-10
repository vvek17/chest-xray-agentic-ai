"""Run the five current binary prompts on five Normal and five Pneumonia images.

From chest-xray-pneumonia:
    python -m prompt_engineering.run_prompt_logs
    python -m prompt_engineering.run_prompt_logs --image-dir /path/to/images

Defaults to processed Training/Normal and Training/Pneumonia.
Files are sorted before fixed-seed selection; archived and four-class prompts
are excluded. Saved scores are independent sigmoid scores, not probabilities.
This is offline prompt exploration: scope failures are recorded but do not
suppress MedGemma calls. They always produce a final output of Needs review.
The interactive app's scope gate is unchanged. No accuracy is calculated.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from backend import config
from backend.clean_image import clean_xray
from backend.medgemma import ask_medgemma, load_medgemma
from backend.predict import predict
from backend.xray_check import ACCEPTED, check_xray, load_checker
from prompt_engineering.photos import IMAGE_TYPES, get_examples
from prompt_engineering.prompts import PROMPTS
from frontend.app import PROMPT_LABELS, binary_hint, make_prompt, recognized_answer

SCORE_KEYS = {"Normal": "normal", "Pneumonia": "pneumonia"}
PROMPT_NAMES = ("with_score", "label_only", "double_check", "2c_v1", "2c_fs1")
BINARY_CLASSES = ["Normal", "Pneumonia"]


def choose_images(folder, seed):
    """Choose once, then use this exact ordered list for every prompt."""
    chosen = []
    shuffler = random.Random(seed)
    for label in BINARY_CLASSES:
        paths = sorted((path for path in (folder / label).rglob("*")
                        if path.is_file() and path.suffix.lower() in IMAGE_TYPES),
                       key=lambda path: (path.name, str(path)))
        if len(paths) < 5:
            raise ValueError(f"Need at least 5 images in {folder / label}; found {len(paths)}.")
        chosen.extend(shuffler.sample(paths, 5))
    return sorted(chosen, key=lambda path: (path.name, str(path)))


def build_prompt(definition, scores):
    """Use the exact prompt formatting used by the Streamlit application."""
    version = next(name for name in PROMPT_NAMES if PROMPTS[name] == definition)
    return make_prompt(version, binary_hint(scores))


def prepare_examples(prompt, image):
    """Reuse the project's example selection and target-overlap protection."""
    count = prompt.get("examples_per_class", 0)
    examples, context = [], []
    if count:
        for path, label in get_examples(BINARY_CLASSES, count):
            example = clean_xray(path)
            if example.size == image.size and example.tobytes() == image.tobytes():
                raise ValueError("Target image matches a solved example; refusing label leakage.")
            examples.append((example, label))
            context.append({"image_path": str(path), "label": label,
                            "input": prompt.get("example_user", prompt["user"]),
                            "output": prompt.get("example_answer", "{class}").format(**{"class": label})})
    return examples, context


class Models:
    """Load once on first use; share the checker's frozen classifier weights."""

    def __init__(self):
        self.checker = None
        self.medgemma = None

    def classifier(self):
        if self.checker is None:
            self.checker = load_checker()
        return ({key: self.checker[key] for key in ("processor", "model", "number_type")}
                | self.checker["four"] | {"activation": "sigmoid"})

    def generator(self):
        if self.medgemma is None:
            self.medgemma = load_medgemma()
        return self.medgemma


def run_image(path, definition, models):
    """Keep partial evidence and timings when an individual stage fails."""
    started = perf_counter()
    row = {"image_path": str(path), "actual_label": next((parent.name for parent in path.parents if parent.name in BINARY_CLASSES), None), "input": None,
           "output": {"scores": None, "medgemma_raw": None},
           "time_s": {"clean": None, "classifier": None, "scope_check": None,
                      "prompt": None, "medgemma": None, "total": None},
           "final_output": "Needs review"}
    stage = "clean"

    def timed(name, action):
        nonlocal stage
        stage = name
        tick = perf_counter()
        try:
            return action()
        finally:
            row["time_s"][name] = perf_counter() - tick

    try:
        def clean():
            with Image.open(path) as original:
                return clean_xray(original)

        image = timed("clean", clean)
        scores = timed("classifier", lambda: predict(models.classifier(), image))
        for label in SCORE_KEYS:
            value = scores[label]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid sigmoid score for {label}: {value}")
        row["output"]["scores"] = {key: scores[label] for label, key in SCORE_KEYS.items()}

        # Offline exploration still records generations for rejected inputs.
        # A failed checker must never promote an answer to an accepted result.
        def scope_check():
            try:
                return check_xray(models.checker, image)
            except Exception as error:
                row["scope_error"] = f"{type(error).__name__}: {error}"
                return {"status": "error", "reason": "scope_check_failed"}

        scope = timed("scope_check", scope_check)
        row["scope_check"] = {key: scope[key] for key in ("status", "reason") if key in scope}

        def prepare():
            prompt = build_prompt(definition, scores)
            examples, context = prepare_examples(prompt, image)
            row["input"] = prompt["user"]
            if context:
                row["examples"] = context
            return prompt, examples

        prompt, examples = timed("prompt", prepare)
        raw = timed("medgemma", lambda: ask_medgemma(models.generator(), image, prompt, examples))
        row["output"]["medgemma_raw"] = raw
        # Match Streamlit: accept exactly Normal or Pneumonia, otherwise review.
        label = recognized_answer(raw)
        row["output"]["medgemma_label"] = label
        classifier_label = max(BINARY_CLASSES, key=lambda name: scores[name])
        if scope["status"] == ACCEPTED and label in BINARY_CLASSES and label == classifier_label:
            row["final_output"] = label
    except Exception as error:
        row["error"] = {"stage": stage, "message": f"{type(error).__name__}: {error}"}
    finally:
        row["time_s"]["total"] = perf_counter() - started
    return row


def simplify_log(log):
    """Keep the experiment evidence without repeated prompt/model metadata."""
    simple = {key: log[key] for key in
              ("prompt_name", "prompt_content", "seed", "total_time_s") if key in log}
    simple["results"] = []
    for result in log.get("results", []):
        row = {key: result[key] for key in
               ("image_path", "actual_label", "input", "time_s", "final_output", "error", "scope_error")
               if key in result}
        output = result.get("output", {})
        scores = output.get("scores")
        row["output"] = {
            "scores": {key: scores[key] for key in ("normal", "pneumonia") if key in scores}
                      if scores is not None else None,
            "medgemma_raw": output.get("medgemma_raw"),
        }
        if "scope_check" in result:
            row["scope_check"] = {key: result["scope_check"][key] for key in
                                  ("status", "reason") if key in result["scope_check"]}
        if result.get("examples"):
            row["examples"] = result["examples"]
        simple["results"].append(row)
    return simple


def save_log(path, log):
    """Checkpoint each finished image so interrupted runs retain their logs."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(simplify_log(log), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(image_dir, output_dir, seed=config.SEED):
    images = choose_images(image_dir.resolve(), seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = Models()
    for name in PROMPT_NAMES:
        definition = PROMPTS[name]
        started = perf_counter()
        log = {"prompt_name": name, "prompt_option": PROMPT_LABELS[name],
               "prompt_content": definition["system"] + "\n\n" + definition["user"],
               "seed": seed,
               "total_time_s": 0.0, "results": []}
        destination = output_dir / f"{name}.json"
        for index, path in enumerate(images, 1):
            row = run_image(path, definition, models)
            log["results"].append(row)
            log["total_time_s"] = perf_counter() - started
            save_log(destination, log)
            print(f"{name} [{index}/10] {path.name}: {row['final_output']}", flush=True)
        log["total_time_s"] = perf_counter() - started
        save_log(destination, log)
        print(f"Saved {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=config.TRAIN_DIR,
                        help="Folder containing Normal/ and Pneumonia/ subfolders (default: processed Training).")
    parser.add_argument("--output-dir", type=Path,
                        default=config.PROJECT_ROOT / "logs" / "prompt_runs")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    try:
        run(args.image_dir, args.output_dir, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot run prompt logs: {error}\n")


if __name__ == "__main__":
    main()
