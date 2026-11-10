"""A research interface for MedSigLIP hints followed by frozen MedGemma.

How to run, from chest-xray-pneumonia:
    streamlit run frontend/app.py --server.port 8502

No inference function needs replacing. To choose a default, set CURRENT_VERSION
in prompt_engineering/prompts.py to one of the five keys in PROMPT_LABELS. The evaluation folder is
RESULTS_DIR in frontend/evaluation.py. Styling is in frontend/style.css.
"""

import hashlib
import html
import io
import json
import math
import sys
from pathlib import Path
from time import perf_counter

# Add the project root so this file works with the existing Streamlit run command.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image, UnidentifiedImageError

from backend import config
from backend.clean_image import clean_xray
from backend.medgemma import ask_medgemma, load_medgemma, read_answer
from backend.predict import predict
from backend.xray_check import ACCEPTED, check_xray, load_checker
from frontend.evaluation import show_evaluation
from prompt_engineering.prompts import CURRENT_VERSION, PROMPTS
from prompt_engineering.photos import get_examples

# Change these plain labels here; the prompt keys stay linked to prompts.py.
PROMPT_LABELS = {
    "with_score": "Hint + score",
    "label_only": "Hint only",
    "double_check": "Double-check",
    "2c_v1": "No hint",
    "2c_fs1": "No hint + 2 examples",
}
CLASSES = ["Normal", "Pneumonia"]
FALLBACK_VERSION = "label_only"


# Keep each model in memory so clicks and sidebar reruns do not reload its weights.
@st.cache_resource
def get_checker():
    return load_checker()


def get_classifier():
    # Reuse the same frozen backbone and probe already loaded by the checker.
    checker = get_checker()
    return {key: checker[key] for key in ("processor", "model", "number_type")} | checker["four"] | {"activation": "sigmoid"}


# Cache MedGemma separately so the no-upload page never loads either large model.
@st.cache_resource
def get_medgemma():
    return load_medgemma()


# Load local styles; theme variables follow Streamlit's light/dark setting.
def apply_style():
    path = config.PROJECT_ROOT / "frontend" / "style.css"
    st.markdown("<style>" + path.read_text() + "</style>", unsafe_allow_html=True)


# Keep every displayed label in the single dictionary above.
def prompt_label(version):
    return PROMPT_LABELS[version]


# Only offer the five named choices when their definitions exist in prompts.py.
def available_versions():
    versions = []
    for name in PROMPT_LABELS:
        if name in PROMPTS:
            versions.append(name)
    return versions


# Respect CURRENT_VERSION and show plain labels instead of internal version keys.
def show_sidebar():
    with st.sidebar:
        st.markdown('<div class="eyebrow">ANALYSIS SETTINGS</div>', unsafe_allow_html=True)
        st.subheader("Prompt browser")
        versions = available_versions()
        if not versions:
            st.error("Add the named prompts to PROMPTS before running analysis.")
            return None, False
        default = FALLBACK_VERSION
        if CURRENT_VERSION in versions:
            default = CURRENT_VERSION
        elif default not in versions:
            default = versions[0]
        version = st.selectbox("Prompt", versions, index=versions.index(default), format_func=prompt_label)
        if CURRENT_VERSION not in versions:
            st.caption("The configured default is unavailable. Using " + prompt_label(default) + ".")
        if version == "with_score":
            st.caption("MedGemma sees the classifier hint and rescaled model score. Scores are not calibrated certainty.")
        elif version == "label_only":
            st.caption("MedGemma sees the classifier hint without a score.")
        elif version == "double_check":
            st.caption("MedGemma assesses the image first and overrides the hint only when the image clearly disagrees.")
        elif version == "2c_v1":
            st.caption("MedGemma sees the image without a hint or examples.")
        elif version == "2c_fs1":
            st.caption("MedGemma sees the image and two solved examples: one Normal and one Pneumonia. No classifier hint is sent.")
        compare = st.toggle("Also show MedGemma alone", value=False)
        st.caption("The comparison uses the same image without a hint or examples.")
        with st.expander("See the exact prompt"):
            st.write("**System**")
            st.write(PROMPTS[version]["system"])
            st.write("**Image question**")
            st.write(PROMPTS[version]["user"])
            st.write("**max_tokens**")
            st.write(PROMPTS[version]["max_tokens"])
        st.divider()
        st.caption("First use loads model weights. The optional comparison adds a separate generation call.")
        st.caption("No hybrid winner has been established yet. Select the prompt you want to explore.")
    return version, compare


# Validate all four scores before showing bars or dividing the two binary scores.
def check_scores(scores):
    for label in config.CLASS_NAMES:
        score = scores.get(label)
        if not isinstance(score, (int, float)):
            raise ValueError("Missing model score: " + label)
        if not math.isfinite(score) or score < 0 or score > 1:
            raise ValueError("Invalid model score: " + label)


# Rescale only Normal and Pneumonia; this does not turn scores into disease probabilities.
def binary_hint(scores):
    check_scores(scores)
    total = scores["Normal"] + scores["Pneumonia"]
    if total <= 0:
        raise ValueError("Both binary scores are zero, so a hint cannot be rescaled.")
    normalized = {}
    for label in CLASSES:
        normalized[label] = scores[label] / total
    label = max(normalized, key=normalized.get)
    return {"label": label, "score": normalized[label], "scores": normalized}


# Insert the current image's hint without changing the stored prompt dictionary.
def make_prompt(version, hint):
    prompt = dict(PROMPTS[version])
    # Fill only the two placeholders requested; leave the prompt wording intact.
    prompt["user"] = prompt["user"].format(label=hint["label"], conf=hint["score"])
    return prompt


# Require a single exact label, then reuse the project parser for the recognized answer.
def recognized_answer(answer):
    if answer.strip() not in CLASSES:
        return "Other"
    return read_answer(answer.strip(), CLASSES)


# Preserve the raw answer so an invalid generation can be inspected instead of hidden.
def ask_for_label(model, image, prompt, examples=None):
    answer = ask_medgemma(model, image, prompt, examples)
    return {"raw": answer, "label": recognized_answer(answer)}


# Load exactly two labeled examples only for the named few-shot option.
def selected_examples(version, image):
    if version != "2c_fs1":
        return []
    examples = []
    for path, label in get_examples(CLASSES, 1):
        example_image = clean_xray(path)
        # Do not show the target image with its solved label as a context example.
        if example_image.size == image.size and example_image.tobytes() == image.tobytes():
            raise ValueError("This upload matches a solved example. Choose a different image or a prompt without examples.")
        examples.append((example_image, label))
    return examples


# The controller only orders existing tools and applies the requested stop rules.
PLAN = [
    ("Receive image", "PIL.Image"),
    ("Clean image", "backend.clean_image.clean_xray"),
    ("Scope check", "backend.xray_check.check_xray"),
    ("Classifier", "backend.predict.predict"),
    ("Two-class comparison", "binary_hint"),
    ("MedGemma", "backend.medgemma.ask_medgemma"),
    ("MedGemma alone", "backend.medgemma.ask_medgemma"),
    ("Final decision", "controller"),
]


def image_record(image):
    return {"mode": image.mode, "size": list(image.size),
            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()}


def analyze(original, version, compare, render=None):
    started = perf_counter()
    result = {"version": version, "trace": [
        {"step": name, "tool": tool, "status": "pending", "seconds": None,
         "input": None, "output": None} for name, tool in PLAN], "tool_calls": 0}
    st.session_state.analysis_result = result

    def refresh():
        result["elapsed"] = perf_counter() - started
        if render:
            render(result)

    def step(index, inputs, action, tool_call=True):
        row = result["trace"][index]
        row.update(status="running", input=inputs)
        if tool_call:
            result["tool_calls"] += 1
        refresh()
        tick = perf_counter()
        try:
            value, raw = action()
            row.update(status="done", output=raw)
            return value
        except Exception as error:
            row.update(status="failed", output={"error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            row["seconds"] = perf_counter() - tick
            refresh()

    def skip(index, reason):
        result["trace"][index].update(status="skipped", output={"reason": reason})

    try:
        step(0, image_record(original), lambda: (original, image_record(original)), False)
        image = step(1, image_record(original), lambda: clean_step(original))
        result["cleaned"] = image
        scope = step(2, image_record(original), lambda: scope_step(original))
        result["scope"] = scope
        if scope["status"] != ACCEPTED:
            for index in range(3, 7):
                skip(index, "Scope checks did not pass; binary analysis stopped.")
            decision = "Outside supported scope - needs review"
        else:
            scores = step(3, image_record(original), lambda: classifier_step(original))
            check_scores(scores)
            result.update(scores=scores, top_class=max(scores, key=scores.get))
            hint = step(4, scores, lambda: comparison_step(scores), False)
            result["hint"] = hint
            prompt = make_prompt(version, hint)
            result.update(prompt=prompt, uses_hint="{label}" in PROMPTS[version]["user"])
            examples = selected_examples(version, image)
            result["example_count"] = len(examples)
            inputs = {"image": image_record(image), "prompt": prompt,
                      "examples": [{"image": image_record(photo), "label": label,
                                    "user": prompt.get("example_user", prompt["user"]),
                                    "answer": prompt.get("example_answer", "{class}").format(**{"class": label})}
                                   for photo, label in examples]}
            def ask(prompt, examples=None):
                answer = ask_for_label(get_medgemma(), image, prompt, examples)
                return answer, answer
            result["hybrid"] = step(5, inputs, lambda: ask(prompt, examples))
            if compare:
                result["alone"] = step(6, {"image": image_record(image), "prompt": PROMPTS["2c_v1"], "examples": []},
                                       lambda: ask(PROMPTS["2c_v1"]))
            else:
                skip(6, "Optional comparison is turned off.")
            final = result["hybrid"]["label"]
            result["agreement"] = "Agree" if final == hint["label"] else "Disagree"
            decision = final if final in CLASSES and final == hint["label"] else "Needs review"
        result["decision"] = step(7, {"scope": scope, "comparison": result.get("hint"),
                                     "medgemma": result.get("hybrid")},
                                  lambda: (decision, {"decision": decision}), False)
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", decision="Analysis failed - needs review")
        for row in result["trace"]:
            if row["status"] == "pending":
                row.update(status="skipped", output={"reason": "An earlier step failed."})
    refresh()
    return result


def clean_step(original):
    image = clean_xray(original)
    return image, image_record(image)


def scope_step(original):
    value = check_xray(get_checker(), original)
    return value, value


def classifier_step(original):
    scores = predict(get_classifier(), original)
    return scores, scores


def comparison_step(scores):
    hint = binary_hint(scores)
    return hint, hint


def show_trace(result):
    st.subheader("Agent plan and execution")
    st.caption("Deterministic controller: follows this plan and stops when scope checks fail. No autonomous reasoning is claimed.")
    st.write(f"Elapsed: {result['elapsed']:.1f} seconds | Tool calls: {result['tool_calls']}")
    st.caption("Tool calls count top-level cleaning, checking, classification, and generation calls, including failed attempts. Cached model loading and controller operations are excluded. Elapsed time updates at step transitions.")
    for row in result["trace"]:
        duration = "--" if row["seconds"] is None else f"{row['seconds']:.2f}s"
        with st.expander(f"{row['step']} | {row['status']} | {row['tool']} | {duration}"):
            st.write("Raw input (images represented by size, mode, and pixel hash)")
            st.json(row["input"])
            st.write("Raw output")
            st.json(row["output"])


# Keep a new upload or changed settings from displaying a previous image's answer.
def reset_stale_result(content, version, compare):
    fingerprint = hashlib.sha256(content).hexdigest()
    prompt_text = json.dumps(PROMPTS[version], sort_keys=True)
    key = ("scope-controller-v1", fingerprint, version, prompt_text, compare,
           json.dumps(PROMPTS["2c_v1"], sort_keys=True),
           config.XRAY_SIMILARITY_LIMIT, config.OUT_OF_SCOPE_LIMIT, config.CONFIDENCE_LIMIT)
    if st.session_state.get("analysis_key") != key:
        st.session_state.pop("analysis_result", None)
        st.session_state.analysis_key = key


# Show all four independent scores, using a distinct row and bar for the raw top class.
def show_scores(result):
    st.subheader("MedSigLIP - four-class sigmoid scores")
    scores = result["scores"]
    for label in config.CLASS_NAMES:
        top = label == result["top_class"]
        row_class = "score-row top" if top else "score-row"
        tag = '<span class="top-tag">TOP</span>' if top else ""
        score = scores[label]
        name = html.escape(label)
        st.markdown(
            f'<div class="{row_class}"><div class="score-label">'
            f'<span class="score-name">{name}{tag}</span>'
            f'<span class="score-value">{score:.4f}</span></div>'
            f'<div class="score-track"><div class="score-fill" style="width:{score * 100:.2f}%"></div></div></div>',
            unsafe_allow_html=True,
        )
    st.caption("Independent sigmoid scores, not diagnostic certainty. They need not sum to one.")
    if result["top_class"] not in CLASSES:
        st.caption(f"Note: the raw top class is {result['top_class']}. The hint below compares only Normal and Pneumonia.")


def show_result(result):
    with st.container(border=True):
        st.subheader("Final decision")
        st.header(result.get("decision", "Running analysis"))
        st.caption("Model output, not a verified diagnosis.")
        if "agreement" in result:
            st.write("Agreement status: " + result["agreement"])
    if "scores" in result:
        show_scores(result)
    if "hint" in result:
        st.write("**Normal vs Pneumonia - rescaled two-class comparison**")
        for column, label in zip(st.columns(2), CLASSES):
            column.metric(label, f"{result['hint']['scores'][label]:.1%}")
        st.caption("Raw score divided by the sum of the two scores. Not calibrated probabilities.")
    if "hybrid" in result:
        st.write("**MedGemma raw answer**")
        st.code(result["hybrid"]["raw"], language=None)
        st.caption("Selected prompt: " + prompt_label(result["version"]))
        if not result["uses_hint"]:
            st.caption("No classifier prediction was sent to this prompt.")
        with st.expander("Full supplied prompt and example context"):
            st.json(result["trace"][5]["input"])
    if "alone" in result:
        st.write("**MedGemma alone - no hint or examples**")
        st.code(result["alone"]["raw"], language=None)
    if "error" in result:
        st.write("Analysis could not finish. Check model access, saved files, and available memory.")
        with st.expander("Error details"):
            st.code(result["error"], language=None)


def show_analysis(version, compare):
    uploaded = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])
    original = None
    if uploaded is None:
        st.session_state.pop("analysis_result", None)
    else:
        try:
            content = uploaded.getvalue()
            with Image.open(io.BytesIO(content)) as source:
                original = source.convert("RGB")
            if version is not None:
                reset_stale_result(content, version, compare)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            st.session_state.pop("analysis_result", None)
            st.error("This file could not be opened. Please choose a readable JPG or PNG image.")
    if version is None:
        st.session_state.pop("analysis_result", None)
    run = st.button("Run analysis", type="primary", disabled=original is None or version is None)
    final_slot = st.empty()
    images_slot = st.empty()
    timeline_slot = st.empty()
    if original is not None:
        with images_slot.container():
            st.image(original, caption="Original upload", width=300)
    if run:
        def render(result):
            with final_slot.container():
                st.subheader(result.get("decision", "Running analysis"))
            if "cleaned" in result:
                with images_slot.container():
                    left, right = st.columns(2)
                    left.image(original, caption="Original upload", use_container_width=True)
                    right.image(result["cleaned"], caption="Cleaned image", use_container_width=True)
            with timeline_slot.container():
                show_trace(result)
        analyze(original, version, compare, render)
    result = st.session_state.get("analysis_result")
    if result:
        with final_slot.container():
            show_result(result)
        if "cleaned" in result:
            with images_slot.container():
                left, right = st.columns(2)
                left.image(original, caption="Original upload", use_container_width=True)
                right.image(result["cleaned"], caption="Cleaned image", use_container_width=True)
        with timeline_slot.container():
            show_trace(result)
        summary = "\n".join(f"{row['step']}: {row['status']} | {row['tool']} | "
                             + (f"{row['seconds']:.2f}s" if row['seconds'] is not None else "--")
                             + (" | " + row['output']['reason'] if isinstance(row['output'], dict) and 'reason' in row['output'] else "")
                             for row in result["trace"])
        st.write("**Copyable agent trace**")
        st.code(summary + f"\nDecision: {result['decision']}\nTool calls: {result['tool_calls']}\nElapsed: {result['elapsed']:.2f}s", language=None)
    elif original is None:
        st.info("Upload a chest X-ray to begin. No models are loaded until you run analysis.")
    else:
        st.write("**Agent plan**")
        for name, tool in PLAN:
            st.write(f"{name} | pending | {tool}")


# Explain what the app does and the limits users need to understand its outputs.
def show_about():
    st.subheader("Two models, one controlled research workflow")
    st.write("A thin controller runs image cleaning, the existing scope checker, four-class classification, binary comparison, and MedGemma in order. Failed scope checks stop the run; disagreement needs review. Existing model logic, thresholds, and prompt text are unchanged.")
    # Use the shared labels so the table follows future dropdown renaming.
    prompt_rows = [
        {"Prompt": prompt_label("with_score"), "What it does": "Shows MedSigLIP's answer and rescaled model score."},
        {"Prompt": prompt_label("label_only"), "What it does": "Shows MedSigLIP's answer without a score."},
        {"Prompt": prompt_label("double_check"), "What it does": "Asks MedGemma to review the image first and disagree only when it clearly sees otherwise."},
    ]
    st.table(prompt_rows)
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**1 · MedSigLIP reads the image**")
            st.write("A frozen MedSigLIP backbone and trained four-class probe produce scores for COVID-19, Normal, Pneumonia, and Tuberculosis.")
            st.markdown("**2 · A binary hint is formed**")
            st.write("Normal and Pneumonia scores are divided by their sum. Their higher score becomes the comparison label only after the existing scope checks pass.")
            st.markdown("**3 · MedGemma gives the final label**")
            st.write("Frozen MedGemma receives the cleaned image and selected hint prompt. Choose a hint-based prompt or a no-hint prompt. The few-shot option supplies one solved Normal and one solved Pneumonia image. The optional comparison always omits hints and examples.")
    with right:
        with st.container(border=True):
            st.markdown("**What these outputs mean**")
            st.write("Research only. This system has not undergone clinical validation and must not guide patient care.")
            st.write("Patient independence across data splits is unverified. Dataset source differences and pretraining overlap may affect results.")
            st.write("MedGemma generates labels, not calibrated probabilities. Neither score normalization nor model agreement establishes diagnostic certainty.")
            st.write("The final decision is limited to Normal or Pneumonia. COVID-19 and Tuberculosis remain visible only in the classifier evidence.")
            st.write("Saved evaluation conditions can differ from this app's settings. Read their report before attributing performance to this interface.")


# Render all tabs even when no image is uploaded; evaluation never loads a model.
def main():
    st.set_page_config(page_title="Chest X-Ray Hybrid", layout="wide")
    apply_style()
    st.markdown('<div class="eyebrow">CHEST X-RAY · RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Chest X-Ray Agent Workspace")
    st.write("Classifier evidence. Image review. A transparent view of both model outputs.")
    st.markdown('<div class="research-banner"><strong>Research prototype. Not for clinical use.</strong></div>', unsafe_allow_html=True)
    version, compare = show_sidebar()
    analysis, evaluation, about = st.tabs(["Analyze X-ray", "Evaluation results", "About"])
    with analysis:
        show_analysis(version, compare)
    with evaluation:
        show_evaluation(PROMPT_LABELS)
    with about:
        show_about()


if __name__ == "__main__":
    main()
