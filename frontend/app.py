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
from backend.predict import PROBE_4_CLASSES, load_classifier, predict
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
def get_classifier():
    return load_classifier(PROBE_4_CLASSES)


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
        st.caption("First use loads model weights. MedGemma usually takes about 4–5 seconds per image on Apple MPS; comparison adds another pass.")
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


# Show classifier evidence, then run the selected hint-based or no-hint prompt.
def analyze(original, version, compare):
    result = {}
    started = perf_counter()
    with st.spinner("Reading the image with MedSigLIP…"):
        scores = predict(get_classifier(), original)
    check_scores(scores)
    result["scores"] = scores
    result["hint"] = binary_hint(scores)
    result["top_class"] = max(scores, key=scores.get)
    image = clean_xray(original)
    prompt = make_prompt(version, result["hint"])
    result["prompt"] = prompt
    result["version"] = version
    result["uses_hint"] = "{label}" in PROMPTS[version]["user"]
    examples = selected_examples(version, image)
    result["example_count"] = len(examples)
    # Retain the classifier output even if the second model cannot finish.
    st.session_state.analysis_result = result
    with st.spinner("MedGemma is reviewing the image…"):
        model = get_medgemma()
        result["hybrid"] = ask_for_label(model, image, prompt, examples)
    if compare and version == "2c_v1":
        result["alone"] = result["hybrid"]
    elif compare:
        with st.spinner("Running MedGemma alone for comparison…"):
            result["alone"] = ask_for_label(model, image, PROMPTS["2c_v1"])
    result["elapsed"] = perf_counter() - started
    return result


# Keep a new upload or changed settings from displaying a previous image's answer.
def reset_stale_result(content, version, compare):
    fingerprint = hashlib.sha256(content).hexdigest()
    prompt_text = json.dumps(PROMPTS[version], sort_keys=True)
    key = (fingerprint, version, prompt_text, compare)
    if st.session_state.get("analysis_key") != key:
        st.session_state.pop("analysis_result", None)
        st.session_state.analysis_key = key


# Show all four independent scores, using a distinct row and bar for the raw top class.
def show_scores(result):
    st.subheader("MedSigLIP · four-class scores")
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
    st.caption("Independent model scores, not diagnostic certainty. They need not sum to one.")
    if result["top_class"] not in CLASSES:
        st.caption(f"Note: the raw top class is {result['top_class']}. The hint below compares only Normal and Pneumonia.")


# Keep the final labels visible side by side, with an explicit agreement badge.
def show_result(result):
    if "hybrid" not in result:
        return
    hint = result["hint"]["label"]
    final = result["hybrid"]["label"]
    badge = "Agree"
    badge_style = "agree"
    if final == "Other":
        badge = "Not recognized"
        badge_style = "other"
    elif hint != final:
        badge = "Disagree"
        badge_style = "disagree"
    with st.container(border=True):
        st.markdown(f'<div class="result-heading">Final result <span class="badge {badge_style}">{badge}</span></div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        left.metric("MedSigLIP hint", hint)
        right.metric("MedGemma final answer", final)
        if final == "Other":
            st.warning("MedGemma's answer was not recognized as exactly Normal or Pneumonia.")
        elif badge == "Disagree":
            st.warning("The two models disagree. Review both outputs; this prototype cannot resolve the disagreement clinically.")
        st.caption("Selected prompt: " + prompt_label(result["version"]))
        st.caption("Any confidence wording in the prompt refers to a rescaled model score, not calibrated diagnostic certainty.")
        if not result["uses_hint"]:
            st.caption("The selected prompt received no classifier hint. Agreement here is a comparison only.")
        st.caption("Model agreement does not establish that either answer is correct.")
    if "alone" in result:
        st.metric("MedGemma alone · no hint", result["alone"]["label"])
        if result["alone"]["label"] == "Other":
            st.warning("The no-hint MedGemma answer was not recognized.")
    with st.expander("Model responses and supplied prompt"):
        st.write("**Selected prompt response**")
        st.code(result["hybrid"]["raw"], language=None)
        if "alone" in result:
            st.write("**No-hint raw response**")
            st.code(result["alone"]["raw"], language=None)
        st.write(result["prompt"]["user"])
    if "elapsed" in result:
        st.caption(f"Analysis completed in {result['elapsed']:.1f} seconds.")


# Keep the uploader and image on the left, with model evidence and final output on the right.
def show_analysis(version, compare):
    left, right = st.columns([1, 1.2], gap="large")
    original = None
    with left:
        with st.container(border=True):
            st.subheader("Your chest X-ray")
            st.caption("Choose a JPG or PNG to begin.")
            uploaded = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])
            if uploaded is None:
                st.session_state.pop("analysis_result", None)
                st.info("Your image preview will appear here.")
            else:
                try:
                    content = uploaded.getvalue()
                    with Image.open(io.BytesIO(content)) as source:
                        original = source.convert("RGB")
                    st.image(original, caption="Uploaded image", use_container_width=True)
                    st.caption("A cleaned copy is passed to the models.")
                    if version is not None:
                        reset_stale_result(content, version, compare)
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                    st.session_state.pop("analysis_result", None)
                    st.error("This file could not be opened. Please choose a readable JPG or PNG image.")
    with right:
        st.subheader("Image analysis")
        st.write("Review the classifier evidence, then compare it with MedGemma's answer.")
        run = st.button("Analyze X-ray", type="primary", use_container_width=True, disabled=original is None or version is None)
        if run:
            st.session_state.pop("analysis_result", None)
            try:
                st.session_state.analysis_result = analyze(original, version, compare)
            except Exception as error:
                result = st.session_state.get("analysis_result", {})
                result["error"] = f"{type(error).__name__}: {error}"
                st.session_state.analysis_result = result
        result = st.session_state.get("analysis_result")
        if result:
            if "scores" in result:
                show_scores(result)
                with st.container(border=True):
                    st.write("**Normal vs Pneumonia · rescaled hint scores**")
                    columns = st.columns(2)
                    for column, label in zip(columns, CLASSES):
                        column.metric(label, f"{result['hint']['scores'][label]:.1%}")
                    st.caption("Each raw score divided by their sum; these are not calibrated probabilities.")
            if "error" in result:
                st.error("Analysis could not finish. Check model access, saved weights, and available memory, then retry.")
                with st.expander("Error details"):
                    st.code(result["error"], language=None)
            show_result(result)
        else:
            st.info("Upload an X-ray and select Analyze X-ray. Results will appear here.")


# Explain what the app does and the limits users need to understand its outputs.
def show_about():
    st.subheader("Two models, one research workflow")
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
            st.write("Normal and Pneumonia scores are divided by their sum. Their higher score becomes the hint, even if another class led the four-class output.")
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
    st.set_page_config(page_title="Chest X-Ray Hybrid", page_icon="🩻", layout="wide")
    apply_style()
    st.markdown('<div class="eyebrow">CHEST X-RAY · RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Chest X-Ray Hybrid")
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
