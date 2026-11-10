"""Exercise the hybrid app with fake models; no weights or patient images needed."""

import io
from pathlib import Path

from PIL import Image
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from backend import medgemma, predict, xray_check
from frontend import evaluation
from frontend.app import PROMPT_LABELS, binary_hint, make_prompt, recognized_answer
from prompt_engineering import photos

APP = Path(__file__).resolve().parent.parent / "frontend/app.py"


# Use an in-memory neutral image so UI tests do not expose real dataset images.
def uploaded_image():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buffer, format="PNG")
    return buffer


# Make every cache independent between tests so calls can be counted reliably.
@pytest.fixture(autouse=True)
def clear_model_caches():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


# Install tiny fake models and record exactly what each pipeline passes onward.
def fake_models(monkeypatch):
    calls = {"classifier_loads": 0, "medgemma_loads": 0, "questions": [], "examples": [], "classifications": 0}

    def load_classifier(path):
        assert path == predict.PROBE_4_CLASSES
        calls["classifier_loads"] += 1
        return "probe"

    def load_medgemma():
        calls["medgemma_loads"] += 1
        return "medgemma"

    def classify(model, image):
        assert model["model"] == "probe"
        assert image.mode == "RGB"
        calls["classifications"] += 1
        return {"Tuberculosis": .05, "Normal": .2, "Pneumonia": .1, "Covid-19": .04}

    def ask(model, image, prompt, examples=None):
        assert model == "medgemma"
        assert image.mode == "RGB"
        calls["questions"].append(prompt)
        calls["examples"].append(examples or [])
        return "Pneumonia"

    monkeypatch.setattr(photos, "get_examples", lambda classes, count: [
        (Image.new("RGB", (64, 64), "black"), "Normal"),
        (Image.new("RGB", (64, 64), "white"), "Pneumonia"),
    ])
    def load_checker():
        calls["classifier_loads"] += 1
        return {"processor": None, "model": "probe", "number_type": None,
                "four": {"probe": None, "mean": None, "std": None,
                         "classes": ["Covid-19", "Normal", "Pneumonia", "Tuberculosis"]}}

    monkeypatch.setattr(xray_check, "load_checker", load_checker)
    monkeypatch.setattr(xray_check, "check_xray", lambda checker, image: {"status": "accepted", "reason": "all_checks_passed"})
    monkeypatch.setattr(predict, "predict", classify)
    monkeypatch.setattr(medgemma, "load_medgemma", load_medgemma)
    monkeypatch.setattr(medgemma, "ask_medgemma", ask)
    return calls


# The landing page must keep all three tabs accessible without loading models.
def test_empty_page_has_tabs_and_no_model_loads(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: None)
    calls = fake_models(monkeypatch)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Analyze X-ray", "Evaluation results", "About"]
    assert app.button[0].disabled
    assert calls["classifier_loads"] == 0
    assert calls["medgemma_loads"] == 0


# A scope-accepted input can still produce disagreement and require review.
def test_hybrid_upload_cache_and_changed_settings(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: uploaded_image())
    calls = fake_models(monkeypatch)
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.selectbox[0].set_value("label_only").run()
    app.button[0].click().run()
    assert not app.exception
    assert calls["classifier_loads"] == 1
    assert calls["medgemma_loads"] == 1
    assert app.session_state["analysis_result"]["hint"]["label"] == "Normal"
    assert app.session_state["analysis_result"]["hint"]["score"] == pytest.approx(2/3)
    assert app.session_state["analysis_result"]["hybrid"]["label"] == "Pneumonia"
    assert app.session_state["analysis_result"]["decision"] == "Needs review"
    assert app.session_state["analysis_result"]["top_class"] == "Normal"
    assert len(calls["questions"]) == 1
    app.run()
    assert len(calls["questions"]) == 1
    app.sidebar.toggle[0].set_value(True).run()
    assert "analysis_result" not in app.session_state
    app.button[0].click().run()
    assert calls["classifier_loads"] == 1
    assert calls["medgemma_loads"] == 1
    assert len(calls["questions"]) == 3
    assert "suggests" not in calls["questions"][-1]["user"]
    app.sidebar.selectbox[0].set_value("double_check").run()
    assert "analysis_result" not in app.session_state


# An invalid model generation is visible as Other, never silently coerced to a label.
def test_unparseable_medgemma_answer(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: uploaded_image())
    fake_models(monkeypatch)
    monkeypatch.setattr(medgemma, "ask_medgemma", lambda *a, **kw: "No pneumonia, probably normal")
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.selectbox[0].set_value("label_only").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["analysis_result"]["hybrid"]["label"] == "Other"
    assert app.session_state["analysis_result"]["decision"] == "Needs review"


# Broken uploads must not block the other tabs or load the models.
def test_unreadable_image_is_friendly(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: io.BytesIO(b"not an image"))
    calls = fake_models(monkeypatch)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.error
    assert app.button[0].disabled
    assert calls["classifier_loads"] == 0


# A failed second model keeps the four-class evidence without a misleading final card.
def test_model_failure_keeps_classifier_evidence(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: uploaded_image())
    fake_models(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(medgemma, "ask_medgemma", fail)
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.selectbox[0].set_value("label_only").run()
    app.button[0].click().run()
    assert not app.exception
    assert "error" in app.session_state["analysis_result"]
    assert "scores" in app.session_state["analysis_result"]
    assert "hybrid" not in app.session_state["analysis_result"]


# Use a synthetic fixture only in a temporary test folder, never as displayed project results.
def test_saved_metrics_and_predictions_are_read(tmp_path, monkeypatch):
    import json

    values = {"accuracy": .75, "macro_f1": .7, "mcc": .4, "invalid_outputs": 1,
              "hint_follow_rate": .5, "medsiglip_errors_fixed": 2, "correct_hints_broken": 1,
              "confusion_labels": ["Normal", "Pneumonia", "Other"],
              "confusion_matrix": [[2, 0, 0], [0, 1, 1], [0, 0, 0]]}
    (tmp_path / "val_metrics.json").write_text(json.dumps({"with_score": values}))
    (tmp_path / "val_predictions.csv").write_text("path,true,predicted\nfixture.png,Normal,Normal\n")
    (tmp_path / "report.md").write_text("Fixture report")
    monkeypatch.setattr(evaluation, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: None)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert len(app.dataframe) == 2
    assert app.dataframe[0].value.iloc[0]["Accuracy"] == .75
    assert app.dataframe[0].value.iloc[0]["Condition"] == "Hint + score"
    assert [metric.value for metric in app.metric] == ["50.0%", "2", "1"]


# Missing or damaged result files should give useful empty states rather than exceptions.
def test_missing_and_corrupt_results(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "RESULTS_DIR", tmp_path / "missing")
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: None)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any("No evaluation files yet" in item.value for item in app.info)
    monkeypatch.setattr(evaluation, "RESULTS_DIR", tmp_path)
    (tmp_path / "val_metrics.json").write_text("{broken")
    app.run()
    assert not app.exception
    assert app.warning


# Rescaling is explicit and cannot turn missing scores or a zero denominator into a guess.
def test_score_math_and_prompt_definition():
    scores = {"Covid-19": .9, "Normal": .2, "Pneumonia": .1, "Tuberculosis": .05}
    hint = binary_hint(scores)
    assert hint["score"] == pytest.approx(2/3)
    prompt = make_prompt("with_score", hint)
    assert "confidence 67%" in prompt["user"]
    assert "{label}" not in prompt["user"]
    assert "{conf" not in prompt["user"]
    scores["Normal"] = 0
    scores["Pneumonia"] = 0
    with pytest.raises(ValueError, match="zero"):
        binary_hint(scores)
    assert recognized_answer("Normal or Pneumonia") == "Other"


# Every plain label must select the intended key and send the right hint/example inputs.
@pytest.mark.parametrize("version, hint_expected, example_count", [
    ("with_score", True, 0),
    ("label_only", True, 0),
    ("double_check", True, 0),
    ("2c_v1", False, 0),
    ("2c_fs1", False, 2),
])
def test_five_named_choices(monkeypatch, version, hint_expected, example_count):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: uploaded_image())
    calls = fake_models(monkeypatch)
    app = AppTest.from_file(str(APP)).run()
    assert app.sidebar.selectbox[0].options == list(PROMPT_LABELS.values())
    assert app.sidebar.selectbox[0].value == "2c_fs1"
    app.sidebar.selectbox[0].set_value(version).run()
    app.button[0].click().run()
    assert not app.exception
    assert len(calls["questions"]) == 1
    question = calls["questions"][0]["user"]
    assert ("suggests it is Normal" in question) == hint_expected
    assert len(calls["examples"][0]) == example_count
    assert app.session_state["analysis_result"]["uses_hint"] == hint_expected
    if version == "label_only":
        assert "confidence" not in question
    if version == "with_score":
        assert "confidence 67%" in question
    if version == "double_check":
        assert "First look at the image yourself." in question


# Keep the exact supplied template unchanged when substituting per-image values.
def test_exact_user_prompt_text_and_archives():
    from prompt_engineering.prompts import PROMPTS

    expected = {
        "with_score": "Look at this chest X-ray. A separate image classifier suggests it is {label} (confidence {conf:.0%}). That classifier can be wrong, so judge the image yourself. Is it Normal or Pneumonia? Answer with only one word.",
        "label_only": "Look at this chest X-ray. A separate image classifier suggests it is {label}. That classifier can be wrong, so judge the image yourself. Is it Normal or Pneumonia? Answer with only one word.",
        "double_check": "Look at this chest X-ray. A separate image classifier suggests it is {label} (confidence {conf:.0%}). First look at the image yourself. Only disagree with the classifier if the image clearly shows otherwise. Is it Normal or Pneumonia? Answer with only one word.",
    }
    hint = {"label": "Normal", "score": .75}
    for version, text in expected.items():
        assert PROMPTS[version]["user"] == text
        assert PROMPTS[version]["system"] == "You are an expert radiologist."
        assert PROMPTS[version]["max_tokens"] == 10
        assert make_prompt(version, hint)["user"] == text.format(label="Normal", conf=.75)
        assert version + "_previous_v1" in PROMPTS


# The prompt browser exposes live settings, and About explains all three hint choices.
def test_prompt_browser_and_about_table(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: None)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any(item.label == "See the exact prompt" for item in app.expander)
    assert any(item.value == "Prompt browser" for item in app.subheader)
    assert any("max_tokens" in item.value for item in app.markdown)
    assert list(app.table[0].value["Prompt"]) == ["Hint + score", "Hint only", "Double-check"]


@pytest.mark.parametrize("status", ["rejected", "needs_review"])
def test_scope_failure_stops_binary_and_generation(monkeypatch, status):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: uploaded_image())
    calls = fake_models(monkeypatch)
    monkeypatch.setattr(xray_check, "check_xray", lambda *a: {
        "status": status, "reason": "outside_supported_labels"})
    app = AppTest.from_file(str(APP)).run()
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["analysis_result"]
    assert result["decision"] == "Outside supported scope - needs review"
    assert "hint" not in result and "scores" not in result
    assert not calls["questions"] and calls["classifications"] == 0
    assert calls["medgemma_loads"] == 0
    assert [row["status"] for row in result["trace"]] == [
        "done", "done", "done", "skipped", "skipped", "skipped", "skipped", "done"]
    assert result["tool_calls"] == 2


def test_agreement_trace_and_optional_second_run(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: uploaded_image())
    calls = fake_models(monkeypatch)
    monkeypatch.setattr(medgemma, "ask_medgemma", lambda *a, **kw: "Normal")
    app = AppTest.from_file(str(APP)).run()
    app.sidebar.selectbox[0].set_value("2c_v1").run()
    app.sidebar.toggle[0].set_value(True).run()
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["analysis_result"]
    assert result["decision"] == "Normal"
    assert all(row["status"] == "done" for row in result["trace"])
    assert result["tool_calls"] == 5
    assert result["trace"][5]["input"]["examples"] == []
    assert result["trace"][6]["input"]["examples"] == []
    assert result["trace"][5]["input"]["prompt"] == result["trace"][6]["input"]["prompt"]


def test_new_upload_clears_trace_and_result(monkeypatch):
    upload = [uploaded_image()]
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: upload[0])
    calls = fake_models(monkeypatch)
    app = AppTest.from_file(str(APP)).run()
    app.button[0].click().run()
    assert "analysis_result" in app.session_state
    changed = io.BytesIO()
    Image.new("RGB", (64, 64), "black").save(changed, format="PNG")
    upload[0] = changed
    app.run()
    assert not app.exception
    assert "analysis_result" not in app.session_state
    assert len(calls["questions"]) == 1


def test_scope_exception_never_generates(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: uploaded_image())
    calls = fake_models(monkeypatch)

    def fail(*args):
        raise RuntimeError("Scope model unavailable")

    monkeypatch.setattr(xray_check, "check_xray", fail)
    app = AppTest.from_file(str(APP)).run()
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["analysis_result"]
    assert result["trace"][2]["status"] == "failed"
    assert all(row["status"] == "skipped" for row in result["trace"][3:])
    assert "hint" not in result and not calls["questions"]
    assert result["decision"] == "Analysis failed - needs review"
