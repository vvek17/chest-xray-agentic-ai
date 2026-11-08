"""
Integration tests with the REAL models (MedSigLIP + both probes), real X-rays,
non-X-ray images, broken files, and the web page.

Run from the chest-xray-pneumonia folder (about 1-2 minutes):
    python -m pytest tests/test_xray_integration.py
"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from backend import config
from backend import predict
from backend.xray_check import ACCEPTED, NEEDS_REVIEW, REJECTED, analyze_upload, load_checker

pytestmark = pytest.mark.slow

NEEDED_FILES = [predict.PROBE_2_CLASSES, predict.PROBE_4_CLASSES, config.TEST_DIR]
if not all(Path(f).exists() for f in NEEDED_FILES):
    pytest.skip("models or dataset not found", allow_module_level=True)

UNSUPPORTED = [name for name in config.CLASS_NAMES if name not in config.SUPPORTED_LABELS]
FORBIDDEN_WORDS = [name.lower() for name in UNSUPPORTED] + ["covid", "tuberculosis", " tb "]
PHOTOS_PER_CLASS = 10


@pytest.fixture(scope="module")
def checker():
    return load_checker()


def photos_from_testing(disease):
    folder = config.TEST_DIR / disease
    return sorted(folder.iterdir())[:PHOTOS_PER_CLASS]


def assert_safe(result):
    """Rules every single result must follow."""
    assert result["status"] in (ACCEPTED, REJECTED, NEEDS_REVIEW)
    text = f" {str(result).lower()} "
    for word in FORBIDDEN_WORDS:
        assert word not in text, f"{word!r} found in {result}"
    if result["status"] == ACCEPTED:
        assert result["label"] in config.SUPPORTED_LABELS
        assert result["confidence"] >= config.CONFIDENCE_LIMIT
    else:
        assert "label" not in result


# ------------------------------------------------------------
# Real X-rays
# ------------------------------------------------------------
@pytest.mark.parametrize("disease", config.SUPPORTED_LABELS)
def test_supported_xrays(checker, disease):
    results = [analyze_upload(checker, path)[0] for path in photos_from_testing(disease)]
    for result in results:
        assert_safe(result)
    accepted = [r for r in results if r["status"] == ACCEPTED]
    correct = [r for r in accepted if r["label"] == disease]
    assert len(accepted) >= PHOTOS_PER_CLASS * 0.7, "too many good X-rays were not accepted"
    assert len(correct) == len(accepted), "an accepted X-ray got the wrong label"


@pytest.mark.parametrize("disease", UNSUPPORTED)
def test_unsupported_xrays_never_name_other_diseases(checker, disease):
    results = [analyze_upload(checker, path)[0] for path in photos_from_testing(disease)]
    for result in results:
        assert_safe(result)
    not_accepted = [r for r in results if r["status"] != ACCEPTED]
    assert len(not_accepted) >= PHOTOS_PER_CLASS * 0.8, "too many unsupported X-rays got an answer"


# ------------------------------------------------------------
# Images that are not chest X-rays
# ------------------------------------------------------------
def make_noise():
    return Image.fromarray(np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8))


def make_solid():
    return Image.new("RGB", (400, 400), (200, 30, 30))


def make_text():
    image = Image.new("RGB", (600, 300), "white")
    ImageDraw.Draw(image).text((20, 120), "Hello, this is a document, not an X-ray.", fill="black")
    return image


def make_gradient():
    row = np.linspace(0, 255, 500, dtype=np.uint8)
    return Image.fromarray(np.tile(row, (300, 1))).convert("RGB")


@pytest.mark.parametrize("make_image", [make_noise, make_solid, make_text, make_gradient])
def test_non_xray_images_are_rejected(checker, make_image):
    buffer = io.BytesIO()
    make_image().save(buffer, format="PNG")
    buffer.seek(0)
    result, _ = analyze_upload(checker, buffer)
    assert_safe(result)
    assert result["status"] == REJECTED


def test_broken_file_is_rejected(checker):
    result, image = analyze_upload(checker, io.BytesIO(b"not an image at all"))
    assert result["status"] == REJECTED and image is None


def test_uploaded_files_are_never_changed(checker, tmp_path):
    original = photos_from_testing("Normal")[0]
    copy = tmp_path / original.name
    copy.write_bytes(original.read_bytes())
    before = copy.read_bytes()
    analyze_upload(checker, copy)
    assert copy.read_bytes() == before


# ------------------------------------------------------------
# The web page
# ------------------------------------------------------------
def test_web_page_starts_and_offers_all_prompts():
    from streamlit.testing.v1 import AppTest
    from backend.zero_shot_ensemble import PROMPTS

    app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "frontend" / "app.py"), default_timeout=120).run()
    assert not app.exception
    page_text = " ".join(c.value for c in app.caption).lower()
    assert "not a medical diagnosis" in page_text
    prompt_options = app.sidebar.selectbox[0].options
    assert prompt_options == list(PROMPTS)
