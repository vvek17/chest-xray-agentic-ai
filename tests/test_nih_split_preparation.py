"""Data-only checks; no model or final-test inference."""

import io
import zipfile

from PIL import Image
import pytest

from training import prepare_nih_splits as prep


def test_balanced_sampling_excludes_patients_and_is_deterministic():
    metadata = []
    for patient in range(1, 13):
        metadata.append({"Patient ID": str(patient), "Image Index": f"{patient:08}_000.png",
                         "Finding Labels": "Pneumonia" if patient < 7 else "No Finding"})
    metadata.append({"Patient ID": "2", "Image Index": "00000002_001.png", "Finding Labels": "No Finding"})
    metadata.append({"Patient ID": "99", "Image Index": "00000099_000.png", "Finding Labels": "Effusion|Pneumonia"})
    protected = [{"image_id": "00000001_000", "patient_id": "1", "class": "Pneumonia"}]
    parts, availability = prep.choose_splits(metadata, protected)
    assert parts == prep.choose_splits(metadata, protected)[0]
    assert availability["Pneumonia"]["patients"] == 5
    patients = []
    for split in ["dev", "validation"]:
        assert len(parts[split]) == 4  # largest balanced size: two of each class
        assert sum(r["true_label"] == "Normal" for r in parts[split]) == 2
        patients.append({r["patient_id"] for r in parts[split]})
    assert not patients[0] & patients[1]
    assert "1" not in patients[0] | patients[1]
    assert "99" not in patients[0] | patients[1]


def test_range_zip_member_preserves_original_png(monkeypatch, tmp_path):
    image = io.BytesIO()
    Image.new("L", (16, 16), 60).save(image, format="PNG")
    original = image.getvalue()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("images/00000001_000.png", original)
    contents = archive.getvalue()
    monkeypatch.setattr(prep, "get_bytes", lambda url, start, end: contents[start:end + 1])
    with zipfile.ZipFile(prep.RemoteZip("test", len(contents))) as zipped:
        info = zipped.infolist()[0]
    entry = {"url": "test", "archive": "fixture.zip", "archive_size": len(contents),
             "offset": info.header_offset, "compressed_size": info.compress_size,
             "size": info.file_size, "crc": info.CRC, "compression": info.compress_type,
             "filename": info.filename}
    row = {"path": str(tmp_path / "sample.png"), "image_id": "00000001_000"}
    prep.download_image(row, entry)
    assert (tmp_path / "sample.png").read_bytes() == original


def test_refuses_server_ignoring_range(monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {}

    monkeypatch.setattr(prep.urllib.request, "urlopen", lambda *a, **kw: Response(b"not a ranged response"))
    monkeypatch.setattr(prep.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="refusing full archive"):
        prep.get_bytes("https://example.invalid/archive.zip", 0, 10)
