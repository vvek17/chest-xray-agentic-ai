"""Fail on image/patient overlap, label/source mismatch, missing files or changed test data.

Run: python -m training.assert_nih_splits
No model imports or inference.
"""

import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "data/external/NIH_development/manifests"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def check_splits(folder=DEFAULT):
    plan = json.loads((folder / "split_plan.json").read_text())
    metadata_path = folder.parent / "source/Data_Entry_2017_v2020.csv"
    assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == plan["protocol"]["metadata_sha256"]
    metadata = {Path(r["Image Index"]).stem: r for r in read_csv(metadata_path)}
    parts = {name: read_csv(folder / f"{name}.csv") for name in ["dev", "validation", "final_test"]}
    report = {"splits": {}, "overlaps": {}, "passed": False, "models_run": False}
    for name, rows in parts.items():
        assert rows and set(rows[0]) == {"image_id", "patient_id", "path", "true_label"}
        assert rows == plan["splits"][name], f"Changed {name} manifest"
        assert len({r["image_id"] for r in rows}) == len(rows), f"Duplicate image in {name}"
        counts = Counter(r["true_label"] for r in rows)
        assert set(counts) == {"Normal", "Pneumonia"} and counts["Normal"] == counts["Pneumonia"]
        for row in rows:
            path = Path(row["path"])
            assert path.is_file() and path.suffix.lower() == ".png" and path.stem == row["image_id"]
            official = metadata[row["image_id"]]
            assert int(official["Patient ID"]) == int(row["patient_id"])
            if name != "final_test":
                expected = "No Finding" if row["true_label"] == "Normal" else "Pneumonia"
                assert official["Finding Labels"] == expected, f"Wrong source label: {path.name}"
            else:
                assert (official["Finding Labels"] == "No Finding" if row["true_label"] == "Normal"
                        else "Pneumonia" in official["Finding Labels"].split("|"))
                assert hashlib.sha256(path.read_bytes()).hexdigest() == plan["protocol"]["protected_test_images"][row["image_id"]]
            with Image.open(path) as image:
                assert image.format == "PNG", f"Wrong image format: {path}"
                image.verify()
        report["splits"][name] = {"images": len(rows), "patients": len({int(r["patient_id"]) for r in rows}),
                                  "class_counts": dict(counts), "format": "PNG", "source": "NIH ChestX-ray14"}
    for a, b in itertools.combinations(parts, 2):
        images = {r["image_id"] for r in parts[a]} & {r["image_id"] for r in parts[b]}
        patients = {int(r["patient_id"]) for r in parts[a]} & {int(r["patient_id"]) for r in parts[b]}
        assert not images, f"Image leakage: {a}/{b}: {sorted(images)}"
        assert not patients, f"Patient leakage: {a}/{b}: {sorted(patients)}"
        report["overlaps"][f"{a}/{b}"] = {"images": len(images), "patients": len(patients)}
    protected = ROOT / "data/external/NIH/manifest.csv"
    assert hashlib.sha256(protected.read_bytes()).hexdigest() == plan["protocol"]["protected_original_manifest_sha256"]
    report["passed"] = True
    return report


if __name__ == "__main__":
    print(json.dumps(check_splits(), indent=2))
