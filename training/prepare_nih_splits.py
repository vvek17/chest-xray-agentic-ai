"""Step 0 only: acquire original NIH PNGs and prepare patient-disjoint manifests.

No model imports or inference. Run: python -m training.prepare_nih_splits
Downloads selected ZIP members using HTTP ranges; never downloads full archives.
"""

import binascii
import concurrent.futures
import csv
import hashlib
import io
import json
import random
import struct
import time
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/external/NIH_development"
SOURCE = DATA / "source"
FINAL = ROOT / "data/external/NIH/manifest.csv"
REPO = "alkzar90/NIH-Chest-X-ray-dataset"
SEED = 42
PER_CLASS = 100
FIELDS = ["image_id", "patient_id", "path", "true_label"]


def get_bytes(url, start=None, end=None):
    headers = {"User-Agent": "NIH-Step0-patient-disjoint-split-preparation"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
        url += ("&" if "?" in url else "?") + f"range={start}-{end}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                if start is not None:
                    expected = f"bytes {start}-{end}/"
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                        raise RuntimeError("Server did not honor the exact byte range; refusing full archive download")
                    data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise RuntimeError("Incorrect range response length")
                    return data
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        if end - self.position > 16 * 1024 * 1024:
            raise RuntimeError("Unexpected large ZIP index read")
        data = get_bytes(self.url, self.position, end - 1)
        self.position = end
        return data


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_splits(metadata, protected):
    """One image per patient, with deterministic balanced sampling before any inference."""
    excluded = {int(row["patient_id"]) for row in protected}
    groups = {"Normal": defaultdict(list), "Pneumonia": defaultdict(list)}
    labels = {"No Finding": "Normal", "Pneumonia": "Pneumonia"}
    for row in metadata:
        label = labels.get(row["Finding Labels"])
        patient = int(row["Patient ID"])
        if label and patient not in excluded:
            groups[label][patient].append(row)
    # Exclude every eligible pneumonia patient's normal follow-ups from normal sampling.
    # This guarantees patients cannot cross labels or splits among selected images.
    for patient in groups["Pneumonia"]:
        groups["Normal"].pop(patient, None)
    largest = min(PER_CLASS, len(groups["Normal"]) // 2, len(groups["Pneumonia"]) // 2)
    if largest == 0:
        raise RuntimeError("No balanced patient-disjoint dev and validation split is possible")
    rng = random.Random(SEED)
    parts = {"dev": [], "validation": []}
    availability = {}
    for label in ["Normal", "Pneumonia"]:
        patients = sorted(groups[label])
        availability[label] = {"patients": len(patients), "images": sum(map(len, groups[label].values()))}
        rng.shuffle(patients)
        for index, patient in enumerate(patients[:2 * largest]):
            choices = sorted(groups[label][patient], key=lambda row: row["Image Index"])
            row = rng.choice(choices)
            part = "dev" if index < largest else "validation"
            parts[part].append({"image_id": Path(row["Image Index"]).stem,
                                "patient_id": str(patient),
                                "path": str((DATA / "images" / row["Image Index"]).resolve()),
                                "true_label": label})
    for rows in parts.values():
        rng.shuffle(rows)
    parts["final_test"] = [{"image_id": row["image_id"], "patient_id": str(int(row["patient_id"])),
                            "path": str((FINAL.parent / row["class"] / (row["image_id"] + ".png")).resolve()),
                            "true_label": row["class"]} for row in protected]
    return parts, availability


def archive_index(archive, revision):
    cache = SOURCE / "archive_indexes" / (Path(archive["path"]).stem + ".json")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{archive['path']}?download=true"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["revision"] == revision:
            return saved["entries"]
    entries = {}
    with zipfile.ZipFile(RemoteZip(url, archive["size"])) as zipped:
        for info in zipped.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".png"):
                continue
            name = Path(info.filename).name
            if name in entries:
                raise RuntimeError("Duplicate image filename in archive")
            entries[name] = {"url": url, "archive": archive["path"], "archive_size": archive["size"],
                             "offset": info.header_offset, "compressed_size": info.compress_size,
                             "size": info.file_size, "crc": info.CRC, "compression": info.compress_type,
                             "filename": info.filename}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"revision": revision, "entries": entries}))
    print("Indexed", archive["path"], len(entries), "PNGs", flush=True)
    return entries


def download_image(row, entry):
    path = Path(row["path"])
    if path.exists():
        data = path.read_bytes()
    else:
        start = entry["offset"]
        end = min(entry["archive_size"] - 1, start + entry["compressed_size"] +
                  30 + len(entry["filename"].encode("utf-8")) + 4096 - 1)
        chunk = get_bytes(entry["url"], start, end)
        header = struct.unpack("<4s5H3I2H", chunk[:30])
        if header[0] != b"PK\x03\x04" or header[2] & 1 or header[3] != entry["compression"]:
            raise RuntimeError("Unexpected or encrypted ZIP member")
        payload_offset = 30 + header[-2] + header[-1]
        length = entry["compressed_size"]
        if payload_offset + length > len(chunk):
            payload = get_bytes(entry["url"], start + payload_offset, start + payload_offset + length - 1)
        else:
            payload = chunk[payload_offset:payload_offset + length]
        if entry["compression"] == zipfile.ZIP_STORED:
            data = payload
        elif entry["compression"] == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(payload, -15)
        else:
            raise RuntimeError("Unsupported ZIP compression")
    if len(data) != entry["size"] or binascii.crc32(data) & 0xffffffff != entry["crc"]:
        raise RuntimeError(f"ZIP integrity check failed: {path.name}")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG":
            raise RuntimeError("Source image is not PNG")
        image.verify()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {"image_id": row["image_id"], "sha256": hashlib.sha256(data).hexdigest(),
            "archive": entry["archive"], "size": len(data), "zip_crc32": entry["crc"]}


def main():
    SOURCE.mkdir(parents=True, exist_ok=True)
    pin_path = SOURCE / "mirror_revision.json"
    if pin_path.exists():
        pin = json.loads(pin_path.read_text())
    else:
        api = json.loads(get_bytes(f"https://huggingface.co/api/datasets/{REPO}"))
        pin = {"repo": REPO, "revision": api["sha"]}
        pin_path.write_text(json.dumps(pin, indent=2))
    revision = pin["revision"]
    metadata_path = SOURCE / "Data_Entry_2017_v2020.csv"
    pinned_metadata = get_bytes(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/data/Data_Entry_2017_v2020.csv")
    if metadata_path.exists() and metadata_path.read_bytes() != pinned_metadata:
        raise RuntimeError("Local metadata differs from pinned revision")
    if not metadata_path.exists():
        metadata_path.write_bytes(pinned_metadata)
    metadata, protected = read_csv(metadata_path), read_csv(FINAL)
    official = {Path(row["Image Index"]).stem: row for row in metadata}
    for row in protected:
        assert int(official[row["image_id"]]["Patient ID"]) == int(row["patient_id"])
    parts, availability = choose_splits(metadata, protected)
    manifest_dir = DATA / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "step": 0, "seed": SEED, "target_per_class_per_split": PER_CLASS,
        "eligibility": {"Normal": "Finding Labels exactly No Finding", "Pneumonia": "Finding Labels exactly Pneumonia"},
        "availability_after_patient_exclusion": availability,
        "one_image_per_patient": True, "source_revision": pin,
        "metadata_sha256": digest(metadata_path), "protected_original_manifest_sha256": digest(FINAL),
        "protected_test_images": {row["image_id"]: digest(Path(row["path"])) for row in parts["final_test"]},
        "source": "NIH Clinical Center; original PNGs distributed through a pinned public mirror",
        "nih_download": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "citation": "Wang et al., ChestX-ray8, IEEE CVPR 2017, pp. 3462-3471; doi:10.1109/CVPR.2017.369",
        "limitations": ["The fixed final set contains 85 co-labeled pneumonia images; dev/validation use Pneumonia-only images.",
                        "The final set was evaluated in earlier project experiments; this protocol cannot erase that exposure.",
                        "Step 0 performs no model inference and does not change prompts or thresholds."],
    }
    plan_path = manifest_dir / "split_plan.json"
    plan = {"protocol": protocol, "splits": parts}
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("Existing split plan differs; refusing to resample")
    else:
        plan_path.write_text(json.dumps(plan, indent=2))
    archives = json.loads(get_bytes(f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/data/images"))
    archives = [a for a in archives if a["path"].endswith(".zip")]
    entries = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(archive_index, a, revision) for a in archives]
        for future in concurrent.futures.as_completed(futures):
            batch = future.result()
            if entries.keys() & batch.keys():
                raise RuntimeError("Duplicate image IDs between ZIP archives")
            entries.update(batch)
    selected = parts["dev"] + parts["validation"]
    assert len({r["image_id"] for r in selected}) == len(selected)
    missing = [r["image_id"] for r in selected if r["image_id"] + ".png" not in entries]
    if missing:
        raise RuntimeError(f"Selected images missing from mirror: {missing}")
    provenance = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download_image, row, entries[row["image_id"] + ".png"]) for row in selected]
        for future in concurrent.futures.as_completed(futures):
            provenance.append(future.result())
            if len(provenance) % 25 == 0 or len(provenance) == len(selected):
                print("Verified original PNGs:", len(provenance), "/", len(selected), flush=True)
    (SOURCE / "selected_image_provenance.json").write_text(json.dumps(sorted(provenance, key=lambda row: row["image_id"]), indent=2))
    for name, rows in parts.items():
        path = manifest_dir / (name + ".csv")
        if path.exists():
            assert read_csv(path) == rows, "Existing manifest differs"
        else:
            with path.open("x", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
    from training.assert_nih_splits import check_splits
    result = check_splits(manifest_dir)
    (manifest_dir / "step0_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print("STEP 0 CHECKPOINT. No models were run. Stop here for review.", flush=True)


if __name__ == "__main__":
    main()
