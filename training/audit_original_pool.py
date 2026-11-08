"""Step 1 only: read-only original-data audit, no splits or model inference.

python -m training.audit_original_pool
Writes audit artifacts under outputs/new_distribution_audit only.
64-bit DCT pHash, Hamming radius <=6; candidate near-duplicates are not proof
of identity and are never deleted by this audit.
"""

import concurrent.futures
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT.parent / "Chest_X-ray"
OUT = ROOT / "outputs/new_distribution_audit"
SPLITS = ("Training", "Validation", "Testing")
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".dcm"}
PHASH_RADIUS = 6


def filename_info(name):
    stem = Path(name).stem
    match = re.match(r"^person(\d+)_", stem, re.I)
    if match:
        return "person<id>_<pathogen>_<image>", "filename:person" + match.group(1), "person prefix"
    match = re.match(r"^sub-([^_]+)_ses-", stem)
    if match:
        return "sub-<subject>_ses-<session>...", "filename:sub-" + match.group(1), "explicit sub subject prefix"
    for prefix in ["CHNCXR_", "NORMAL2-IM-", "IM-", "Normal-", "Tuberculosis-"]:
        if stem.startswith(prefix):
            return prefix + "<numbers>", "", "no confirmed patient/group identity"
    if re.fullmatch(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", stem, re.I):
        return "UUID filename", "", "image identifier; patient unknown"
    if stem.isdigit():
        return "numeric filename", "", "image identifier; patient unknown"
    return re.sub(r"\d+", "<n>", stem), "", "no confirmed patient/group identity"


def phash(image):
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    values = dctn(np.asarray(gray, dtype=np.float64), type=2)[:8, :8]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite perceptual-hash coefficients")
    bits = values > np.median(values)
    return f"{int(''.join('1' if b else '0' for b in bits.flat), 2):016x}"


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value, item):
        if self.root is None:
            self.root = [value, [item], {}]
            return
        node = self.root
        while True:
            distance = (node[0] ^ value).bit_count()
            if distance == 0:
                node[1].append(item)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [item], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        stack = [self.root] if self.root is not None else []
        found = []
        while stack:
            node = stack.pop()
            distance = (node[0] ^ value).bit_count()
            if distance <= radius:
                found.extend((item, distance) for item in node[1])
            stack.extend(child for edge, child in node[2].items()
                         if distance - radius <= edge <= distance + radius)
        return found


def inspect_file(item):
    split, label, path = item
    before = path.stat()
    pattern, group, evidence = filename_info(path.name)
    row = {"image_id": str(path.relative_to(ORIGINAL)), "path": str(path.resolve()),
           "true_label": label, "original_split": split, "filename": path.name,
           "extension": path.suffix.lower(), "bytes": before.st_size,
           "filename_pattern": pattern, "filename_group_id": group, "group_evidence": evidence,
           "sha256": "", "phash64": "", "format": "", "width": "", "height": "", "mode": "",
           "status": "ok", "error": "", "nonfinite_pixels": 0, "original_unchanged": True}
    try:
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            row.update(format=image.format, width=image.width, height=image.height, mode=image.mode)
            image.load()
            if image.mode == "F":
                row["nonfinite_pixels"] = int((~np.isfinite(np.asarray(image))).sum())
                if row["nonfinite_pixels"]:
                    raise ValueError("Non-finite original pixels")
            row["phash64"] = phash(image)
    except Exception as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}")
    after = path.stat()
    row["original_unchanged"] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return row


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    classes = sorted(path.name for path in (ORIGINAL / SPLITS[0]).iterdir() if path.is_dir())
    assert classes
    for split in SPLITS:
        assert sorted(p.name for p in (ORIGINAL / split).iterdir() if p.is_dir()) == classes
    items, excluded = [], []
    for split in SPLITS:
        for label in classes:
            for path in sorted((ORIGINAL / split / label).rglob("*")):
                if path.is_file():
                    if path.suffix.lower() in IMAGE_TYPES:
                        items.append((split, label, path))
                    else:
                        excluded.append({"path": str(path), "original_split": split, "true_label": label,
                                         "reason": "Non-image extension; explicitly reported, not silently omitted"})
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(inspect_file, items):
            rows.append(row)
            if len(rows) % 1000 == 0 or len(rows) == len(items):
                print("Audited", len(rows), "/", len(items), flush=True)
    assert all(r["original_unchanged"] for r in rows), "An original changed during inspection"
    write_csv(OUT / "image_inventory.csv", rows, list(rows[0]))
    write_csv(OUT / "load_failures.csv", [r for r in rows if r["status"] != "ok"], list(rows[0]))
    write_csv(OUT / "excluded_nonimage_files.csv", excluded, ["path", "original_split", "true_label", "reason"])
    by_sha = defaultdict(list)
    for i, row in enumerate(rows):
        if row["sha256"]:
            by_sha[row["sha256"]].append(i)
    exact = {sha: members for sha, members in by_sha.items() if len(members) > 1}
    uf = UnionFind(len(rows))
    for members in exact.values():
        for index in members[1:]:
            uf.union(members[0], index)
    tree = BKTree()
    near_pair_count, distances, near_participants = 0, Counter(), set()
    representatives = {}
    for number, (sha, members) in enumerate(by_sha.items(), 1):
        representative = next((i for i in members if rows[i]["phash64"]), None)
        if representative is None:
            continue
        representatives[sha] = representative
        value = int(rows[representative]["phash64"], 16)
        for other_sha, distance in tree.query(value, PHASH_RADIUS):
            uf.union(representative, representatives[other_sha])
            near_participants.update([representative, representatives[other_sha]])
            near_pair_count += 1
            distances[distance] += 1
        tree.add(value, sha)
        if number % 3000 == 0:
            print("Perceptual duplicate search:", number, "/", len(by_sha), flush=True)
    components = defaultdict(list)
    for i in range(len(rows)):
        components[uf.find(i)].append(i)
    near_roots = {uf.find(i) for i in near_participants}
    near = {root: components[root] for root in sorted(near_roots)}
    def group_summary(groups):
        return {"groups": len(groups), "member_images": sum(map(len, groups.values())),
                "groups_spanning_original_splits": sum(len({rows[i]["original_split"] for i in members}) > 1 for members in groups.values()),
                "groups_spanning_classes": sum(len({rows[i]["true_label"] for i in members}) > 1 for members in groups.values())}
    memberships = []
    for kind, groups in [("exact", exact), ("near_candidate", near)]:
        for index, members in enumerate(groups.values(), 1):
            group_id = f"{kind}_{index:05}"
            for i in members:
                memberships.append({"duplicate_type": kind, "duplicate_group_id": group_id,
                                    "image_id": rows[i]["image_id"], "true_label": rows[i]["true_label"],
                                    "original_split": rows[i]["original_split"], "sha256": rows[i]["sha256"],
                                    "phash64": rows[i]["phash64"]})
    write_csv(OUT / "duplicate_groups.csv", memberships,
              ["duplicate_type", "duplicate_group_id", "image_id", "true_label", "original_split", "sha256", "phash64"])
    per_class = {}
    for label in classes:
        subset = [r for r in rows if r["true_label"] == label]
        sizes = Counter(f"{r['width']}x{r['height']}" for r in subset if r["status"] == "ok")
        grouped = [r for r in subset if r["filename_group_id"]]
        per_class[label] = {
            "images": len(subset), "original_splits": dict(Counter(r["original_split"] for r in subset)),
            "formats": dict(Counter(r["format"] for r in subset)),
            "extensions": dict(Counter(r["extension"] for r in subset)),
            "color_modes": dict(Counter(r["mode"] for r in subset)),
            "image_sizes": dict(sizes), "most_common_sizes": sizes.most_common(5),
            "filename_patterns": dict(Counter(r["filename_pattern"] for r in subset)),
            "filename_group_id_coverage": {"images": len(grouped), "fraction": len(grouped) / len(subset),
                                           "unique_groups": len({r["filename_group_id"] for r in grouped}),
                                           "evidence": dict(Counter(r["group_evidence"] for r in grouped))},
        }
    differences = []
    for dimension in ["formats", "extensions", "color_modes", "image_sizes", "filename_patterns"]:
        signatures = [frozenset(per_class[label][dimension]) for label in classes]
        if any(signature != signatures[0] for signature in signatures[1:]):
            differences.append(dimension)
    counts = [per_class[label]["images"] for label in classes]
    ratio = max(counts) / min(counts)
    report = {"stage": "STEP 1 CHECKPOINT", "original_dataset": str(ORIGINAL), "classes": classes,
              "K": len(classes), "total_images": len(rows), "seed_for_future_splits": 42,
              "per_class": per_class, "load_failures": sum(r["status"] != "ok" for r in rows),
              "group_id_parse_errors": 0, "nonfinite_pixel_values": sum(r["nonfinite_pixels"] for r in rows),
              "excluded_nonimage_files": len(excluded), "original_files_changed": 0,
              "exact_duplicates": dict(group_summary(exact), redundant_copies=sum(len(g) - 1 for g in exact.values())),
              "near_duplicate_candidates": dict(group_summary(near), pair_count_distinct_sha=near_pair_count,
                                                pair_distance_histogram=dict(distances),
                                                method="64-bit DCT pHash, 32x32 grayscale, top-left 8x8, median threshold; Hamming <=6; transitive components",
                                                caveat="Perceptual matches are candidates, not confirmed patient identity or clinical equivalence; no images removed"),
              "class_imbalance": {"max_min_ratio": ratio, "severe_threshold": 2.0, "severe": ratio >= 2},
              "source_check": {"differences": differences,
                               "status": "WARN_AND_STOP" if differences else "PASS",
                               "interpretation": "Class-correlated image properties and naming differ; these are source indicators, not verified hospital metadata."},
              "next_step_allowed_without_confirmation": False, "new_distribution_created": False,
              "dataset_copies_or_symlinks_created": "Neither; audit only", "model_inference_or_training": False}
    (OUT / "audit_summary.json").write_text(json.dumps(report, indent=2))
    print("\nCLASSES:", classes, "K:", len(classes), "TOTAL:", len(rows), flush=True)
    for label, data in per_class.items():
        print(label, json.dumps({k: data[k] for k in ["images", "original_splits", "formats", "color_modes", "most_common_sizes", "filename_group_id_coverage"]}), flush=True)
    print("EXACT DUPLICATES:", json.dumps(report["exact_duplicates"]), flush=True)
    print("NEAR-DUPLICATE CANDIDATES:", json.dumps(report["near_duplicate_candidates"]), flush=True)
    print("FAILURES:", report["load_failures"], "NONFINITE:", report["nonfinite_pixel_values"], "IMBALANCE:", json.dumps(report["class_imbalance"]), flush=True)
    print("SOURCE CHECK:", json.dumps(report["source_check"]), flush=True)
    print("STOP. No new dataset, copies, symlinks, model training or NIH data used.", flush=True)


if __name__ == "__main__":
    main()
