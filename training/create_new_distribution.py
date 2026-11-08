"""Create Step 2 only: independent dataset copies; never write to Chest_X-ray.

Run from the project root: python -B -m training.create_new_distribution
Refuses to overwrite an existing output directory. No model is loaded.
"""

import concurrent.futures
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from training.audit_original_pool import IMAGE_TYPES, ORIGINAL, ROOT, SPLITS, UnionFind

AUDIT = ROOT / "outputs/new_distribution_audit"
OUTPUT = ROOT / "data/new_distribution"
NEW_SPLITS = ("train", "val", "test")
RATIOS = (0.70, 0.15, 0.15)
SEED = 42


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_groups(rows, memberships):
    """Union filename evidence and audit duplicate components, then deduplicate."""
    ids = {r["image_id"]: i for i, r in enumerate(rows)}
    assert len(ids) == len(rows), "Repeated image IDs in audit"
    union, seen = UnionFind(len(rows)), {}
    for i, row in enumerate(rows):
        for kind, value in (("sha", row["sha256"]), ("filename", row["filename_group_id"])):
            if value:
                key = (kind, value)
                if key in seen:
                    union.union(i, seen[key])
                else:
                    seen[key] = i
    for member in memberships:
        i = ids[member["image_id"]]
        key = ("duplicate", member["duplicate_group_id"])
        if key in seen:
            union.union(i, seen[key])
        else:
            seen[key] = i
    components = defaultdict(list)
    for i in range(len(rows)):
        components[union.find(i)].append(i)
    group_ids = {}
    for members in components.values():
        identity = "\n".join(sorted(rows[i]["image_id"] for i in members))
        group_id = "group_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
        for i in members:
            group_ids[i] = group_id
    groups, representatives, removed = defaultdict(list), {}, []
    for i in sorted(range(len(rows)), key=lambda j: rows[j]["image_id"]):
        row = rows[i]
        digest = row["sha256"]
        if digest in representatives:
            representative = representatives[digest]
            assert row["true_label"] == rows[representative]["true_label"], "Conflicting exact-duplicate labels"
            removed.append({"image_id": row["image_id"], "source_path": row["path"],
                            "true_label": row["true_label"], "original_split": row["original_split"],
                            "kept_image_id": rows[representative]["image_id"], "sha256": digest,
                            "group_id": group_ids[i], "reason": "Exact file SHA256 duplicate; original retained untouched"})
        else:
            representatives[digest] = i
            groups[group_ids[i]].append(i)
    return dict(groups), removed


def assign_groups(rows, groups, classes, seed=SEED):
    """Seeded class-stratified group allocation; integer targets, no group splits."""
    totals = np.array([sum(r["true_label"] == c for members in groups.values()
                           for r in (rows[i] for i in members)) for c in classes])
    targets = np.zeros((3, len(classes)), dtype=int)
    for c, total in enumerate(totals):
        ideal = np.asarray(RATIOS) * total
        target = np.floor(ideal).astype(int)
        remainder_order = sorted(range(3), key=lambda s: (-(ideal[s] - target[s]), s))
        for s in remainder_order[:int(total - target.sum())]:
            target[s] += 1
        targets[:, c] = target
    vectors = {g: np.array([sum(rows[i]["true_label"] == c for i in members)
                            for c in classes]) for g, members in groups.items()}
    rng = random.Random(seed)
    order = sorted(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: len(groups[g]), reverse=True)
    counts = np.zeros_like(targets)
    assignment = {}
    # Minimize summed squared deviations from per-class integer targets.
    # A common class-total denominator keeps each class equally weighted.
    def objective(array):
        return float(np.square((array - targets) / totals).sum())
    for g in order:
        choices = []
        for s in range(3):
            proposed = counts.copy()
            proposed[s] += vectors[g]
            choices.append((objective(proposed), s))
        s = min(choices)[1]
        assignment[g] = s
        counts[s] += vectors[g]
    # Deterministic improving whole-group moves remove avoidable rounding error.
    passes = 0
    while True:
        changed = False
        for g in reversed(order):
            old = assignment[g]
            best_value, best = objective(counts), old
            for s in range(3):
                if s == old:
                    continue
                proposed = counts.copy()
                proposed[old] -= vectors[g]
                proposed[s] += vectors[g]
                value = objective(proposed)
                if value < best_value - 1e-15:
                    best_value, best = value, s
            if best != old:
                counts[old] -= vectors[g]
                counts[best] += vectors[g]
                assignment[g] = best
                changed = True
        passes += 1
        if not changed:
            break
        assert passes < 1000, "Group assignment did not converge"
    assert (counts > 0).all(), "Every split must contain every class"
    return {g: NEW_SPLITS[s] for g, s in assignment.items()}, targets, counts


def copy_image(row):
    source, destination = Path(row["source_path"]), Path(row["path"])
    digest = hashlib.sha256()
    # Opening the source read-only and the destination exclusively prevents overwrites.
    with source.open("rb") as src, destination.open("xb") as dst:
        while chunk := src.read(1024 * 1024):
            digest.update(chunk)
            dst.write(chunk)
    assert digest.hexdigest() == row["sha256"], f"Original differs from audit: {source}"
    assert sha256(destination) == row["sha256"], f"Copy checksum failed: {destination}"


def main():
    assert not OUTPUT.exists(), f"Refusing to overwrite {OUTPUT}"
    rows = read_csv(AUDIT / "image_inventory.csv")
    assert rows and all(r["status"] == "ok" and int(r["nonfinite_pixels"]) == 0 for r in rows), "Audit contains failures"
    actual = {str(p.relative_to(ORIGINAL)) for split in SPLITS
              for p in (ORIGINAL / split).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_TYPES}
    assert actual == {r["image_id"] for r in rows}, "Original file list differs from audit"
    classes = sorted({r["true_label"] for r in rows})
    snapshot = {}
    for row in rows:
        source = Path(row["path"])
        assert source.resolve() == (ORIGINAL / row["image_id"]).resolve()
        stat = source.stat()
        assert stat.st_size == int(row["bytes"]), f"Original size changed: {source}"
        snapshot[row["image_id"]] = [stat.st_size, stat.st_mtime_ns]
    groups, removed = build_groups(rows, read_csv(AUDIT / "duplicate_groups.csv"))
    assignment, targets, counts = assign_groups(rows, groups, classes)
    print("Class order:", classes, flush=True)
    print("Targets:", targets.tolist(), "Actual:", counts.tolist(), flush=True)
    print("Groups:", len(groups), "Largest:", max(map(len, groups.values())), "Removed exact copies:", len(removed), flush=True)
    OUTPUT.mkdir()
    manifests = OUTPUT / "manifests"
    manifests.mkdir()
    for split in NEW_SPLITS:
        for label in classes:
            (OUTPUT / split / label).mkdir(parents=True)
    output_rows = []
    for g in sorted(groups):
        split = assignment[g]
        for i in groups[g]:
            row = rows[i]
            name = f"{i:05d}__{Path(row['path']).name}"
            output_rows.append({"image_id": row["image_id"], "group_id": g,
                                "path": str(OUTPUT / split / row["true_label"] / name),
                                "true_label": row["true_label"], "original_split": row["original_split"],
                                "new_split": split, "source_path": row["path"], "sha256": row["sha256"],
                                "filename_group_id": row["filename_group_id"]})
    output_rows.sort(key=lambda r: (r["new_split"], r["true_label"], r["image_id"]))
    failures = []
    def attempt(row):
        try:
            copy_image(row)
        except Exception as error:
            return {"image_id": row["image_id"], "error": f"{type(error).__name__}: {error}"}
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for number, failure in enumerate(pool.map(attempt, output_rows), 1):
            if failure:
                failures.append(failure)
            if number % 1000 == 0 or number == len(output_rows):
                print(f"Copied and checksum-verified {number}/{len(output_rows)}", flush=True)
    write_csv(manifests / "copy_failures.csv", failures, ["image_id", "error"])
    assert not failures, f"Copy failures: {len(failures)}; see copy_failures.csv. Dataset is incomplete."
    fields = list(output_rows[0])
    write_csv(manifests / "all_images.csv", output_rows, fields)
    for split in NEW_SPLITS:
        write_csv(manifests / f"{split}.csv", [r for r in output_rows if r["new_split"] == split], fields)
    write_csv(manifests / "removed_exact_duplicates.csv", removed,
              ["image_id", "source_path", "true_label", "original_split", "kept_image_id", "sha256", "group_id", "reason"])
    write_csv(manifests / "source_inventory.csv", rows, list(rows[0]))
    (manifests / "original_snapshot.json").write_text(json.dumps(snapshot))
    summary = {"seed": SEED, "requested_ratios": dict(zip(NEW_SPLITS, RATIOS)),
               "storage": "Independent byte-for-byte copies; no symlinks or hard links",
               "original_dataset": str(ORIGINAL), "classes": classes,
               "original_images": len(rows), "retained_images": len(output_rows),
               "exact_duplicate_copies_omitted_from_new_dataset": len(removed),
               "groups": len(groups), "largest_group_images": max(map(len, groups.values())),
               "counts": {s: dict(Counter(r["true_label"] for r in output_rows if r["new_split"] == s)) for s in NEW_SPLITS},
               "target_counts": {s: dict(zip(classes, map(int, targets[k]))) for k, s in enumerate(NEW_SPLITS)},
               "load_failures_in_audit": 0, "copy_failures": len(failures), "nonfinite_pixels_in_audit": 0,
               "source_check": "Known class-correlated formats, dimensions, modes and filename patterns persist; user authorized new splits after audit checkpoint.",
               "audit_input_sha256": {name: sha256(AUDIT / name) for name in ("image_inventory.csv", "duplicate_groups.csv", "audit_summary.json")},
               "model_training_or_inference": False, "NIH_used": False}
    (manifests / "split_summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT / "README.md").write_text(
        "# New Chest X-ray distribution\n\n"
        "Independent byte-for-byte copies pooled from Chest_X-ray/Training, Validation and Testing. "
        "Original files are never modified, moved or deleted. No NIH data or model inference/training.\n\n"
        "Seed: 42. Target ratios: train 70%, val 15%, test 15%, stratified by class over whole groups. "
        "Integer class targets use largest-remainder rounding; whole-group greedy assignment and improving moves minimize class-count deviations. "
        "Grouping takes precedence over exact ratios. Original split prefixes and inventory indices avoid filename collisions.\n\n"
        "Groups union filename person/sub subject IDs, SHA256 exact duplicates, and audited candidate near-duplicate "
        "components (64-bit DCT pHash Hamming distance <=6), including transitive links across classes. "
        "Perceptual similarity is a candidate match, not proof of shared patient identity or clinical equivalence. "
        f"Largest group: {summary['largest_group_images']} retained images. "
        f"Exact duplicate copies omitted only from the NEW dataset: {len(removed)}; every omission is recorded. "
        "Lexicographically first original relative path is the retained representative. All near-duplicates remain in their assigned group.\n\n"
        "There is no complete patient metadata. Only supported person/sub filename IDs are grouped; "
        "images from one unidentified patient may appear in several splits. Results are internal to this dataset only. "
        "Class-correlated source indicators remain after redistribution; random mixing does not remove them.\n\n"
        "Manifests contain original identity, assigned group, new copy path, class, original/new splits, source path and checksum. "
        "source_inventory.csv includes every original; removed_exact_duplicates.csv accounts for omitted copies. "
        "split_summary.json records targets, actual counts and audit hashes. copy_failures.csv records copying errors.\n\n"
        "Recheck from the project root with `python -B -m training.assert_new_distribution`. "
        "This checks images, hashes, filename/duplicate grouping, all source checksums and original modification times. "
        "Use only validation for future selection; evaluate test once after freezing a model. No model was built at this checkpoint.\n"
    )
    from training.assert_new_distribution import verify
    result = verify(OUTPUT)
    (manifests / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"counts": summary["counts"], "verification": result}, indent=2), flush=True)
    print("STEP 2 CHECKPOINT: new copies ready; no model run.", flush=True)


if __name__ == "__main__":
    main()
