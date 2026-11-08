"""Read-only assertions for copies, complete accounting and group separation."""

import concurrent.futures
import json
from collections import defaultdict
from pathlib import Path

from training.create_new_distribution import NEW_SPLITS, OUTPUT, build_groups, read_csv, sha256


def verify(dataset=OUTPUT):
    dataset = Path(dataset)
    manifests = dataset / "manifests"
    originals = read_csv(manifests / "source_inventory.csv")
    combined = read_csv(manifests / "all_images.csv")
    removed = read_csv(manifests / "removed_exact_duplicates.csv")
    summary = json.loads((manifests / "split_summary.json").read_text())
    snapshot = json.loads((manifests / "original_snapshot.json").read_text())
    sources = {r["image_id"]: r for r in originals}
    retained = {r["image_id"]: r for r in combined}
    omitted = {r["image_id"]: r for r in removed}
    assert len(sources) == len(originals) and len(retained) == len(combined) and len(omitted) == len(removed)
    assert not retained.keys() & omitted.keys()
    assert retained.keys() | omitted.keys() == sources.keys(), "Unaccounted original images"
    seen = {name: set() for name in ("image_id", "group_id", "path", "sha256", "filename_group_id")}
    for split in NEW_SPLITS:
        rows = read_csv(manifests / f"{split}.csv")
        assert rows == [r for r in combined if r["new_split"] == split]
        for field, previous in seen.items():
            values = {r[field] for r in rows if r[field]}
            assert not previous & values, f"{field} overlaps across splits"
            previous.update(values)
        files = {str(p.absolute()) for p in (dataset / split).rglob("*") if p.is_file()}
        assert files == {r["path"] for r in rows}, f"Folder/manifest mismatch: {split}"
    assert len({r["sha256"] for r in combined}) == len(combined), "Exact duplicate retained"
    for row in removed:
        representative = retained[row["kept_image_id"]]
        assert row["sha256"] == representative["sha256"]
        assert row["group_id"] == representative["group_id"]
    # Validate the cached grouping inputs and reconstruct all transitive groups.
    audit = Path(__file__).resolve().parent.parent / "outputs/new_distribution_audit"
    for name, expected in summary["audit_input_sha256"].items():
        assert sha256(audit / name) == expected, f"Audit input changed: {name}"
    groups, expected_removed = build_groups(originals, read_csv(audit / "duplicate_groups.csv"))
    assert expected_removed == removed
    for group_id, indices in groups.items():
        members = [retained[originals[i]["image_id"]] for i in indices]
        assert all(r["group_id"] == group_id for r in members)
        assert len({r["new_split"] for r in members}) == 1
    def check_original(row):
        source = Path(row["path"])
        before = source.stat()
        assert [before.st_size, before.st_mtime_ns] == snapshot[row["image_id"]], f"Original metadata changed: {source}"
        assert sha256(source) == row["sha256"], f"Original checksum changed: {source}"
        after = source.stat()
        assert [after.st_size, after.st_mtime_ns] == snapshot[row["image_id"]]
    def check_copy(row):
        copy, source = Path(row["path"]), Path(row["source_path"])
        original = sources[row["image_id"]]
        assert row["true_label"] == original["true_label"] and row["original_split"] == original["original_split"]
        assert str(source) == original["path"] and row["sha256"] == original["sha256"]
        assert copy.parent == dataset / row["new_split"] / row["true_label"]
        assert not copy.is_symlink() and copy.stat().st_nlink == 1, f"Expected independent copy: {copy}"
        assert (copy.stat().st_dev, copy.stat().st_ino) != (source.stat().st_dev, source.stat().st_ino)
        assert sha256(copy) == row["sha256"], f"Copy checksum differs: {copy}"
    for label, rows, function in (("Original", originals, check_original), ("Copy", combined, check_copy)):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for n, _ in enumerate(pool.map(function, rows), 1):
                if n % 2000 == 0 or n == len(rows):
                    print(f"{label} assertions: {n}/{len(rows)}", flush=True)
    return {"status": "PASS", "original_images_verified_unchanged": len(originals),
            "independent_copies_checksum_verified": len(combined), "omitted_exact_copies_accounted_for": len(removed),
            "image_overlap": 0, "group_overlap": 0, "sha256_overlap": 0,
            "known_filename_id_overlap": 0, "grouping_verified": True, "unaccounted_images": 0}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
