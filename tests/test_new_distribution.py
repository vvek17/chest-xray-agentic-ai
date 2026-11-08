from collections import Counter

from training.create_new_distribution import assign_groups, build_groups


def test_transitive_groups_join_filename_and_near_duplicates_before_deduplication():
    rows = [
        {"image_id": "a", "sha256": "same", "filename_group_id": "person1", "true_label": "A", "path": "/a", "original_split": "Training"},
        {"image_id": "b", "sha256": "same", "filename_group_id": "", "true_label": "A", "path": "/b", "original_split": "Testing"},
        {"image_id": "c", "sha256": "c", "filename_group_id": "person1", "true_label": "A"},
        {"image_id": "d", "sha256": "d", "filename_group_id": "", "true_label": "B"},
        {"image_id": "e", "sha256": "e", "filename_group_id": "", "true_label": "B"},
    ]
    groups, removed = build_groups(rows, [{"image_id": key, "duplicate_group_id": "near1"} for key in ("c", "d")])
    assert sorted(sorted(indices) for indices in groups.values()) == [[0, 2, 3], [4]]
    assert removed[0]["image_id"] == "b" and removed[0]["kept_image_id"] == "a"


def test_seeded_group_assignment_is_balanced_and_keeps_large_multiclass_group():
    rows = [{"true_label": label} for label in ("A", "B") for _ in range(100)]
    groups = {"large": list(range(30)) + list(range(100, 130))}
    groups.update({f"single{i}": [i] for i in range(200) if i not in groups["large"]})
    assignment, targets, counts = assign_groups(rows, groups, ["A", "B"])
    repeated, _, _ = assign_groups(rows, groups, ["A", "B"])
    assert assignment == repeated
    assert assignment["large"] == "train"
    assert counts.tolist() == targets.tolist() == [[70, 70], [15, 15], [15, 15]]
    assert sum(Counter(assignment.values()).values()) == len(groups)
