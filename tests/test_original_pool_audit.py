import random

from training.audit_original_pool import BKTree, UnionFind, filename_info


def test_filename_group_evidence_without_invented_ids():
    assert filename_info("person1345_bacteria_3426.jpeg")[1] == "filename:person1345"
    assert filename_info("sub-S308252_ses-E31279_run-1.png")[1] == "filename:sub-S308252"
    assert filename_info("Normal-1234.png")[1] == ""
    assert filename_info("CHNCXR_0015_0.png")[1] == ""


def test_hamming_tree_matches_brute_force_including_duplicate_hashes():
    rng = random.Random(42)
    values = [rng.getrandbits(64) for _ in range(30)]
    values += [values[0], values[0] ^ 7, values[0] ^ 127]
    tree = BKTree()
    for i, value in enumerate(values):
        expected = {(j, (value ^ previous).bit_count()) for j, previous in enumerate(values[:i])
                    if (value ^ previous).bit_count() <= 6}
        assert set(tree.query(value, 6)) == expected
        tree.add(value, i)


def test_duplicate_components_are_transitive():
    union = UnionFind(4)
    union.union(0, 1)
    union.union(1, 2)
    assert union.find(0) == union.find(2)
    assert union.find(0) != union.find(3)
