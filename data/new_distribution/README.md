# New Chest X-ray distribution

Independent byte-for-byte copies pooled from Chest_X-ray/Training, Validation and Testing. Original files are never modified, moved or deleted. No NIH data or model inference/training.

Seed: 42. Target ratios: train 70%, val 15%, test 15%, stratified by class over whole groups. Integer class targets use largest-remainder rounding; whole-group greedy assignment and improving moves minimize class-count deviations. Grouping takes precedence over exact ratios. Original split prefixes and inventory indices avoid filename collisions.

Groups union filename person/sub subject IDs, SHA256 exact duplicates, and audited candidate near-duplicate components (64-bit DCT pHash Hamming distance <=6), including transitive links across classes. Perceptual similarity is a candidate match, not proof of shared patient identity or clinical equivalence. Largest group: 3685 retained images. Exact duplicate copies omitted only from the NEW dataset: 38; every omission is recorded. Lexicographically first original relative path is the retained representative. All near-duplicates remain in their assigned group.

There is no complete patient metadata. Only supported person/sub filename IDs are grouped; images from one unidentified patient may appear in several splits. Results are internal to this dataset only. Class-correlated source indicators remain after redistribution; random mixing does not remove them.

Manifests contain original identity, assigned group, new copy path, class, original/new splits, source path and checksum. source_inventory.csv includes every original; removed_exact_duplicates.csv accounts for omitted copies. split_summary.json records targets, actual counts and audit hashes. copy_failures.csv records copying errors.

Recheck from the project root with `python -B -m training.assert_new_distribution`. This checks images, hashes, filename/duplicate grouping, all source checksums and original modification times. Use only validation for future selection; evaluate test once after freezing a model. No model was built at this checkpoint.
