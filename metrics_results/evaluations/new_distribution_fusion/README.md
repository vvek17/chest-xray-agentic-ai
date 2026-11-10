# Combined MedSigLIP and MedGemma model

Uses only the independently copied `data/new_distribution` train, validation and test splits. Original `Chest_X-ray` files and existing model checkpoints remain unchanged. No NIH data is used.

One shared `backend.clean_image.clean_xray` operation per image feeds both frozen backbones. Each backbone then uses its own processor settings recorded in `extractor.json`. MedSigLIP contributes a normalized 1,152-dimensional image embedding. MedGemma contributes four full-vocabulary next-token log-probabilities. The concatenated vector has 1,156 dimensions. Raw scores are retained; MedGemma features are clipped at -20 before train-only standardization. Shared first-token collisions, if present, use complete class-name sequences with the same image prefill. Actual token IDs and scoring mode are recorded before inference.

Fixed class order: Covid-19, Normal, Pneumonia, Tuberculosis.

Fixed MedGemma prompt:

- System: `You are an expert radiologist.`
- User: `Look at this chest X-ray. Which one is it: Covid-19, Normal, Pneumonia, Tuberculosis? Answer with exactly one of these names and nothing else.`

Only two linear heads train. One uses softmax and weighted cross-entropy; the other uses sigmoid and positive-weighted binary cross-entropy. Their probability vectors are averaged, then renormalized to sum to one. `backend.dual_backbone.DualBackboneClassifier.predict` exposes one prediction call. Output probabilities have not been independently calibrated.

Seed 42. Standardization mean and population standard deviation are learned from training only (standard deviation floor 1e-6). Head optimizer: AdamW. Validation grid: learning rate 0.001/0.003/0.01 and weight decay 0.0001/0.01. Maximum 200 epochs, patience 20, batch size 128. Select validation macro-F1, breaking ties by balanced accuracy then lower validation loss; remaining ties keep the first candidate. No pretrained parameters or prompts are tuned.

Frozen features are cached per image with source hashes, raw class scores, raw first-token class logits, vocabulary argmax token and cleaned-image hashes. Caching permits interrupted extraction to resume. A failed batch is recorded in `failures.jsonl` and stops the run; no image is silently dropped. There is no generated label parsing in this numeric feature pipeline. Source, manifests, prompt, targets and processor configurations are guarded against changing cached definitions.

Validation selects the model and freezes `models/new_distribution_fusion.pth` before any test image is processed. The user explicitly authorized training, validation and final test. Test is evaluated once using the frozen checkpoint, with no retuning. Test reports include 2,000 class-stratified bootstrap intervals (seed 42) for macro AUC, macro-F1 and balanced accuracy, and per-class Wilson intervals for sensitivity/specificity. Separate softmax, sigmoid and averaged outputs describe the heads of this single model.

Run from the project root: `../.venv/bin/python -B -u -m training.train_dual_backbone`. `status.json` records current progress. Completed test results are reused without further inference. `training.log` records the launched run. Feature caches and exported NPZ files hold embeddings and scores; CSV files hold raw scores and per-image probabilities. Validation/test JSON reports and PNG plots are produced as each stage finishes.

There is no complete patient metadata. Supported filename IDs and audited duplicate groups are kept together, but images of an unidentified patient may occur in different splits. Class-correlated source differences remain. Results are internal to this dataset only.
