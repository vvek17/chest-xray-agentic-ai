# Run the new combined frozen model manually

Uses `data/new_distribution` only. Both MedSigLIP and MedGemma remain frozen; the original Chest_X-ray database is read-only. No DINOv2, RAD-DINO or NIH experiments are included.

From the project folder, run the complete workflow:

```bash
cd /Users/vivek/Desktop/Alzaimer-AgenticAI/chest-xray-pneumonia
../.venv/bin/python -B -u -m training.train_probe --new-model
```

Alternatively, run stages separately:

```bash
../.venv/bin/python -B -u -m training.run_new_model_baselines --stage features
../.venv/bin/python -B -u -m training.run_new_model_baselines --stage train
../.venv/bin/python -B -u -m training.run_new_model_baselines --stage test
```

The main file to read is `training/train_probe.py`. Its new `train_new_probe_with_epochs` function follows the original simple training loop: train on batches, check validation, remember the best epoch. `main_new_model` starts the three-step workflow. Feature caching and detailed result saving are kept in helpers so the training loop stays readable. Running without `--new-model` retains the existing experiment.

The features stage caches training and validation only. The test stage opens test images after model selection has been frozen. Repeating a completed test prints the saved result without further inference. Interrupted extraction resumes existing per-image caches. Compatible partial features from the earlier stopped run are copied and reused.

## Methods

All methods use the same new split manifests and the same fixed class order: Covid-19, Normal, Pneumonia, Tuberculosis.

- Zero-shot combined baseline: a fixed equal average of softmax-normalized MedSigLIP cosine text matches and MedGemma class-score probabilities. This is an explicit untrained score-fusion baseline, with no fitted temperatures or weights. Existing class text descriptions and the fixed four-class MedGemma prompt are preserved in the protocol.
- k-NN: cosine neighbors of the standardized, normalized combined features. Choose k from 1, 3, 5, 7, 11, 21 using validation macro-F1, then balanced accuracy, then smaller k. k-NN has no epochs.
- Scikit-learn logistic regression: train-only scaling, balanced class weights, C=1 and maximum 2,000 solver iterations. These solver iterations are distinct from PyTorch epochs. Convergence failure stops and is recorded.
- PyTorch linear probe: two linear heads with softmax and sigmoid, jointly trained with weighted cross-entropy plus positive-weighted binary cross-entropy. Average the probabilities and renormalize. Train for exactly 10 epochs; save the epoch with best validation macro-F1, then balanced accuracy, then lower validation loss. Learning rate 0.001, weight decay 0.0001, batch size 128. No early stopping shortens the ten epochs.

Frozen feature extraction runs once per image: the same cleaned RGB image feeds both processors; MedSigLIP gives 1,152 normalized image features and MedGemma gives four class log-probabilities. Raw scores are saved; log-probability features are clipped at -20 before train-only standardization. The tokenizer targets and each image processor's settings are recorded. Full class-name sequence scores are used if first-token targets collide.

## Outputs

- `outputs/features/medsiglip_medgemma_new_distribution/`: per-image caches, train/val/test feature NPZ files, raw score CSVs, frozen text embeddings, processor settings and progress in `status.json`.
- `models/new_model_baselines/linear_probe.pth`: best epoch, both learned heads, train scaling, fixed class order and extractor metadata.
- `models/new_model_baselines/classifiers.joblib`: fitted k-NN/logistic classifiers and train scaling; local trusted artifact.
- `outputs/new_model_baselines/`: frozen protocol/selection, validation/test metrics, per-image predictions, confusion matrices, test ROC plots and per-method 2,000-resample bootstrap files. Test metrics include 95% bootstrap intervals for macro AUC, macro-F1 and balanced accuracy, and Wilson intervals for per-class sensitivity/specificity. `test_head_information.json` describes the softmax, sigmoid and averaged heads of the same PyTorch model.
- `outputs/new_model_baselines/linear_probe/`: ten-epoch history, training curves and validation details for the best epoch.

Model definitions, manifests and preprocessing are guarded by hashes; the code refuses to mix changed definitions with cached features. Errors are recorded in `failures.jsonl` and stop the relevant stage. No images are silently dropped. Generated answer parsing is not used in this numeric feature workflow.

Feature extraction through MedGemma is slow; the initial measured rate suggests many hours for all images. Head training starts once train/validation features are ready. No automatic training was launched for this manual runner.

There is no complete patient metadata. Known filename IDs and duplicate clusters are grouped, but unknown patient overlap cannot be ruled out. Source differences remain after redistribution. Results are internal to this dataset only; predictions are not independently calibrated.
