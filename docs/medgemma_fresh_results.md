# Pretrained MedGemma: Normal vs Pneumonia

Frozen 2c_fs1 prompt; no model weights were trained. Development uses 20 images per class, validation 40, test 70.
Images in previous prompt logs and the interrupted run were excluded by SHA-256. Patient independence and pretraining overlap have not been established.
All predictions are generated for this fresh run with new few-shot examples; only this run’s checkpoints are reused on resume.
ROC-AUC, PR-AUC, log loss and calibration are unavailable because these runs produce labels, not class probabilities.

| Split | N | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| train | 40 | 0.9000 | 0.9000 | 0.8990 | 0.8990 | 0.8165 |
| val | 80 | 0.8875 | 0.8875 | 0.8873 | 0.8873 | 0.7772 |
| test | 140 | 0.9071 | 0.9071 | 0.9071 | 0.9071 | 0.8144 |

## train

| Class | Precision | Recall / sensitivity | Specificity | F1 | Support |
|---|---:|---:|---:|---:|---:|
| Normal | 0.8333 | 1.0000 | 0.8000 | 0.9091 | 20.0 |
| Pneumonia | 1.0000 | 0.8000 | 1.0000 | 0.8889 | 20.0 |

Unparsed answers: 0. Confusion matrix rows = true class; columns = predicted class; order: Normal, Pneumonia, Other.

```
[[20  0  0]
 [ 4 16  0]
 [ 0  0  0]]
```


## val

| Class | Precision | Recall / sensitivity | Specificity | F1 | Support |
|---|---:|---:|---:|---:|---:|
| Normal | 0.8605 | 0.9250 | 0.8500 | 0.8916 | 40.0 |
| Pneumonia | 0.9189 | 0.8500 | 0.9250 | 0.8831 | 40.0 |

Unparsed answers: 0. Confusion matrix rows = true class; columns = predicted class; order: Normal, Pneumonia, Other.

```
[[37  3  0]
 [ 6 34  0]
 [ 0  0  0]]
```


## test

| Class | Precision | Recall / sensitivity | Specificity | F1 | Support |
|---|---:|---:|---:|---:|---:|
| Normal | 0.9014 | 0.9143 | 0.9000 | 0.9078 | 70.0 |
| Pneumonia | 0.9130 | 0.9000 | 0.9143 | 0.9065 | 70.0 |

Unparsed answers: 0. Confusion matrix rows = true class; columns = predicted class; order: Normal, Pneumonia, Other.

```
[[64  6  0]
 [ 7 63  0]
 [ 0  0  0]]
```

This is the completed standalone few-shot MedGemma evaluation, not a hybrid result. Published prediction CSVs, metrics and confusion matrices are available in [metrics results](../metrics_results/evaluations/medgemma_fresh_20_40_70/).
