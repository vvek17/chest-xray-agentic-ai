# Frozen hybrid evaluation

Research only. Patient independence and pretraining overlap are unverified.
Original Testing images only. Exact SHA-256 duplicates and recorded earlier MedGemma inputs are excluded.
Exclusion is limited to available logs/manifests; unrecorded historical use cannot be established.
All three hybrids share the same two few-shot examples as 2c_fs1. The probe hint is binary argmax, not the four-class winner.
Hint confidence means the uncalibrated winning sigmoid score, without renormalizing the two scores.
Invalid outputs count as errors and as broken correct hints where applicable.
For no-hint baselines, hint-follow metrics mean agreement only, not exposure to a hint.
Rates use all cases for hint following, wrong hints for fixes, and correct hints for breaks. Null means no eligible cases.
ROC-AUC/calibration are not reported for label-only MedGemma answers.

