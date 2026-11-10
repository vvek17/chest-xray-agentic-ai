# Prompts used to build this project

These are saved development instructions for the coding assistant. The first two
correspond to the implemented interface and prompt-log runner. Prompts 3 and 4
are proposed follow-up work; their presence does not establish a patient-grouped
split or a completed report.

| File | What it builds |
| --- | --- |
| [01_agentic_streamlit_interface.md](01_agentic_streamlit_interface.md) | The agentic Streamlit interface on top of the existing backend |
| [02_prompt_run_logs.md](02_prompt_run_logs.md) | A script that runs every prompt version on 10 images and writes one JSON log per prompt |
| [03_patient_grouped_prompt_split.md](03_patient_grouped_prompt_split.md) | A patient-grouped, source-tagged prompt split, to remove patient leakage |
| [04_gold_standard_prompt_report.md](04_gold_standard_prompt_report.md) | A CLAIM-style report per prompt: metrics with confidence intervals, per-source accuracy, failure analysis |

These prompts follow the same rules: reuse the existing backend, real calls
only, no change to model logic or thresholds, and honest score labels
(sigmoid scores, not calibrated probabilities).
