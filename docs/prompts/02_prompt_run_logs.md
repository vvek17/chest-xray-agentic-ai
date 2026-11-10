# Prompt 2: Per-prompt classification logs

Use this prompt to generate the logging script that runs every prompt version
on the same 10 images and saves one JSON log per prompt.

```text
Write one script: chest-xray-pneumonia/prompt_engineering/run_prompt_logs.py

Goal: for every prompt version in prompt_engineering/prompts.py, classify the
same 10 images, and save one JSON log per prompt.

Reuse the existing backend (clean_image, classifier, medgemma, xray_check).
Do not change model code, prompt text, or thresholds.

For each prompt, for each image:
- clean the image
- get the four classifier sigmoid scores
- build the prompt as the project already does (no-hint prompt gets no hint)
- call MedGemma and keep the raw answer
- time each stage with time.perf_counter()

Save to logs/prompt_runs/<prompt_name>.json:

{
  "prompt_name": "...",
  "prompt_content": "full prompt text",
  "total_time_s": 0.0,
  "results": [
    {
      "image_path": "...",
      "input": "exact text sent to MedGemma",
      "output": {
        "scores": {"covid19":0,"normal":0,"pneumonia":0,"tuberculosis":0},
        "medgemma_raw": "..."
      },
      "time_s": {"clean":0,"classifier":0,"medgemma":0,"total":0},
      "final_output": "Normal | Pneumonia | Needs review"
    }
  ]
}

Rules:
- same 10 images for every prompt (sort file names, fixed seed)
- if one image fails, log the error and keep going
- json indent=2
- no new dependencies, no emojis
```
