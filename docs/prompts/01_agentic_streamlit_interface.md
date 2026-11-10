# Prompt 1: Agentic Streamlit interface

Use this prompt to generate the Streamlit interface for the project.

```text
Build a clean, beginner-friendly Streamlit interface for my Chest X-Ray AI
research project. The interface must look and behave like an AGENTIC system:
the screen shows the agent's plan, each step as it runs, the tool it used,
what it found, and the final decision.

Reuse my existing backend only. Do not write new model code. Do not invent
mock results. Every number on screen must come from a real backend call.

=== What the project does ===

The app takes one uploaded chest X-ray and shows the output of two AI parts.
It is a research prototype. It is NOT a clinical diagnostic tool. Say this
clearly in the interface.

=== Existing parts (reuse them, do not rebuild) ===

1. Image cleaning: converts the image to grayscale and cuts empty borders.
2. MedSigLIP classifier: a frozen image encoder plus a trained linear probe.
   It gives one independent sigmoid score for each of four classes:
   Covid-19, Normal, Pneumonia, Tuberculosis.
3. MedGemma: looks at the X-ray and answers Normal or Pneumonia, using my
   existing prompt versions.
4. Prompt options: prompts with a classifier hint, prompts with no hint, and
   a few-shot prompt that gives one Normal example and one Pneumonia example.
   The no-hint prompt must NOT receive the classifier prediction.
5. Upload checker (scope checker): existing logic that checks image
   similarity, supported classes, and confidence.
6. Evaluation scripts: they already saved experiment result files on disk.

The project runs models one after another. There is no autonomous agent yet.
So build the AGENT LAYER AS A THIN CONTROLLER on top of the existing calls.
It only decides the order of steps and when to stop. It must not change any
model logic, any threshold, or any prompt text.

=== Agent run, step by step ===

Step 1  Receive image. Show a preview.
Step 2  Clean image. Show the cleaned image next to the original.
Step 3  Scope check (run the existing upload checker).
        If the checks fail, STOP here. Final answer is:
        "Outside supported scope - needs review".
        Do not run the binary step. Do not show rescaled percentages.
Step 4  Classifier. Show all four sigmoid scores.
Step 5  Build the Normal vs Pneumonia comparison from the classifier output.
        Label it clearly as a rescaled two-class comparison, not a
        calibrated probability.
Step 6  MedGemma with the prompt chosen in the sidebar.
Step 7  Optional second MedGemma run with no hint and no examples, only if
        the user turned the comparison on in the sidebar.
Step 8  Final decision:
        - both agree  -> show that answer, marked "model output, not a
          verified diagnosis"
        - models disagree -> "Needs review"
        - scope check failed -> "Outside supported scope - needs review"

=== Agent display (this is the main design request) ===

- A live step timeline. Each step is one row: step name, status icon as a
  plain text tag (pending / running / done / skipped / failed), the tool it
  called, and the time it took.
- Each row expands to show the raw input and raw output of that step.
- Show the running total time and the total number of tool calls.
- When a step is skipped, say why in one short sentence.
- After the run, show a short "agent trace" summary the user can copy.
- Keep the final answer at the top in a large, calm box. No alarming colors.

=== Pages ===

1. "Analyze X-ray": upload PNG or JPG, image preview, Run button, the live
   step timeline, the four classifier scores, the MedGemma answer, the
   agreement status, the full prompt that was sent, and the elapsed time.
2. "Evaluation results": read and show my real saved evaluation files and
   their configuration. If a file is missing, show a clear empty state. Do
   not generate fake numbers.
3. "About": explain the parts, the workflow, the four supported classes, and
   the limitations.
4. Sidebar: prompt choice, the optional MedGemma comparison toggle, and the
   details of the selected prompt.

=== Rules ===

- Keep my current project structure and file layout.
- Real backend calls only.
- Label every score honestly: sigmoid score, or rescaled comparison. Never
  call them calibrated diagnostic probabilities.
- Results must always match the uploaded image and the current settings.
  Clear old results when a new image is uploaded or a setting changes.
- Handle these states clearly: loading, error, empty, and needs-review.
- Add no new dependencies beyond Streamlit and what the project already uses.
- No emojis anywhere.
- Ask me before changing any inference code or any decision rule.
```
