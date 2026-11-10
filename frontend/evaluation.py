"""Display saved evaluation files without running models or inventing metrics."""

import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from backend import config

# Change this folder only if your saved evaluation lives somewhere else.
RESULTS_DIR = config.PROJECT_ROOT / "outputs" / "hybrid_siglip_medgemma"


# Keep missing values missing, so absent metrics never appear as fabricated zeroes.
def saved_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


# Support both per-condition dictionaries and the older single-condition metric files.
def metric_conditions(data, file_name):
    if not isinstance(data, dict):
        return {}
    for wrapper in ["conditions", "results"]:
        if isinstance(data.get(wrapper), dict):
            return data[wrapper]
    if "accuracy" in data:
        condition = data.get("condition", data.get("arm", file_name))
        return {str(condition): data}
    return data


# Load only the saved numbers; predictions are not used to recompute performance.
def load_metrics(folder):
    records = []
    for path in sorted(folder.glob("*_metrics.json")):
        try:
            data = json.loads(path.read_text())
            conditions = metric_conditions(data, path.stem)
            split = path.stem.removesuffix("_metrics")
            for name, values in conditions.items():
                if not isinstance(values, dict):
                    continue
                record = {"Split": split, "Condition": name, "values": values}
                records.append(record)
        except (OSError, ValueError, TypeError) as error:
            st.warning(f"Could not read {path.name}. Please check its format.")
    return records


# Allow an existing invalid-output field name without filling missing data with zero.
def invalid_count(values):
    for name in ["invalid_outputs", "unparsed_answers", "invalid_answers"]:
        if name in values:
            return saved_number(values[name])
    return None


# Display friendly names using the same dictionary as the sidebar, without renaming stored data.
def condition_label(name, prompt_labels):
    if name in prompt_labels:
        return prompt_labels[name]
    if name == "medsiglip":
        return "MedSigLIP alone"
    return name.replace("_", " ")


# Compare every recorded condition in one table, preserving absent cells as blank.
def comparison_table(records, prompt_labels):
    rows = []
    for record in records:
        values = record["values"]
        rows.append({
            "Split": record["Split"], "Condition": condition_label(record["Condition"], prompt_labels),
            "Accuracy": saved_number(values.get("accuracy")),
            "Macro F1": saved_number(values.get("macro_f1")),
            "MCC": saved_number(values.get("mcc")),
            "Invalid outputs": invalid_count(values),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Metrics are displayed as saved. Blank cells mean the metric was not recorded.")


# Draw a heatmap only when a valid square matrix and its labels exist in the file.
def show_confusion(values):
    matrix = values.get("confusion_matrix")
    labels = values.get("confusion_labels", values.get("matrix_labels"))
    if not isinstance(matrix, list) or not isinstance(labels, list) or not labels:
        st.info("No labeled confusion matrix was saved for this condition.")
        return
    if len(matrix) != len(labels):
        st.info("The saved confusion matrix and labels have different sizes.")
        return
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(labels):
            st.info("The saved confusion matrix could not be displayed.")
            return
        for value in row:
            if saved_number(value) is None or value < 0:
                st.info("The saved confusion matrix contains an invalid value.")
                return
    figure, axis = plt.subplots(figsize=(5.5, 4.3))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(j, i, str(matrix[i][j]), ha="center", va="center")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


# Format a saved value without manufacturing a replacement for a missing field.
def show_saved_metric(column, title, value, percentage=False):
    number = saved_number(value)
    display = "Not recorded"
    if number is not None:
        if percentage:
            display = f"{number:.1%}"
        else:
            display = f"{number:g}"
    column.metric(title, display)


# Expose the stored hint diagnostics and explain how no-hint agreement should be read.
def show_hint_metrics(values):
    columns = st.columns(3)
    show_saved_metric(columns[0], "Hint follow rate", values.get("hint_follow_rate"), True)
    show_saved_metric(columns[1], "MedSigLIP errors fixed", values.get("medsiglip_errors_fixed"))
    show_saved_metric(columns[2], "Correct hints broken", values.get("correct_hints_broken"))
    st.caption("For a no-hint condition, hint follow rate measures agreement only.")


# Let the user inspect stored predictions without silently deriving new metric values.
def show_prediction_files(folder):
    paths = sorted(folder.glob("*_predictions.csv"))
    with st.expander("Inspect individual predictions"):
        if not paths:
            st.info("No prediction CSV files have been saved in this folder yet.")
            return
        selected = st.selectbox("Prediction file", paths, format_func=lambda path: path.name)
        try:
            with selected.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            else:
                st.info("This prediction file has no rows yet.")
            st.download_button("Download predictions", selected.read_bytes(), selected.name, "text/csv")
        except (OSError, ValueError, csv.Error):
            st.warning("This prediction file could not be read. Please check its format.")


# Keep results accessible even when no image is uploaded and no models are loaded.
def show_evaluation(prompt_labels):
    st.subheader("Evaluation results")
    st.write("Compare saved conditions on the same evaluation images.")
    st.caption("Saved evaluations may use raw hints and few-shot context. This app first gates uploads with the existing scope checker and then rescales binary hints; only the few-shot option supplies context examples. Check the saved report before comparing configurations.")
    folder = RESULTS_DIR
    if not folder.exists():
        st.info("No evaluation files yet. Save your results in outputs/hybrid_siglip_medgemma/ to display them here.")
        return
    with st.expander("Saved evaluation configuration"):
        paths = [path for path in sorted(folder.glob("*.json"))
                 if not path.name.endswith("_metrics.json")]
        if not paths:
            st.info("No configuration JSON files have been saved in this folder.")
        for path in paths:
            st.write(path.name)
            try:
                st.json(json.loads(path.read_text()))
            except (OSError, ValueError, UnicodeError):
                st.warning(f"Could not read {path.name}.")
    records = load_metrics(folder)
    if records:
        comparison_table(records, prompt_labels)
        options = []
        for record in records:
            options.append(record["Split"] + " · " + condition_label(record["Condition"], prompt_labels))
        choice = st.selectbox("Inspect a condition", options)
        record = records[options.index(choice)]
        with st.container(border=True):
            show_hint_metrics(record["values"])
            show_confusion(record["values"])
    else:
        st.info("No readable *_metrics.json files have been saved yet.")
    show_prediction_files(folder)
    with st.expander("Read the saved report"):
        report = folder / "report.md"
        if not report.exists():
            st.info("The report has not been saved yet.")
        else:
            try:
                st.markdown(report.read_text())
            except (OSError, UnicodeError):
                st.warning("The saved report could not be read.")
