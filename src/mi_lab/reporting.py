"""Write self-contained local-evaluation reports for the ECG subtype experiment."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import markdown as markdown_renderer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


def _curve_arrays(test_curve: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(test_curve["y_true"], dtype=int)
    probability = np.asarray(test_curve["probability"], dtype=float)
    if (
        y_true.ndim != 1
        or probability.shape != y_true.shape
        or set(np.unique(y_true)) != {0, 1}
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("test_curve requires matching binary labels and finite probabilities in [0, 1]")
    return y_true, probability


def _metric_rows(metrics: dict[str, Any]) -> str:
    labels = {
        "auroc": "AUROC",
        "average_precision": "Average precision (AP; not trapezoidal PR AUC)",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "sensitivity": "Sensitivity (STEMI)",
        "specificity": "Specificity (NSTEMI)",
        "precision": "Precision (STEMI)",
        "f1": "F1 (STEMI)",
        "brier_score": "Brier score",
    }
    return "\n".join(
        f"| {labels[key]} | {float(metrics[key]):.3f} |"
        for key in labels
        if key in metrics
    )


def _evaluation_figure(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> bytes:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    RocCurveDisplay.from_predictions(y_true, probability, name="STEMI positive", ax=axes[0])
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.6)
    axes[0].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_true, probability, name="STEMI positive", ax=axes[1])
    axes[1].axhline(np.mean(y_true), linestyle='--', color='gray', alpha=0.6)
    axes[1].set_title("Precision-recall curve")
    predictions = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    image = axes[2].imshow(matrix, cmap="Blues")
    axes[2].set(
        title=f"Confusion matrix (threshold {threshold:.3f})",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["NSTEMI", "STEMI"],
        yticklabels=["NSTEMI", "STEMI"],
    )
    for row in range(2):
        for column in range(2):
            axes[2].text(column, row, str(matrix[row, column]), ha="center", va="center",
                         color='white' if matrix[row, column] > matrix.max() / 2 else 'black')
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=160)
    plt.close(figure)
    return output.getvalue()


def _markdown(result: dict[str, Any]) -> str:
    test_metrics = result["test_metrics"]
    dummy_metrics = result["dummy_metrics"]
    selection_rows = "\n".join(
        f"| {entry['C']} | {float(entry['validation_auroc']):.3f} |"
        for entry in result["selection"]
    )
    split_rows = "\n".join(
        f"| {name} | {values['n_patients']} | {values['n_stemi']} | {values['n_nstemi']} |"
        for name, values in result["split_counts"].items()
    )
    interval_rows = "\n".join(
        f"| {name} | [{float(interval[0]):.3f}, {float(interval[1]):.3f}] |"
        for name, interval in result["confidence_intervals"].items()
    )
    return f"""# {result['title']}

![Local held-out evaluation](evaluation.png)

## Scope and limitations

This experiment uses one ECG per patient and evaluates a local held-out test only. Hidden publisher labels were not evaluated.
The source population was selected for angiography;
these results do not describe unrestricted screening performance. This is ECG-only STEMI
versus NSTEMI subtype classification, not MI detection. The fixed waveform features are
heuristic features, not clinical ST-segment measurements.

## Cohort and split

```json
{json.dumps(result['cohort_audit'], indent=2, sort_keys=True)}
```

| Split | Patients | STEMI | NSTEMI |
| --- | ---: | ---: | ---: |
{split_rows}

## Model selection

Feature count: {result['feature_count']}. Selected regularization C: {result['selected_C']}.
The validation-selected decision threshold was {float(result['threshold']):.3f}.

| C | Validation AUROC |
| ---: | ---: |
{selection_rows}

## Local held-out test metrics

| Metric | Model |
| --- | ---: |
{_metric_rows(test_metrics)}

## Dummy baseline metrics

The dummy predicts the training STEMI prevalence for every patient and uses a fixed
threshold of 0.5; the trained model uses its validation-selected threshold.

| Metric | Dummy baseline |
| --- | ---: |
{_metric_rows(dummy_metrics)}

## Patient-bootstrap confidence intervals

Intervals are patient-bootstrap intervals with the model fixed; they quantify sampling
uncertainty on this local test cohort and do not provide external validation.

| Metric | 95% interval |
| --- | --- |
{interval_rows}

## Reproducibility

```json
{json.dumps({'config': result['config'], 'provenance': result['provenance']}, indent=2, sort_keys=True)}
```
"""


def write_report(result: dict[str, Any], report_dir: Path) -> None:
    """Write Markdown, standalone HTML, and a three-panel local-evaluation figure."""
    y_true, probability = _curve_arrays(result["test_curve"])
    threshold = float(result["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    report_dir.mkdir(parents=True, exist_ok=True)
    image_bytes = _evaluation_figure(y_true, probability, threshold)
    (report_dir / "evaluation.png").write_bytes(image_bytes)
    markdown = _markdown(result)
    (report_dir / "report.md").write_text(markdown, encoding="utf-8")

    embedded_image = base64.b64encode(image_bytes).decode("ascii")
    safe_title = html.escape(str(result["title"]))
    safe_provenance = html.escape(json.dumps(result["provenance"], indent=2, sort_keys=True))
    visible_markdown = markdown.replace('![Local held-out evaluation](evaluation.png)', '')
    body = markdown_renderer.markdown(html.escape(visible_markdown, quote=False),
                                      extensions=['tables', 'fenced_code'])
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{safe_title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;line-height:1.6;margin:2rem auto;padding:0 1.5rem;max-width:72rem;color:#172b3a;background:#fafcfd}}
h1,h2{{line-height:1.2;color:#123e54}}h2{{margin-top:2rem;border-bottom:1px solid #d5e1e6;padding-bottom:.5rem}}
img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f5;padding:1rem;font-size:.85rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0;background:white}}th,td{{border-bottom:1px solid #d5e1e6;padding:.6rem .8rem;text-align:left}}th{{background:#e8f0f4}}
</style></head><body><h1>{safe_title}</h1><img alt="ROC, precision-recall, and confusion matrix; STEMI positive" src="data:image/png;base64,{embedded_image}">
<p>STEMI positive. ROC x-axis: False positive rate. Precision-recall y-axis: Precision.</p>
{body}<details><summary>Source provenance (plain JSON)</summary><pre>{safe_provenance}</pre></details></body></html>"""
    (report_dir / "report.html").write_text(document, encoding="utf-8")
