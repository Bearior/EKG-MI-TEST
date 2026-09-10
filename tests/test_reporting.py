from pathlib import Path

from mi_lab.reporting import write_report


def _result() -> dict:
    metrics = {
        "auroc": 0.8,
        "average_precision": 0.75,
        "balanced_accuracy": 0.7,
        "sensitivity": 0.6,
        "specificity": 0.8,
        "confusion_matrix": [[8, 2], [4, 6]],
    }
    return {
        "title": "<STEMI> report",
        "cohort_audit": {"cohort_patients": 20, "stemi_records": 10, "nstemi_records": 10},
        "split_counts": {
            "train": {"n_patients": 12, "n_stemi": 6, "n_nstemi": 6},
            "validation": {"n_patients": 4, "n_stemi": 2, "n_nstemi": 2},
            "test": {"n_patients": 4, "n_stemi": 2, "n_nstemi": 2},
        },
        "feature_count": 42,
        "selection": [{"C": 0.1, "validation_auroc": 0.7}, {"C": 1.0, "validation_auroc": 0.8}],
        "selected_C": 1.0,
        "threshold": 0.4,
        "test_metrics": metrics,
        "dummy_metrics": {**metrics, "auroc": 0.5},
        "confidence_intervals": {"auroc": [0.6, 0.9]},
        "config": {"seed": 42, "n_bootstrap": 100},
        "provenance": {"source": "<publisher & source>"},
        "test_curve": {"y_true": [0, 0, 1, 1], "probability": [0.1, 0.6, 0.7, 0.9]},
    }


def test_write_report_creates_standalone_artifacts_with_required_interpretation(tmp_path: Path) -> None:
    write_report(_result(), tmp_path)

    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    image = tmp_path / "evaluation.png"
    assert image.exists() and image.stat().st_size > 0
    assert "evaluation.png" in markdown
    assert "one ECG per patient" in markdown
    assert "local held-out test only" in markdown
    assert "Hidden publisher labels were not evaluated" in markdown
    assert "Average precision (AP; not trapezoidal PR AUC)" in markdown
    assert "patient-bootstrap intervals with the model fixed" in markdown
    assert "## Dummy baseline metrics" in markdown
    assert "data:image/png;base64," in html
    assert "&lt;STEMI&gt; report" in html
    assert "&lt;publisher &amp; source&gt;" in html
    assert "<STEMI> report" not in html
    assert '<table>' in html
    assert '<h2>Local held-out test metrics</h2>' in html


def test_write_report_labels_stemi_as_positive_in_plot(tmp_path: Path) -> None:
    write_report(_result(), tmp_path)

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "STEMI positive" in html
    assert "False positive rate" in html
    assert "Precision" in html
