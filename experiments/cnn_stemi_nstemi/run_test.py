"""Re-evaluate the saved CNN on the original held-out ACS patients."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mi_lab.cnn_experiment import ECGCNN, load_signals, probabilities
from mi_lab.evaluation import binary_metrics


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    experiment = Path(__file__).resolve().parent
    manifest = pd.read_csv(root / "artifacts/baseline/split_manifest.csv", dtype={"Patient_id": str})
    if len(manifest) != 2618 or manifest.Patient_id.duplicated().any():
        raise ValueError("The saved baseline patient split is missing or changed")
    X = load_signals(manifest, root / "data", root / "artifacts/cnn/signals.npy")
    test = np.flatnonzero(manifest.split.to_numpy() == "test")
    if len(test) != 524:
        raise ValueError("Expected the original 524-patient test set")
    saved = torch.load(experiment / "model.pt", map_location="cpu", weights_only=True)
    model = ECGCNN()
    model.load_state_dict(saved["state_dict"])
    summary = json.loads((experiment / "summary.json").read_text(encoding="utf-8"))
    scores = probabilities(model, X, test, batch_size=64)
    metrics = binary_metrics(manifest.target.to_numpy()[test], scores,
                             summary["validation_threshold"])
    for key in ("auroc", "balanced_accuracy", "sensitivity", "specificity"):
        if not np.isclose(metrics[key], summary["test_metrics"][key], atol=1e-6):
            raise AssertionError(f"Stored and recomputed {key} differ")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
