"""Train a 1D CNN on the original ACS STEMI/NSTEMI patient split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mi_lab.evaluation import binary_metrics, choose_threshold
from mi_lab.signals import read_ecg


class ECGCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(signal)).squeeze(1)


def prepare_signal(signal: np.ndarray) -> np.ndarray:
    """Per-record filtering only; no training or test statistics are shared."""
    centered = signal - np.median(signal, axis=0)
    sos = butter(3, [0.5, 40], fs=500, btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, centered, axis=0)
    low = resample_poly(filtered, 1, 5, axis=0)
    return np.clip(low.T, -5, 5).astype(np.float32)


def load_signals(manifest: pd.DataFrame, data_dir: Path, cache: Path) -> np.ndarray:
    if cache.exists():
        X = np.load(cache, mmap_mode="r")
        if X.shape != (len(manifest), 12, 1000):
            raise ValueError("Cached CNN signals do not match the patient manifest")
        return X
    cache.parent.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(cache, mode="w+", dtype="float32",
                                  shape=(len(manifest), 12, 1000))
    for i, record_name in enumerate(manifest.ecg_row_record):
        X[i] = prepare_signal(read_ecg(data_dir / "raw" / "row_data" / record_name))
        if (i + 1) % 250 == 0:
            print(f"Prepared {i + 1}/{len(manifest)} ECGs", flush=True)
    X.flush()
    return X


def probabilities(model: ECGCNN, X: np.ndarray, indices: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = torch.from_numpy(np.array(X[indices[start:start + batch_size]]))
            output.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(output)


def run(data_dir: Path, output_dir: Path, epochs: int = 15, batch_size: int = 64,
        seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(data_dir.parent / "artifacts" / "baseline" / "split_manifest.csv",
                           dtype={"Patient_id": str})
    if len(manifest) != 2618 or manifest.Patient_id.duplicated().any():
        raise ValueError("Expected the original 2,618 distinct ACS patients")
    X = load_signals(manifest, data_dir, output_dir / "signals.npy")
    y = manifest.target.to_numpy(dtype=np.float32)
    split = manifest.split.to_numpy()
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "test")
    if (len(train), len(validation), len(test)) != (1570, 524, 524):
        raise ValueError("Original split counts changed")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.array(X[train])), torch.from_numpy(y[train])),
        batch_size=batch_size, shuffle=True,
    )
    model = ECGCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_epoch, best_state = -1.0, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        p_val = probabilities(model, X, validation, batch_size)
        auc = float(roc_auc_score(y[validation], p_val))
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "validation_auroc": auc})
        print(f"Epoch {epoch}/{epochs}: loss {np.mean(losses):.4f}; val AUROC {auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    p_val = probabilities(model, X, validation, batch_size)
    threshold = choose_threshold(y[validation], p_val)
    p_test = probabilities(model, X, test, batch_size)
    metrics = binary_metrics(y[test], p_test, threshold)
    result = {
        "title": "1D CNN STEMI versus NSTEMI on ACS ECG",
        "dataset": "Du et al. ACS ECG dataset, Figshare 29925314 v1",
        "positive_class": "STEMI",
        "same_patient_split_as_logistic_baseline": True,
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "model": "12-lead 1D CNN, convolution channels 16/32/64, global average pooling",
        "optimizer": "Adam", "learning_rate": 0.001,
        "epochs_run": epochs, "batch_size": batch_size,
        "selected_epoch": best_epoch, "validation_auroc": best_auc,
        "validation_threshold": float(threshold), "history": history,
        "test_metrics": metrics,
        "limitations": "Internal held-out test only; no external hospital validation. The CNN is a research baseline, not a clinical device.",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({"Patient_id": manifest.loc[test, "Patient_id"].to_numpy(),
                  "truth_stemi": y[test].astype(int), "probability_stemi": p_test,
                  "predicted_stemi": (p_test >= threshold).astype(int)}).to_csv(
                      output_dir / "test_predictions.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "architecture": result["model"]},
               output_dir / "model.pt")
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.asarray(metrics["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NSTEMI", "STEMI"],
           yticklabels=["NSTEMI", "STEMI"], xlabel="Predicted", ylabel="Actual",
           title="CNN held-out test confusion matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cnn"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir, args.epochs, args.batch_size)
    print(json.dumps(result["test_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
