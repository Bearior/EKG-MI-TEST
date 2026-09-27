# 1D CNN experiment on the original ACS ECG cohort

This experiment replaces the handcrafted-feature classifier with a small 1D CNN that reads the twelve ECG waveforms directly. The clinical task remains **STEMI versus NSTEMI among patients with MI**. It does not classify Normal ECG or detect MI in a general population.

Run from the repository root after completing the baseline data preparation:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.2,<3"
.\.venv\Scripts\python.exe -m mi_lab.cnn_experiment --epochs 15 --batch-size 64
```

The CNN reuses `artifacts/baseline/split_manifest.csv`: 1,570 training, 524 validation, and 524 held-out test patients, with one ECG per patient. Each ten-second, twelve-lead recording is median-centered, bandpass filtered at 0.5–40 Hz, resampled from 500 to 100 Hz, and clipped to ±5 mV. This preprocessing uses no statistics from other patients. The signal array is cached as `artifacts/cnn/signals.npy` for a repeat run.

The network has three 1D convolution blocks with 16, 32 and 64 channels, batch normalization, ReLU and pooling, followed by global average pooling and one output logit. Training uses binary cross-entropy, Adam with learning rate 0.001, batch size 64, and 15 epochs. The epoch with the highest validation AUROC is retained; a classification threshold is chosen on validation using Youden J. The held-out test set is evaluated only after those choices.

Outputs: `artifacts/cnn/summary.json`, `test_predictions.csv`, `confusion_matrix.png`, and `model.pt`. This is an internal research comparison using the same test patients as the logistic regression baseline. Neither model is validated for clinical use.

## Actual run (seed 42)

The best validation AUROC was **0.837** at epoch **14**. On the held-out test set (524 patients), the CNN reached AUROC **0.802**, average precision **0.813**, balanced accuracy **0.745**, sensitivity **0.761**, specificity **0.729**, and F1 **0.765**. The validation-selected threshold was **0.512**. The confusion matrix, with actual rows and predicted columns in `[NSTEMI, STEMI]` order, was `[[175, 65], [68, 216]]`.

| Held-out metric | Logistic regression | 1D CNN |
| --- | ---: | ---: |
| AUROC | 0.784 | 0.802 |
| Balanced accuracy | 0.709 | 0.745 |
| STEMI sensitivity | 0.563 | 0.761 |
| NSTEMI specificity | 0.854 | 0.729 |
| F1 | 0.668 | 0.765 |

The CNN detected 216 of 284 STEMI cases, versus 160 for the previous baseline, but called 65 NSTEMI cases STEMI, versus 35 previously. These figures describe one local test set and one training seed; they are not evidence of clinical superiority or transfer to another hospital.
