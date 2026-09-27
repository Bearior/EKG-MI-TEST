# CNN STEMI vs NSTEMI experiment

This folder contains the actual 1D CNN run on the same 2,618 patients and 60/20/20 patient split as the previous L2 logistic regression experiment. It uses the [Du et al. ACS ECG dataset](https://doi.org/10.6084/m9.figshare.29925314). The task is STEMI versus NSTEMI among patients with MI.

Files:

- `run.ps1`: train and evaluate the CNN from the repository root and prepared ECG data.
- `run_test.py`: load the saved CNN and recompute metrics on the original held-out 524 patients. Run with `.\.venv\Scripts\python.exe experiments\cnn_stemi_nstemi\run_test.py` from the repository root.
- `model.pt`: trained CNN weights. The loader uses PyTorch `weights_only=True`.
- `summary.json`: exact configuration, epoch history and aggregate test results.
- `report.md`: readable explanation and comparison with the previous model.
- `evaluation.png`: ROC curve, precision–recall curve and confusion matrix from test predictions.
- `progress_slides.pptx`: five editable summary slides. Only the result plots are an image.

First set up the environment and download the dataset using the repository's main README. Install CPU PyTorch with `.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu`. Raw ECGs and the prepared signal cache are intentionally excluded from Git. This is an internal research test, not external validation or a clinical model.
