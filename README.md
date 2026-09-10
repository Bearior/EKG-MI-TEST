# MI lab: STEMI versus NSTEMI from ECG

A reproducible, CPU-friendly research baseline using the **Du et al. ACS ECG dataset**,
not PTB-XL. The experiment predicts **STEMI (1) versus NSTEMI (0)** among labelled MI
patients. It does **not** detect MI in the general population or provide clinical advice.

## Run on Windows

Python 3.10 or newer is required. From this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m mi_lab.download --data-dir data
.\.venv\Scripts\python.exe -m mi_lab.experiment
```

Or run `./run.ps1`, which performs these steps and stops if a command fails. Allow several
GB of disk space for archives, extracted ECGs, environment and results. The raw archive
is approximately 1.28 GB; downloads are public and do not need credentials.

After the first run, verified downloads are reused. The experiment itself is recomputed
from the raw data; rerunning writes to the same default output locations. For a distinct
run, supply `--output-dir artifacts/my-run --report-dir reports/my-run`.

## Read the result

- `reports/baseline/report.html`: standalone report with embedded evaluation plots.
- `reports/baseline/report.md`: Markdown report.
- `reports/baseline/summary.json`: aggregate metrics, configuration and source provenance.
- `artifacts/baseline/split_manifest.csv`: exact patient/record assignments.
- `artifacts/baseline/test_predictions.csv`: held-out probabilities and decisions.
- `artifacts/baseline/model.joblib`: fitted pipeline, threshold and feature order.
- `artifacts/baseline/metrics.json`: complete machine-readable results.
- `artifacts/baseline/features.npz`: waveform-only features.
- `artifacts/baseline/signal_exclusions.json`: records excluded by signal checks.

The report is generated from an actual run. Test scores are not hardcoded or promised.
Only load joblib files you trust; the saved model is intended for local research.

## What the experiment does

1. Download and verify Figshare version 1 and save source provenance. Every experiment
   verifies both archives against pinned SHA-256 hashes, then checks consumed metadata
   and waveforms against archive CRCs. Altered or unverified files stop the run.
2. Keep rows with exactly one STEMI/NSTEMI label. Exclude conflicting patient labels.
3. Select the lexicographically first eligible ECG filename per patient. This is a
   deterministic convenience sample, **not necessarily the first presenting ECG**.
4. Read the twelve standard leads in a fixed order, check 500 Hz/10 seconds, convert
   physical units to mV, and exclude non-finite or flat-lead signals.
5. Exclude all selected records belonging to any exact duplicate waveform group.
   This does not establish that near-duplicates or misidentified patients are absent.
6. Stratify patients into 60% training, 20% validation and 20% local testing (seed 42).
7. Extract 664 fixed ECG features: amplitude quantiles, RMS, derivative RMS, spectral
   fractions, heuristic QRS-aligned median morphology and rhythm summaries. Filtering
   (0.5–40 Hz) and 100 Hz resampling are per-record, with no learned preprocessing.
8. Fit training-only standardization and class-balanced logistic regression. Select
   `C` from `[0.01, 0.1, 1.0]` using validation AUROC; ties keep the smaller `C`.
9. Choose the validation threshold maximizing Youden J (equivalent to balanced accuracy),
   then evaluate the test set. Do not refit on validation after choosing the threshold.
10. Compare with a training-prevalence dummy classifier; report AUROC, average precision
    (AP), sensitivity, specificity, balanced accuracy, F1, Brier score, confusion matrix,
    and 1,000 patient-bootstrap percentile confidence intervals for key metrics.

**No diagnosis, report, angiography, treatment, timing, age or sex fields enter the model.**
All feature names are explicit in the saved artifact. Median-beat alignment is a fixed
heuristic, not a validated ST-segment measurement algorithm. The class-balanced model's
probabilities are not clinically calibrated; inspect the reported Brier score accordingly.

## Dataset and study limits

- Dataset: https://doi.org/10.6084/m9.figshare.29925314 (Figshare **CC0**).
- Paper: https://doi.org/10.1038/s41597-026-07278-0 .
- Publisher decoding reference: https://github.com/catcatgirl/validation-code .
- The overall release advertises 19,955 ECGs from 18,909 patients. The downloadable
  labelled training CSV has 17,960 ECGs, with 1,442 STEMI and 1,235 NSTEMI labels before
  this experiment's patient and signal exclusions. Published text/table counts differ;
  the downloaded, checksummed files are the authority for this run.
- The publisher's 1,995-record test file hides diagnosis labels. It is **not evaluated**
  here. Our local held-out test is internal validation from the released labelled pool.
- Participants underwent angiography. This selected cohort does not establish screening
  performance in all emergency patients. Labels derive from clinical data and expert
  review, and ECG/angiography timing can vary.
- NSTEMI is a clinical diagnosis, not simply “no ST elevation.” Non-MI patients are
  excluded, so the binary result cannot determine whether a new patient has MI.
- Fixed-model bootstrap intervals describe test sampling uncertainty, not variability
  from repeating the entire training process. A small internal baseline is not evidence
  of clinical utility, safety or transportability.

## Verify the code

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m compileall -q src
```

Tests use synthetic fixtures and need no data download. They exercise safe extraction,
checksums, patient isolation, label handling, decoding, metrics and the full training/report
path. `pyproject.toml` describes supported dependency ranges; `requirements.lock` records
the exact environment used for this experiment.

## Repository layout

```text
src/mi_lab/       download, cohort, signal, evaluation, experiment and reporting modules
tests/           synthetic unit and integration tests
docs/            design and implementation record
reports/         aggregate results and readable reports (tracked)
data/            source archives and ECGs (ignored)
artifacts/       models, features and record-level outputs (ignored)
.venv/           isolated Python environment (ignored)
```

Git tracks the experiment and aggregate reports. Data and model artifacts remain local;
no GitHub repository or other remote publication is created by this project.
