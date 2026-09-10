# Baseline experiment record

## Protocol

The prespecified question is ECG-only STEMI versus NSTEMI classification among
labelled MI patients from Figshare article 29925314, version 1. See design.md and
README.md for the cohort and feature rules. Hyperparameters and decision threshold
are chosen on validation data, never local test data. No clinical metadata is used
as a predictor. The publisher's hidden-label test set is not evaluated.

The first run used 664 fixed features and three logistic-regression regularization
candidates (0.01, 0.1, 1.0). The model is not tuned again after viewing test results.
A second run checks reproducibility after adding stricter source-integrity checks
and improving report presentation. It does not change the model or feature protocol.

## Cohort audit

The labelled source contains 17,960 records; 2,677 have an eligible MI-subtype label.
Thirteen patients have conflicting subtype records and are excluded. Selecting one
eligible ECG per remaining patient gives 2,625 patients. Seven selected ECGs have a
flat lead and are excluded, leaving **2,618**. No exact duplicate selected waveforms
were found. Exclusion counts and source identifiers remain in ignored local artifacts.

| Partition | Patients | STEMI | NSTEMI |
| --- | ---: | ---: | ---: |
| Train | 1,570 | 852 | 718 |
| Validation | 524 | 284 | 240 |
| Local test | 524 | 284 | 240 |

## First-run results

| Measure | Estimate |
| --- | ---: |
| AUROC | 0.784 |
| AUROC 95% patient-bootstrap interval | 0.744–0.820 |
| Average precision | 0.807 |
| Balanced accuracy | 0.709 |
| STEMI sensitivity | 0.563 |
| NSTEMI specificity | 0.854 |

The validation-selected threshold is approximately 0.612. The test confusion matrix
is [[205, 35], [124, 160]], with true rows and predicted columns ordered NSTEMI, STEMI.
The 124 missed STEMI cases out of 284 show a substantial sensitivity limitation at
this threshold. These results justify further research, not clinical use.

## Verification record

Focused tests were run before adding each core module. The integrated suite covers
cohort handling, label conflicts, patient split isolation, raw waveform decoding,
feature invariants, metric examples, bootstrapping, source download checks and a
complete synthetic training/report run. The scaler test shifts a held-out feature
distribution and verifies that the fitted scaler still uses training data only.

Independent review found no patient leakage or test-based model selection. It found
an optional-provenance path, now removed: direct runs require the pinned v1 manifest,
validate archive SHA-256 identities and verify each consumed file against its archive
CRC. The report now explains that the dummy classifier uses a fixed threshold of 0.5.

Final checks: **29 tests passed**, lint and compilation passed, dependency consistency
passed, and the code package built successfully. The complete `run.ps1` path exited
successfully. Its test metrics, bootstrap intervals, regularization selection, threshold
and split counts matched the first run exactly. Selected C was 0.01. The evaluation
figure was visually inspected. Full precision results are retained with the delivered
report and Git commit.
