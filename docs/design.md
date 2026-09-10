# STEMI versus NSTEMI experiment

The user authorized a complete local experiment and Git repository here. Use Du et al.'s
ACS dataset (Figshare 29925314, CC0), not PTB-XL. Target STEMI=1 versus NSTEMI=0
among labelled acute-MI ECGs. Neither label means exclusion, not a healthy control.

Download immutable source metadata and checksum-verified raw ECGs. Audit missing,
contradictory and repeated labels. Use one deterministic ECG per patient for the main
experiment, exclude inconsistent patient labels, and split those patients 60/20/20 with
seed 42 and stratification. Check exact waveform duplicates before splitting. Keep the
publisher's unlabelled test cohort separate; never invent labels or claim external validation.

Use waveform features only, with no diagnosis, angiography, timing, treatment, demographics
or report fields as predictors. Implement a CPU-friendly baseline: fixed signal-distribution
and frequency features, training-only standardization and logistic regression. Select
regularization using validation AUROC, lock a validation balanced-accuracy threshold,
then evaluate the local test once. Include a dummy baseline. This is a starting benchmark,
not an optimized classifier or clinical validation. Patient bootstrap confidence intervals
apply because the primary cohort has one ECG per patient.

Save cohort/split manifests, source provenance, model, probabilities, metrics, plots,
and a readable Markdown/HTML report. Track code/docs/aggregate reports in Git; ignore
downloaded data, record-level outputs, models and the environment. No remote publication.

Key checks: safe downloads/extraction, patient isolation, no predictor leakage, invalid
signals rejected, reproducible feature order, train-only fitting, metrics against known
examples, and an end-to-end synthetic smoke test. Run the actual experiment and all tests.
