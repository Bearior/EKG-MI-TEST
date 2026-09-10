# MI lab implementation plan

Goal: deliver and run one reproducible local STEMI/NSTEMI experiment.
Spec: design.md. Python 3.10+, NumPy, SciPy, pandas, scikit-learn, Matplotlib.

- [x] Download module: test checksum failure and ZIP traversal rejection, implement
  Figshare version manifest + verified caching + extraction, fetch raw files.
- [x] Cohort module: test label exclusion/conflicts and patient split independence,
  implement one-record-per-patient selection and stratified seed-42 splits.
- [x] Signal module: test decoding and fixed feature outputs against synthetic waveforms,
  implement the publisher's raw format and waveform-only feature extraction.
- [x] Experiment: test metric calculations and a synthetic complete run, implement
  validation model selection, frozen threshold, test evaluation and bootstrap intervals.
- [x] Deliver reports, environment lock, README and commands. Execute real data pipeline,
  inspect plots, run tests/lint/build, review independently and commit the verified project.

Ownership: data agent owns download.py/test_download.py/data downloads; cohort agent owns
cohort.py/test_cohort.py; leader owns all remaining implementation and integration.
