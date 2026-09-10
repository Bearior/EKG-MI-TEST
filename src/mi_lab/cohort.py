"""Patient-level STEMI versus NSTEMI cohort construction and splitting."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = ("Patient_id", "ecg_row_record", "STEMI", "NSTEMI")
SPLIT_COLUMN = "split"


def _validate_source(frame: pd.DataFrame) -> None:
    """Validate the small, explicit source-metadata contract."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"source frame is missing required columns: {missing}")
    if frame["Patient_id"].isna().any():
        raise ValueError("Patient_id contains missing values")
    if frame["ecg_row_record"].isna().any():
        raise ValueError("ecg_row_record contains missing values")
    for column in ("STEMI", "NSTEMI"):
        if not frame[column].isin((0, 1)).all():
            raise ValueError(f"{column} must contain only binary 0/1 labels")


def build_cohort(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build one deterministic, labelled ECG per consistent patient.

    A row with neither label is ignored. A row with both labels makes its full
    patient ineligible, as does a patient with STEMI and NSTEMI among separate
    eligible rows. The alphabetically first eligible ``ecg_row_record`` is kept
    for each remaining patient. The returned target is STEMI=1, NSTEMI=0.

    Audit counts are record counts except ``excluded_conflict_patients`` and
    the ``cohort_*``/class counts, which are patient/selected-record counts.
    """
    _validate_source(frame)
    working = frame.copy()
    working["target"] = working["STEMI"].astype(int)
    neither = (working["STEMI"] == 0) & (working["NSTEMI"] == 0)
    both = (working["STEMI"] == 1) & (working["NSTEMI"] == 1)
    eligible = working.loc[~neither & ~both].copy()

    both_patients = set(working.loc[both, "Patient_id"])
    label_counts = eligible.groupby("Patient_id", sort=False)["target"].nunique()
    conflicting_patients = set(label_counts[label_counts > 1].index)
    excluded_patients = both_patients | conflicting_patients
    retained = eligible.loc[~eligible["Patient_id"].isin(excluded_patients)].copy()
    retained = retained.sort_values(
        ["Patient_id", "ecg_row_record"], kind="stable"
    ).drop_duplicates("Patient_id", keep="first")
    cohort = retained.sort_values("Patient_id", kind="stable").reset_index(drop=True)

    audit = {
        "input_records": len(working),
        "excluded_neither_records": int(neither.sum()),
        "excluded_both_label_records": int(both.sum()),
        "excluded_conflict_patients": len(excluded_patients),
        "eligible_records": len(eligible),
        "cohort_records": len(cohort),
        "cohort_patients": int(cohort["Patient_id"].nunique()),
        "stemi_records": int((cohort["target"] == 1).sum()),
        "nstemi_records": int((cohort["target"] == 0).sum()),
    }
    return cohort, audit


def _validate_cohort_for_split(cohort: pd.DataFrame) -> None:
    required = ("Patient_id", "ecg_row_record", "target")
    missing = [column for column in required if column not in cohort.columns]
    if missing:
        raise ValueError(f"cohort is missing required columns: {missing}")
    if cohort.empty:
        raise ValueError("cohort is empty")
    if cohort["Patient_id"].isna().any() or cohort["ecg_row_record"].isna().any():
        raise ValueError("cohort contains missing patient or record identifiers")
    if not cohort["target"].isin((0, 1)).all():
        raise ValueError("target must contain only binary 0/1 labels")
    if not cohort["Patient_id"].is_unique:
        raise ValueError("cohort must contain exactly one ECG per patient")
    if not cohort["ecg_row_record"].is_unique:
        raise ValueError("cohort contains duplicate ecg_row_record values")
    counts = cohort["target"].value_counts()
    if set(counts.index) != {0, 1} or counts.min() < 5:
        raise ValueError("each class needs at least 5 records for 60/20/20 stratified splits")


def split_cohort(cohort: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add deterministic stratified 60/20/20 train/validation/test labels.

    The function requires the one-row-per-patient cohort returned by
    :func:`build_cohort`, so patient leakage is impossible by construction.
    """
    _validate_cohort_for_split(cohort)
    indices = cohort.index
    _train_indices, held_out_indices = train_test_split(
        indices, test_size=0.4, random_state=seed, stratify=cohort["target"]
    )
    held_out = cohort.loc[held_out_indices]
    validation_indices, test_indices = train_test_split(
        held_out_indices,
        test_size=0.5,
        random_state=seed,
        stratify=held_out["target"],
    )
    result = cohort.copy()
    result[SPLIT_COLUMN] = "train"
    result.loc[validation_indices, SPLIT_COLUMN] = "validation"
    result.loc[test_indices, SPLIT_COLUMN] = "test"

    partition_classes = result.groupby(SPLIT_COLUMN)["target"].nunique()
    if set(partition_classes.index) != {"train", "validation", "test"} or (
        partition_classes != 2
    ).any():
        raise ValueError("stratification did not place both classes in every split")
    return result
