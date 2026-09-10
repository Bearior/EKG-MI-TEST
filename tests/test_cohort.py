import pandas as pd
import pytest

from mi_lab.cohort import build_cohort, split_cohort


def _source(rows: list[tuple[object, str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Patient_id", "ecg_row_record", "STEMI", "NSTEMI"])


def test_build_cohort_excludes_neither_and_entire_conflicting_patients() -> None:
    frame = _source([
        ("a", "z-record", 1, 0), ("a", "a-record", 1, 0),
        ("b", "both", 1, 1), ("b", "otherwise-eligible", 1, 0),
        ("c", "stemi", 1, 0), ("c", "nstemi", 0, 1),
        ("d", "neither", 0, 0), ("e", "nstemi", 0, 1),
    ])

    cohort, audit = build_cohort(frame)

    assert cohort["Patient_id"].tolist() == ["a", "e"]
    assert cohort["ecg_row_record"].tolist() == ["a-record", "nstemi"]
    assert cohort["target"].tolist() == [1, 0]
    assert audit == {
        "input_records": 8, "excluded_neither_records": 1,
        "excluded_both_label_records": 1, "excluded_conflict_patients": 2,
        "eligible_records": 6, "cohort_records": 2, "cohort_patients": 2,
        "stemi_records": 1, "nstemi_records": 1,
    }


@pytest.mark.parametrize(
    "frame, message",
    [
        (pd.DataFrame({"Patient_id": ["x"]}), "missing required columns"),
        (_source([(None, "r", 1, 0)]), "Patient_id contains missing"),
        (_source([("x", "r", 2, 0)]), "STEMI must contain only binary"),
        (_source([("x", "r", 1, None)]), "NSTEMI must contain only binary"),
    ],
)
def test_build_cohort_rejects_invalid_source(frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_cohort(frame)


def test_split_cohort_is_deterministic_disjoint_and_stratified() -> None:
    rows = []
    for target, label in ((1, "stemi"), (0, "nstemi")):
        for number in range(10):
            rows.append((f"{label}-{number}", f"{label}-record-{number}", target, 1 - target))
    cohort, _ = build_cohort(_source(rows))

    first = split_cohort(cohort)
    second = split_cohort(cohort)

    assert first.equals(second)
    assert first["Patient_id"].is_unique
    assert first["ecg_row_record"].is_unique
    assert first["split"].value_counts().to_dict() == {"train": 12, "validation": 4, "test": 4}
    assert (first.groupby("split")["target"].nunique() == 2).all()


def test_split_cohort_rejects_duplicate_patient_or_record_and_insufficient_classes() -> None:
    cohort = pd.DataFrame({
        "Patient_id": ["a", "a", "c", "d", "e", "f", "g", "h", "i", "j"],
        "ecg_row_record": [f"r{number}" for number in range(10)],
        "target": [0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    })
    with pytest.raises(ValueError, match="one ECG per patient"):
        split_cohort(cohort)

    unique_patients = cohort.assign(Patient_id=[f"p{number}" for number in range(10)])
    unique_patients.loc[1, "ecg_row_record"] = "r0"
    with pytest.raises(ValueError, match="duplicate ecg_row_record"):
        split_cohort(unique_patients)

    too_small = unique_patients.assign(ecg_row_record=[f"u{number}" for number in range(10)])
    too_small.loc[:5, "target"] = 0
    too_small.loc[6:, "target"] = 1
    with pytest.raises(ValueError, match="at least 5"):
        split_cohort(too_small)
