from __future__ import annotations

from pathlib import Path

import pandas as pd


DIAGNOSIS_COLUMNS = [
    "subject_id",
    "hadm_id",
    "seq_num",
    "icd_code",
    "icd_version",
    "long_title",
]

PRESCRIPTION_COLUMNS = [
    "subject_id",
    "hadm_id",
    "starttime",
    "stoptime",
    "drug",
    "drug_type",
    "dose_val_rx",
    "dose_unit_rx",
    "route",
]


def _read_csv(
    path: str | Path,
    required_columns: list[str],
) -> pd.DataFrame:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        dtype={
            "icd_code": "string",
            "drug": "string",
            "dose_val_rx": "string",
            "dose_unit_rx": "string",
            "route": "string",
        },
    )

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {sorted(missing)}"
        )

    return df


def _optional_string(value) -> str | None:
    if pd.isna(value):
        return None

    return str(value)


def _optional_timestamp(value) -> str | None:
    if pd.isna(value):
        return None

    return pd.Timestamp(value).isoformat(
        sep=" ",
        timespec="seconds",
    )


def load_diagnoses(
    path: str | Path,
) -> pd.DataFrame:
    df = _read_csv(
        path,
        DIAGNOSIS_COLUMNS,
    )

    return df.sort_values(
        ["subject_id", "hadm_id", "seq_num"]
    ).reset_index(drop=True)


def load_prescriptions(
    path: str | Path,
) -> pd.DataFrame:
    df = _read_csv(
        path,
        PRESCRIPTION_COLUMNS,
    )

    df["starttime"] = pd.to_datetime(
        df["starttime"],
        errors="raise",
    )

    df["stoptime"] = pd.to_datetime(
        df["stoptime"],
        errors="raise",
    )

    return df.sort_values(
        ["subject_id", "hadm_id", "starttime"],
    ).reset_index(drop=True)


def build_clinical_record(
    *,
    key: tuple[int, int],
    patient: dict,
    diagnoses_df: pd.DataFrame,
    prescriptions_df: pd.DataFrame,
) -> dict:

    patient_id, admission_id = key

    diagnoses = diagnoses_df.loc[
        (diagnoses_df["subject_id"] == patient_id)
        & (diagnoses_df["hadm_id"] == admission_id)
    ]

    prescriptions = prescriptions_df.loc[
        (prescriptions_df["subject_id"] == patient_id)
        & (prescriptions_df["hadm_id"] == admission_id)
    ]

    diagnosis_records = []

    for row in diagnoses.itertuples(index=False):
        diagnosis_records.append(
            {
                "code": str(row.icd_code),
                "version": int(row.icd_version),
                "description": _optional_string(
                    row.long_title
                ),
                "sequence": int(row.seq_num),

                # Admission-level billing diagnoses
                # have no reliable availability timestamp.
                "available_at_cutoff": False,
            }
        )

    medication_records = []

    for row in prescriptions.itertuples(index=False):
        medication_records.append(
            {
                "drug": _optional_string(row.drug),
                "drug_type": _optional_string(
                    row.drug_type
                ),
                "starttime": _optional_timestamp(
                    row.starttime
                ),
                "stoptime": _optional_timestamp(
                    row.stoptime
                ),
                "dose": _optional_string(
                    row.dose_val_rx
                ),
                "dose_unit": _optional_string(
                    row.dose_unit_rx
                ),
                "route": _optional_string(
                    row.route
                ),

                # These are orders, not confirmed
                # medication administrations.
                "event_type": "prescription",

                # Prescription starttime alone does not
                # establish when the order was known.
                "available_at_cutoff": False,
            }
        )

    return {
        "patient_id": patient_id,
        "admission_id": admission_id,

        "demographics": {
            "age": patient["row"]["Age"],
            "sex": patient["row"]["Sex"],
        },

        "measurements": {
            "row": patient["row"],
            "day_cols": patient["day_cols"],
            "record_times": patient["record_times"],
        },

        "diagnoses": diagnosis_records,
        "medications": medication_records,

        "label": patient["label"],
    }
