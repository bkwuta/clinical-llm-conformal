from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[3]

MIMIC_PREPROCESSOR_ROOT = REPO_ROOT / "vendor" / "mimic_preprocessor"

UPSTREAM_MIMIC_ROOT = (
    MIMIC_PREPROCESSOR_ROOT
    / "mimic_datasets"
    / "mimic_iv"
    / "3.1"
)


def _load_mimic_iv_processor():
    root = str(MIMIC_PREPROCESSOR_ROOT)

    if root not in sys.path:
        sys.path.insert(0, root)

    from mimic_preprocessor.mimic_iv_processor import MIMICIVProcessor

    return MIMICIVProcessor


def prepare_upstream_workspace(
    raw_dir: str | Path,
) -> Path:
    raw_dir = Path(raw_dir)

    (raw_dir / "hosp").mkdir(parents=True, exist_ok=True)
    (raw_dir / "icu").mkdir(parents=True, exist_ok=True)

    for filename in ("config.toml", "mimic4_item2var.csv"):
        source = UPSTREAM_MIMIC_ROOT / filename

        if not source.exists():
            raise FileNotFoundError(
                f"Required upstream asset not found: {source}"
            )

        shutil.copy2(source, raw_dir / filename)

    return raw_dir


def run_exact_clinicrealm_preprocessing(
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    max_window: int = 7,
):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    prepare_upstream_workspace(raw_dir)

    required_files = [
        raw_dir / "hosp" / "patients.csv",
        raw_dir / "hosp" / "admissions.csv",
        raw_dir / "icu" / "icustays.csv",
        raw_dir / "icu" / "chartevents.csv",
    ]

    missing = [path for path in required_files if not path.exists()]

    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing required MIMIC input files:\n{formatted}"
        )

    processed_dir.mkdir(parents=True, exist_ok=True)

    note_dir = raw_dir.parent / "unused_mimic_notes"
    note_dir.mkdir(parents=True, exist_ok=True)

    MIMICIVProcessor = _load_mimic_iv_processor()

    processor = MIMICIVProcessor(
        data_dir=str(raw_dir),
        note_dir=str(note_dir),
        processed_dir=str(processed_dir),
        one_hot_encode_categorical=False,
    )

    ehr = processor._process_ehr_data(max_window=max_window)

    return ehr

def restrict_to_clinicrealm_cohort(
    ehr_df: pd.DataFrame,
    discharge_note_admissions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the admission-level cohort restriction induced by
    mimic_preprocessor._merge_ehr_and_note(), without loading
    discharge-note text.

    Original upstream operation:
        note_df.merge(
            ehr_df,
            how="inner",
            on=["PatientID", "AdmissionID"],
        )
    """

    note_keys = (
        discharge_note_admissions[
            ["subject_id", "hadm_id"]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "subject_id": "PatientID",
                "hadm_id": "AdmissionID",
            }
        )
    )

    df = ehr_df.merge(
        note_keys,
        how="inner",
        on=["PatientID", "AdmissionID"],
    )

    df["RecordID"] = (
        df["PatientID"].astype(str)
        + "_"
        + df["AdmissionID"].astype(str)
    )

    return (
        df.sort_values(
            ["RecordID", "RecordTime"]
        )
        .reset_index(drop=True)
    )


def make_clinicrealm_test_split(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    test_size: int = 200,
) -> pd.DataFrame:
    """
    Reproduce lines 35-43 of ClinicRealm's
    my_datasets/mimic-iv/preprocess_mimic.py.
    """

    df = (
        df.sort_values(
            ["RecordID", "RecordTime"]
        )
        .reset_index(drop=True)
    )

    grouped = df.groupby(
        "RecordID",
        sort=True,
    )

    patients = np.array(
        list(grouped.groups.keys())
    )

    patients_outcome = np.array(
        [
            grouped
            .get_group(patient_id)["Outcome"]
            .iloc[0]
            for patient_id in patients
        ]
    )

    _, test_patients = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
        stratify=patients_outcome,
    )

    test_df = df[
        df["RecordID"].isin(test_patients)
    ].copy()

    return (
        test_df.sort_values(
            ["RecordID", "RecordTime"]
        )
        .reset_index(drop=True)
    )