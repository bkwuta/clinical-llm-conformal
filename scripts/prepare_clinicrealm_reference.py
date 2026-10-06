from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from clinical_llm_conformal.preprocessing.clinicrealm import (
    make_clinicrealm_test_split,
    run_exact_clinicrealm_preprocessing,
)

REQUIRED_MIMIC_FILES = (
    "hosp/patients.csv",
    "hosp/admissions.csv",
    "icu/icustays.csv",
    "icu/chartevents.csv",
)


def validate_mimic_directory(
    data_root: Path,
) -> None:
    missing = [
        data_root / relative_path
        for relative_path in REQUIRED_MIMIC_FILES
        if not (data_root / relative_path).exists()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "MIMIC_DATA_DIR does not contain the "
            "required MIMIC-IV files:\n"
            f"{formatted}"
        )


def main() -> None:
    load_dotenv()

    mimic_data_dir = os.environ.get(
        "MIMIC_DATA_DIR"
    )

    if not mimic_data_dir:
        raise RuntimeError(
            "MIMIC_DATA_DIR is not set.\n"
            "Add it to .env, for example:\n"
            "MIMIC_DATA_DIR=/data/mimiciv"
        )

    data_root = (
        Path(mimic_data_dir)
        .expanduser()
        .resolve()
    )

    validate_mimic_directory(
        data_root
    )

    config_path = Path(
        "configs/clinicrealm_reference.yaml"
    )

    config = yaml.safe_load(
        config_path.read_text()
    )

    seed = int(
        config.get("seed", 42)
    )

    test_size = int(
        config.get(
            "reference",
            {},
        ).get(
            "test_size",
            200,
        )
    )

    work_root = Path(
        config["data"]["work_root"]
    )

    processed_root = (
        work_root / "processed"
    )

    processed_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "MIMIC-IV data directory:",
        data_root,
    )

    print()
    print(
        "Running pinned PKU-AICare "
        "MIMIC-IV processor..."
    )

    ehr_df = (
        run_exact_clinicrealm_preprocessing(
            raw_dir=data_root,
            processed_dir=processed_root,
            max_window=7,
        )
    )

    if not isinstance(
        ehr_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "Expected MIMIC preprocessing "
            "to return a pandas DataFrame, "
            f"got {type(ehr_df)!r}"
        )

    required_columns = {
        "PatientID",
        "AdmissionID",
        "RecordTime",
        "Outcome",
    }

    missing_columns = (
        required_columns
        - set(ehr_df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Processed EHR is missing "
            "required columns: "
            f"{sorted(missing_columns)}"
        )

    # The original ClinicRealm splitting code
    # operates on one RecordID per admission.
    if "RecordID" not in ehr_df.columns:
        ehr_df = ehr_df.copy()

        ehr_df["RecordID"] = (
            ehr_df["PatientID"].astype(str)
            + "_"
            + ehr_df[
                "AdmissionID"
            ].astype(str)
        )

    ehr_df = (
        ehr_df.sort_values(
            [
                "RecordID",
                "RecordTime",
            ]
        )
        .reset_index(drop=True)
    )

    structured_path = (
        processed_root
        / "clinicrealm_structured_ehr.parquet"
    )

    ehr_df.to_parquet(
        structured_path,
        index=False,
    )

    n_admissions = (
        ehr_df["RecordID"].nunique()
    )

    print(
        "Structured EHR admissions:",
        n_admissions,
    )

    print()
    print(
        "Creating ClinicRealm-style "
        "structured-only test split..."
    )

    test_df = (
        make_clinicrealm_test_split(
            ehr_df,
            seed=seed,
            test_size=test_size,
        )
    )

    test_path = (
        processed_root
        / "clinicrealm_structured_test_200.parquet"
    )

    test_df.to_parquet(
        test_path,
        index=False,
    )

    admission_labels = (
        test_df
        .groupby("RecordID")[
            "Outcome"
        ]
        .first()
    )

    counts = (
        admission_labels
        .value_counts()
        .sort_index()
    )

    n_test = len(
        admission_labels
    )

    alive = int(
        counts.get(0, 0)
    )

    dead = int(
        counts.get(1, 0)
    )

    print()
    print(
        "===== STRUCTURED-ONLY "
        "CLINICREALM-STYLE SPLIT ====="
    )

    print(
        "Admissions:",
        n_test,
    )

    print(
        "Alive:",
        alive,
    )

    print(
        "Dead:",
        dead,
    )

    print(
        "Rows/visits:",
        len(test_df),
    )

    print(
        "Processed EHR:",
        structured_path,
    )

    print(
        "Test set:",
        test_path,
    )

    if n_test != test_size:
        raise RuntimeError(
            "Unexpected test-set size: "
            f"{n_test}; expected {test_size}"
        )

    print()
    print(
        "NOTE: This is a structured-only "
        "ClinicRealm-style split."
    )

    print(
        "It does not reproduce ClinicRealm's "
        "discharge-note availability "
        "restriction because MIMIC-IV-Note "
        "is not available in this setup."
    )


if __name__ == "__main__":
    main()