from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from clinical_llm_conformal.preprocessing.clinicrealm import (
    run_exact_clinicrealm_preprocessing,
)

CONFIG_PATH = Path("configs/benchmark.yaml")

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
            "MIMIC_DATA_DIR is missing required files:\n"
            f"{formatted}"
        )


def main() -> None:
    load_dotenv()

    mimic_data_dir = os.environ.get(
        "MIMIC_DATA_DIR"
    )

    if not mimic_data_dir:
        raise RuntimeError(
            "MIMIC_DATA_DIR is not set in .env"
        )

    data_root = (
        Path(mimic_data_dir)
        .expanduser()
        .resolve()
    )

    validate_mimic_directory(
        data_root
    )

    config = yaml.safe_load(
        CONFIG_PATH.read_text()
    )

    work_root = Path(
        config["data"]["work_root"]
    )

    processed_dir = (
        work_root / "processed"
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "MIMIC-IV data directory:",
        data_root,
    )

    print(
        "\nRunning pinned upstream preprocessing..."
    )

    ehr = (
        run_exact_clinicrealm_preprocessing(
            raw_dir=data_root,
            processed_dir=processed_dir,
            max_window=7,
        )
    )

    if not isinstance(
        ehr,
        pd.DataFrame,
    ):
        raise TypeError(
            "Expected preprocessing to return "
            f"a DataFrame, got {type(ehr)!r}"
        )

    print("\nFinished.")
    print("EHR shape:", ehr.shape)

    print(
        "Admissions:",
        ehr[
            ["PatientID", "AdmissionID"]
        ]
        .drop_duplicates()
        .shape[0],
    )

    output_path = (
        processed_dir
        / "mimic4_formatted_ehr.parquet"
    )

    print(
        "Expected upstream output:",
        output_path,
    )


if __name__ == "__main__":
    main()