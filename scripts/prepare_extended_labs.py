from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv


REFERENCE_CONFIG_PATH = Path(
    "configs/clinicrealm_reference.yaml"
)

EXTENDED_CONFIG_PATH = Path(
    "configs/extended_features.yaml"
)

OUTPUT_DIR = Path(
    "tmp/extended_ehr"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "clinicrealm_structured_test_200_plus_labs.parquet"
)


# MIMIC-IV labevents ITEMIDs.
#
# We will verify these against d_labitems rather than
# assuming them blindly.
TARGET_LABELS = {
    "Creatinine",
    "Urea Nitrogen",
    "Lactate",
    "White Blood Cells",
    "Hemoglobin",
    "Platelet Count",
}

DISPLAY_NAMES = {
    "Urea Nitrogen": "BUN",
    "White Blood Cells": "WBC",
    "Platelet Count": "Platelets",
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(
        path.read_text()
    )


def main() -> None:
    load_dotenv()

    mimic_root = Path(
        os.environ["MIMIC_DATA_DIR"]
    )

    reference_config = load_yaml(
        REFERENCE_CONFIG_PATH
    )

    ehr_path = Path(
        reference_config["data"][
            "test_ehr_path"
        ]
    )

    labevents_path = (
        mimic_root
        / "hosp"
        / "labevents.csv"
    )

    d_labitems_path = (
        mimic_root
        / "hosp"
        / "d_labitems.csv"
    )

    if not ehr_path.exists():
        raise FileNotFoundError(
            f"Missing EHR cohort: {ehr_path}"
        )

    if not labevents_path.exists():
        raise FileNotFoundError(
            f"Missing labevents: "
            f"{labevents_path}"
        )

    if not d_labitems_path.exists():
        raise FileNotFoundError(
            f"Missing d_labitems: "
            f"{d_labitems_path}"
        )

    print(
        "Loading frozen ClinicRealm-style cohort..."
    )

    ehr = pd.read_parquet(
        ehr_path
    )

    admission_ids = set(
        ehr["AdmissionID"]
        .astype("int64")
        .unique()
    )

    print(
        "Admissions:",
        len(admission_ids),
    )

    print(
        "\nLoading lab dictionary..."
    )

    d_labitems = pd.read_csv(
        d_labitems_path,
        usecols=[
            "itemid",
            "label",
        ],
    )

    selected_items = (
        d_labitems[
            d_labitems["label"].isin(
                TARGET_LABELS
            )
        ]
        .copy()
    )

    print(
        selected_items
        .sort_values("label")
        .to_string(index=False)
    )

    missing_labels = (
        TARGET_LABELS
        - set(selected_items["label"])
    )

    if missing_labels:
        raise ValueError(
            "Could not find lab labels: "
            f"{sorted(missing_labels)}"
        )

    item_to_name = {
        int(row.itemid):
            DISPLAY_NAMES.get(
                row.label,
                row.label,
            )
        for row in selected_items.itertuples()
    }

    target_itemids = set(
        item_to_name
    )

    print(
        "\nStreaming labevents..."
    )

    pieces = []

    usecols = [
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "valuenum",
    ]

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            labevents_path,
            usecols=usecols,
            chunksize=1_000_000,
        ),
        start=1,
    ):
        chunk = chunk[
            chunk["hadm_id"].isin(
                admission_ids
            )
            & chunk["itemid"].isin(
                target_itemids
            )
        ].copy()

        if chunk.empty:
            continue

        chunk = chunk.dropna(
            subset=[
                "hadm_id",
                "charttime",
                "valuenum",
            ]
        )

        chunk["Variable"] = (
            chunk["itemid"]
            .astype("int64")
            .map(item_to_name)
        )

        chunk["RecordDate"] = (
            pd.to_datetime(
                chunk["charttime"]
            )
            .dt.floor("D")
        )

        pieces.append(
            chunk[
                [
                    "subject_id",
                    "hadm_id",
                    "RecordDate",
                    "charttime",
                    "Variable",
                    "valuenum",
                ]
            ]
        )

        print(
            f"chunk {chunk_number}: "
            f"kept {len(chunk)} rows"
        )

    if not pieces:
        raise RuntimeError(
            "No matching lab events found."
        )

    labs = pd.concat(
        pieces,
        ignore_index=True,
    )

    print(
        "\nMatched lab rows:",
        len(labs),
    )

    labs["charttime"] = (
        pd.to_datetime(
            labs["charttime"]
        )
    )

    # Last measurement within each calendar day.
    labs = (
        labs
        .sort_values(
            "charttime"
        )
        .groupby(
            [
                "subject_id",
                "hadm_id",
                "RecordDate",
                "Variable",
            ],
            as_index=False,
        )
        .last()
    )

    labs_wide = (
        labs
        .pivot(
            index=[
                "subject_id",
                "hadm_id",
                "RecordDate",
            ],
            columns="Variable",
            values="valuenum",
        )
        .reset_index()
    )

    labs_wide = labs_wide.rename(
        columns={
            "subject_id": "PatientID",
            "hadm_id": "AdmissionID",
        }
    )

    # Match the date portion of the existing
    # ClinicRealm RecordTime.
    extended = ehr.copy()

    extended["RecordDate"] = (
        pd.to_datetime(
            extended["RecordTime"]
        )
        .dt.floor("D")
    )

    extended = extended.merge(
        labs_wide,
        on=[
            "PatientID",
            "AdmissionID",
            "RecordDate",
        ],
        how="left",
        validate="many_to_one",
    )

    extended = extended.drop(
        columns=["RecordDate"]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    extended.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n===== EXTENDED EHR ====="
    )

    print(
        "Rows:",
        len(extended),
    )

    print(
        "Admissions:",
        extended[
            ["PatientID", "AdmissionID"]
        ]
        .drop_duplicates()
        .shape[0],
    )

    for feature in [
        "Creatinine",
        "BUN",
        "Lactate",
        "WBC",
        "Hemoglobin",
        "Platelets",
    ]:
        if feature not in extended:
            print(
                f"{feature:12s}: missing column"
            )
            continue

        available = int(
            extended[feature]
            .notna()
            .sum()
        )

        print(
            f"{feature:12s}: "
            f"{available}/{len(extended)} "
            f"rows observed"
        )

    print(
        "\nSaved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
