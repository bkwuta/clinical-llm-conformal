from pathlib import Path

import pandas as pd

LABTEST_FEATURES = [
    "Capillary refill rate",
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale total",
    "Glascow coma scale verbal response",
    "Diastolic blood pressure",
    "Fraction inspired oxygen",
    "Glucose",
    "Heart Rate",
    "Height",
    "Mean blood pressure",
    "Oxygen saturation",
    "Respiratory rate",
    "Systolic blood pressure",
    "Temperature",
    "Weight",
    "pH",
]


def load_clinicrealm_ehr(path: str | Path) -> pd.DataFrame:
    """Load EHR data produced by the official MIMIC-IV preprocessor."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist.\n"
            "Run the official PKU-AICare/mimic_preprocessor first."
        )

    df = pd.read_parquet(path)

    required = {
        "PatientID",
        "AdmissionID",
        "RecordTime",
        "Age",
        "Sex",
        "Outcome",
        *LABTEST_FEATURES,
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Input does not match expected ClinicRealm MIMIC-IV "
            f"preprocessor output. Missing columns: {missing}"
        )

    return df

def build_patient_records(ehr_df: pd.DataFrame) -> dict:
    patients = {}

    for (patient_id, admission_id), group in ehr_df.groupby(
        ["PatientID", "AdmissionID"]
    ):
        group = group.sort_values("RecordTime").reset_index(drop=True)

        day_cols = list(range(len(group)))
        record_times = group["RecordTime"].tolist()

        patient = {
            "Age": group["Age"].iloc[0],
            "Sex": group["Sex"].iloc[0],
        }

        for day, (_, row) in zip(day_cols, group.iterrows()):
            for feature in LABTEST_FEATURES:
                patient[(day, feature)] = row.get(feature, float("nan"))

        patients[(int(patient_id), int(admission_id))] = {
            "row": pd.Series(patient),
            "day_cols": day_cols,
            "record_times": record_times,
            "label": int(group["Outcome"].iloc[0]),
        }

    return patients