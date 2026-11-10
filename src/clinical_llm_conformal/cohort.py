"""Source-independent admission cohort manifests.

One row identifies one MIMIC hospital admission. This deliberately does not
manufacture an ICU stay or prediction cutoff from longitudinal observations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ("subject_id", "hadm_id")
OPTIONAL_COLUMNS = ("stay_id", "prediction_time", "label")


def validate_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an admission-level cohort without selecting it."""
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Cohort missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cohort is empty")

    out = frame.copy()
    for col in REQUIRED_COLUMNS:
        numbers = pd.to_numeric(out[col], errors="raise")
        if numbers.isna().any() or ((numbers % 1) != 0).any():
            raise ValueError(f"{col} must contain non-null integer identifiers")
        out[col] = numbers.astype("int64")

    if out.duplicated(list(REQUIRED_COLUMNS)).any():
        example = out.loc[out.duplicated(list(REQUIRED_COLUMNS), keep=False), list(REQUIRED_COLUMNS)].head(3)
        raise ValueError(f"Repeated admissions in manifest:\n{example.to_string(index=False)}")

    if "stay_id" in out.columns:
        out["stay_id"] = pd.to_numeric(out["stay_id"], errors="raise").astype("Int64")
    if "prediction_time" in out.columns:
        out["prediction_time"] = pd.to_datetime(out["prediction_time"], errors="raise")
    if "label" in out.columns:
        out["label"] = pd.to_numeric(out["label"], errors="raise").astype("Int64")
        valid = out["label"].dropna().isin([0, 1])
        if not valid.all():
            raise ValueError("label must be 0, 1, or missing")

    return out.sort_values(list(REQUIRED_COLUMNS), kind="stable").reset_index(drop=True)


def load_cohort(path: str | Path) -> pd.DataFrame:
    """Load a manifest produced by any upstream selection/preprocessing method."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Expected .csv or .parquet manifest, got {path}")
    return validate_cohort(frame)


def sample_cohort(frame: pd.DataFrame, *, n_admissions: int | None, seed: int) -> pd.DataFrame:
    """Deterministically sample admissions; do not silently sample with replacement."""
    out = validate_cohort(frame)
    if n_admissions is None:
        return out
    if n_admissions < 1 or n_admissions > len(out):
        raise ValueError(f"n_admissions must be between 1 and {len(out)}")
    return validate_cohort(out.sample(n=n_admissions, random_state=seed))


def save_cohort(frame: pd.DataFrame, output: str | Path, *, provenance: dict[str, Any]) -> Path:
    """Write manifest and adjacent provenance metadata; never export patient-level data to git."""
    output = Path(output)
    validated = validate_cohort(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        validated.to_parquet(output, index=False)
    elif output.suffix.lower() == ".csv":
        validated.to_csv(output, index=False)
    else:
        raise ValueError("Output must end in .csv or .parquet")
    metadata_file = output.with_suffix(".metadata.json")
    metadata_file.write_text(
        json.dumps({**provenance, "schema_version": 1,
                    "unit_of_selection": "hospital_admission",
                    "n_admissions": len(validated),
                    "n_subjects": int(validated["subject_id"].nunique()),
                    "columns": list(validated.columns)}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return output
