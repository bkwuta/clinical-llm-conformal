"""Source-independent longitudinal measurement table.

A row is one observation. Time is the actual observation timestamp; no
assumption about a daily visit or prediction cutoff is made here.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ("subject_id", "hadm_id", "timestamp", "feature", "value")
OPTIONAL_COLUMNS = ("unit", "source")


def validate_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Measurements missing columns: {sorted(missing)}")
    out = frame.copy()
    for col in ("subject_id", "hadm_id"):
        value = pd.to_numeric(out[col], errors="raise")
        if value.isna().any() or (value % 1 != 0).any():
            raise ValueError(f"{col} must be non-null integer IDs")
        out[col] = value.astype("int64")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    if out["timestamp"].isna().any():
        raise ValueError("timestamp cannot be missing")
    if out["feature"].isna().any() or out["feature"].astype(str).str.strip().eq("").any():
        raise ValueError("feature must be nonempty")
    out["feature"] = out["feature"].astype(str)
    # Keep categorical measurements (e.g. GCS response) without coercion.
    for col in OPTIONAL_COLUMNS:
        if col not in out:
            out[col] = pd.NA
    # Do not silently reduce repeated measurements at identical times.
    if out.duplicated(["subject_id", "hadm_id", "timestamp", "feature"]).any():
        raise ValueError("Repeated (subject_id, hadm_id, timestamp, feature); resolve with an explicit aggregation policy")
    return out.sort_values(["subject_id", "hadm_id", "timestamp", "feature"], kind="stable").reset_index(drop=True)


def load_measurements(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"value": "string"})
    else:
        raise ValueError("Measurements must be .csv or .parquet")
    return validate_measurements(frame)


def check_cohort_coverage(measurements: pd.DataFrame, cohort: pd.DataFrame) -> dict[str, int]:
    """Reject records outside selected admissions; report missing admissions."""
    selected = set(map(tuple, cohort[["subject_id", "hadm_id"]].itertuples(index=False, name=None)))
    found = set(map(tuple, measurements[["subject_id", "hadm_id"]].itertuples(index=False, name=None)))
    unexpected = found - selected
    if unexpected:
        raise ValueError(f"Measurements contain {len(unexpected)} admission(s) outside cohort")
    return {"admissions_in_cohort": len(selected), "admissions_with_measurements": len(found), "admissions_missing_measurements": len(selected - found)}
