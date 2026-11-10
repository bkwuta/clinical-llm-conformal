"""Adapters for pinned ClinicRealm records and generic observation tables.

The ClinicRealm path deliberately delegates to its original preprocessing,
so benchmark prompt reproduction is unaffected.
"""
from __future__ import annotations

import pandas as pd

from .data import build_patient_records
from .measurement import validate_measurements


def clinicrealm_records(ehr: pd.DataFrame, *, features: list[str]) -> dict:
    """Unchanged ClinicRealm record creation; call with original 17 features for parity."""
    return build_patient_records(ehr, features=features)


def generic_records(
    measurements: pd.DataFrame,
    *,
    demographics: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
) -> dict[tuple[int, int], dict]:
    """Create records shaped for build_clinical_record/extended prompt consumers.

    Unlike ClinicRealm, each unique observation timestamp is a visit. This is
    an explicit representation choice, not a benchmark-equivalent conversion.
    The generic record contains all features; current extended_prompt formats
    only the known ClinicRealm+six-lab features unless separately extended.
    """
    frame = validate_measurements(measurements)
    info = {}
    for name, table in (("demographics", demographics), ("labels", labels)):
        if table is None:
            continue
        if not {"subject_id", "hadm_id"}.issubset(table.columns):
            raise ValueError(f"{name} must contain subject_id and hadm_id")
        if table.duplicated(["subject_id", "hadm_id"]).any():
            raise ValueError(f"{name} has duplicate admission identifiers")
        info[name] = table.set_index(["subject_id", "hadm_id"])

    records = {}
    for (pid, aid), group in frame.groupby(["subject_id", "hadm_id"], sort=True):
        key = (int(pid), int(aid))
        timestamps = list(group["timestamp"].drop_duplicates())
        lookup = {stamp: i for i, stamp in enumerate(timestamps)}
        values = {(lookup[r.timestamp], r.feature): r.value for r in group.itertuples(index=False)}
        # A flat object Index supports both tuple feature keys and scalar Age/Sex keys.
        # Letting pandas infer a MultiIndex makes row["Age"] a Series, breaking prompts.
        row = pd.Series(list(values.values()), index=pd.Index(list(values), tupleize_cols=False), dtype="object")
        demo = info.get("demographics")
        demo_row = demo.loc[key] if demo is not None and key in demo.index else None
        row.loc["Age"] = demo_row["Age"] if demo_row is not None and "Age" in demo_row else float("nan")
        row.loc["Sex"] = demo_row["Sex"] if demo_row is not None and "Sex" in demo_row else float("nan")
        label_table = info.get("labels")
        label_row = label_table.loc[key] if label_table is not None and key in label_table.index else None
        label = label_row["label"] if label_row is not None and "label" in label_row else None
        records[key] = {
            "row": row,
            "day_cols": list(range(len(timestamps))),
            "record_times": [t.isoformat(sep=" ", timespec="seconds") for t in timestamps],
            "label": label,
            "representation": "observation_timestamp",
        }
    return records
