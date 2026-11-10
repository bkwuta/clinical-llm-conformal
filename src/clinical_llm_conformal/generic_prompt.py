"""Format Stage 6C generic_records without imposing ClinicRealm's feature list.

This is a NEW representation. It does not replace the exact ClinicRealm baseline
prompt path. Observations and diagnoses/medications are NOT automatically
prediction-time eligible merely because they can be formatted.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd

from .feature_catalog import FeatureSpec
from .prompt import load_prompt_assets, _normalise_sex


def _value_text(value: object, kind: str) -> str:
    if value is None or (isinstance(value, (float, int)) and not isinstance(value, bool) and pd.isna(value)):
        return "NaN" if kind == "numeric" else "unknown"
    if pd.isna(value):
        return "NaN" if kind == "numeric" else "unknown"
    if kind == "numeric":
        if isinstance(value, str):
            try:
                number = float(value)
            except ValueError as exc:
                raise ValueError(f"Numeric measurement contains nonnumeric value {value!r}") from exc
        else:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Numeric measurement contains nonnumeric value {value!r}") from exc
        return str(number) if math.isfinite(number) else "NaN"
    return str(value)


def format_generic_measurements(
    patient: Mapping[str, object],
    *,
    catalog: Mapping[str, FeatureSpec],
    features: Sequence[str],
    include_units: bool = True,
    include_reference_ranges: bool = True,
) -> str:
    """Render only explicitly selected features, preserving visit order and missingness."""
    selected = list(features)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("features must be a nonempty list without duplicates")
    missing = set(selected) - set(catalog)
    if missing:
        raise ValueError(f"Features missing from catalog: {sorted(missing)}")
    row = patient["row"]
    day_cols = patient["day_cols"]
    lines: list[str] = []
    for feature in selected:
        spec = catalog[feature]
        props: list[str] = []
        if include_units and spec.unit:
            props.append(f"Unit: {spec.unit}")
        if include_reference_ranges and spec.reference_range:
            props.append(f"Reference Range: {spec.reference_range}")
        heading = spec.display_name + (f" ({'. '.join(props)})" if props else "")
        values = [_value_text(row.get((day, feature), None), spec.kind) for day in day_cols]
        lines.append(f"- {heading}: [{', '.join(values)}]")
    return "\n".join(lines)


def build_generic_clinical_prompt(
    *,
    clinical_record: Mapping[str, object],
    catalog: Mapping[str, FeatureSpec],
    features: Sequence[str],
    task: str = "mortality",
    dataset: str = "mimic-iv",
    n_shot: int = 0,
    include_units: bool = True,
    include_reference_ranges: bool = True,
) -> tuple[str, str]:
    """Use pinned ClinicRealm task framing with a general measurement-detail section.

    Does not render retrospective diagnoses or medications; those remain on the
    separate, explicitly exploratory extended-prompt path.
    """
    if n_shot not in {0, 1}:
        raise ValueError("Only n_shot=0 or n_shot=1 are supported")
    assets = load_prompt_assets()
    measurements = clinical_record["measurements"]
    timestamps = list(measurements["record_times"])
    days = list(measurements["day_cols"])
    if len(timestamps) != len(days):
        raise ValueError("record_times and day_cols lengths differ")

    detail = format_generic_measurements(
        measurements,
        catalog=catalog,
        features=features,
        include_units=include_units,
        include_reference_ranges=include_reference_ranges,
    )
    demographic = clinical_record.get("demographics") or {}
    sex_value = demographic.get("sex")
    age_value = demographic.get("age")
    sex = "unknown" if pd.isna(sex_value) else _normalise_sex(sex_value)
    age = "unknown" if pd.isna(age_value) else age_value
    example = "" if n_shot == 0 else "Example:\n" + assets["EXAMPLE"][dataset][task]
    system = assets["SYSTEMPROMPT"][dataset].strip()
    user = assets["USERPROMPT"].format(
        TASK_DESCRIPTION=assets["TASK_DESCRIPTION"][task].strip(),
        EXAMPLE=example,
        SEX=sex,
        AGE=age,
        LENGTH=len(days),
        RECORD_TIME_LIST=", ".join(map(str, timestamps)),
        DETAIL=detail,
        RESPONSE_FORMAT=assets["RESPONSE_FORMAT"][task].strip(),
    )
    return system, user
