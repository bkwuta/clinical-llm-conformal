from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from clinical_llm_conformal.data import LABTEST_FEATURES
from clinical_llm_conformal.prompt import (
    _normalise_sex,
    format_clinicrealm_input,
    load_prompt_assets,
)


EXTENDED_LAB_FEATURES = [
    "Creatinine",
    "BUN",
    "Lactate",
    "WBC",
    "Hemoglobin",
    "Platelets",
]

SUPPORTED_DOMAINS = {
    "demographics",
    "clinicrealm_17",
    "labs",
    "diagnoses",
    "medications",
}

# Admission-associated ICD diagnoses and prescription
# records have not been verified as available at the
# intended mortality-prediction cutoff.
UNVERIFIED_DOMAINS = {
    "diagnoses",
    "medications",
}


def _text(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return str(value)


def format_diagnoses(
    diagnoses: list[dict],
) -> str:
    if not diagnoses:
        return "Diagnoses:\n- None recorded"

    lines = [
        "Diagnoses (admission-associated ICD codes; "
        "timing not verified):"
    ]

    for diagnosis in sorted(
        diagnoses,
        key=lambda d: d["sequence"],
    ):
        code = diagnosis["code"]
        description = (
            diagnosis["description"]
            or "Unknown description"
        )

        lines.append(
            f"- {code}: {description}"
        )

    return "\n".join(lines)


def format_medications(
    medications: list[dict],
    *,
    max_items: int = 20,
) -> str:
    if max_items < 1:
        raise ValueError(
            "max_items must be at least 1"
        )

    if not medications:
        return "Medication orders:\n- None recorded"

    # Deduplicate only fully identical order records.
    # Different routes, doses, or timing are preserved.
    unique = {}

    for medication in medications:
        identity = (
            medication["drug"],
            medication["dose"],
            medication["dose_unit"],
            medication["route"],
            medication["starttime"],
            medication["stoptime"],
        )
        unique[identity] = unique.get(identity, 0) + 1

    ordered = sorted(
        unique.items(),
        key=lambda item: (
            item[0][4] or "",
            item[0][0] or "",
            str(item[0]),
        ),
    )

    lines = [
        "Medication orders "
        "(prescriptions, not confirmed administrations):"
    ]

    for identity, count in ordered[:max_items]:
        (
            drug,
            dose,
            unit,
            route,
            start,
            stop,
        ) = identity

        dose_text = " ".join(
            part
            for part in (
                _text(dose),
                _text(unit),
            )
            if part != "Unknown"
        )

        line = (
            f"- {_text(drug)}"
            f" | Dose: {dose_text or 'Unknown'}"
            f" | Route: {_text(route)}"
            f" | Start: {_text(start)}"
            f" | Stop: {_text(stop)}"
        )

        if count > 1:
            line += f" | Identical records: {count}"

        lines.append(line)

    omitted = len(ordered) - max_items

    if omitted > 0:
        lines.append(
            f"- {omitted} additional distinct "
            "medication orders omitted from this preview."
        )

    return "\n".join(lines)


def build_extended_clinical_prompt(
    *,
    clinical_record: dict,
    domains: Sequence[str],
    task: str = "mortality",
    n_shot: int = 0,
    unit: bool = True,
    reference_range: bool = True,
    format_type: str = "list",
    dataset: str = "mimic-iv",
    allow_unverified_events: bool = False,
    max_medications: int = 20,
) -> tuple[str, str]:
    """
    Build a ClinicRealm-derived clinical prompt.

    The original ClinicRealm assets and longitudinal
    formatting are reused.

    Admission diagnoses and medication prescriptions
    are excluded unless explicitly enabled for
    exploratory inspection.
    """
    domains = list(dict.fromkeys(domains))

    unknown = set(domains) - SUPPORTED_DOMAINS

    if unknown:
        raise ValueError(
            f"Unsupported clinical domains: "
            f"{sorted(unknown)}"
        )

    unverified = set(domains) & UNVERIFIED_DOMAINS

    if unverified and not allow_unverified_events:
        raise ValueError(
            "Diagnoses and medication orders have "
            "unverified prediction-time availability. "
            "For formatting-only exploration, set "
            "allow_unverified_events=True."
        )

    assets = load_prompt_assets()

    if dataset not in assets["SYSTEMPROMPT"]:
        raise KeyError(
            f"Unknown dataset: {dataset}"
        )

    if task not in assets["TASK_DESCRIPTION"]:
        raise KeyError(
            f"Unknown task: {task}"
        )

    measurements = clinical_record["measurements"]

    patient_row = measurements["row"]
    day_cols = measurements["day_cols"]
    record_times = measurements["record_times"]

    # Select only measurement features belonging
    # to the requested clinical domains.
    features = []

    if "clinicrealm_17" in domains:
        features.extend(LABTEST_FEATURES)

    if "labs" in domains:
        features.extend(EXTENDED_LAB_FEATURES)

    features = list(dict.fromkeys(features))

    # Preserve ClinicRealm's longitudinal formatting.
    patient_values = [
        [
            patient_row.get(
                (day, feature),
                float("nan"),
            )
            for feature in features
        ]
        for day in day_cols
    ]

    sections = []

    if features:
        sections.append(
            format_clinicrealm_input(
                patient=patient_values,
                features=features,
                units=assets["UNIT"],
                reference_ranges=assets["REFERENCE_RANGE"],
                unit=unit,
                reference_range=reference_range,
                format_type=format_type,
            )
        )

    if "diagnoses" in domains:
        sections.append(
            format_diagnoses(
                clinical_record["diagnoses"]
            )
        )

    if "medications" in domains:
        sections.append(
            format_medications(
                clinical_record["medications"],
                max_items=max_medications,
            )
        )

    detail = "\n\n".join(sections)

    if not detail:
        detail = "No clinical measurements selected."

    if unverified:
        detail = (
            "EXPLORATORY DATA PREVIEW ONLY: "
            "Diagnosis and/or prescription information "
            "has not been verified as available at the "
            "prediction cutoff. Do not use this prompt "
            "for prospective mortality evaluation.\n\n"
            + detail
        )

    if "demographics" in domains:
        sex = _normalise_sex(
            clinical_record["demographics"]["sex"]
        )
        age = clinical_record["demographics"]["age"]
    else:
        sex = "unknown"
        age = "unknown"

    if n_shot == 0:
        example = ""
    elif n_shot == 1:
        example = (
            "Example:\n"
            + assets["EXAMPLE"][dataset][task]
        )
    else:
        raise ValueError(
            f"Unsupported n_shot: {n_shot}"
        )

    system_prompt = (
        assets["SYSTEMPROMPT"][dataset].strip()
    )

    user_prompt = assets["USERPROMPT"].format(
        TASK_DESCRIPTION=(
            assets["TASK_DESCRIPTION"][task].strip()
        ),
        EXAMPLE=example.strip(),
        SEX=sex,
        AGE=age,
        LENGTH=len(day_cols),
        RECORD_TIME_LIST=", ".join(
            map(str, record_times)
        ),
        DETAIL=detail,
        RESPONSE_FORMAT=(
            assets["RESPONSE_FORMAT"][task].strip()
        ),
    )

    return system_prompt, user_prompt
