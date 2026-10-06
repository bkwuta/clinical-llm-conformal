from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Literal

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CLINICREALM_ROOT = ROOT / "vendor" / "ehr-llm-benchmark"

PROMPT_DIR = (
    CLINICREALM_ROOT
    / "src"
    / "structured_ehr"
    / "prompts"
)

CHUNK_DEFINITIONS = {
    "Demographics": [],
    "Hemodynamics (HR/BP)": [
        "Heart Rate",
        "Systolic blood pressure",
        "Diastolic blood pressure",
        "Mean blood pressure",
    ],
    "Respiratory status": [
        "Respiratory rate",
        "Fraction inspired oxygen",
        "Oxygen saturation",
    ],
    "Temperature": [
        "Temperature",
    ],
    "Neuro status (GCS)": [
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale verbal response",
        "Glascow coma scale total",
    ],
    "Capillary refill": [
        "Capillary refill rate",
    ],
    "Glucose": [
        "Glucose",
    ],
    "Blood pH": [
        "pH",
    ],
    "Weight": [
        "Weight",
    ],
    "Height": [
        "Height",
    ],
}

CHUNK_NAMES = list(CHUNK_DEFINITIONS)


def _extract_assignment_literal(source: str, name: str):
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)

    raise KeyError(name)


def load_prompt_assets() -> dict:
    """Load the exact prompt assets from the pinned ClinicRealm submodule."""
    source = (PROMPT_DIR / "prompt_template.py").read_text()

    unit_path = PROMPT_DIR / "mimic-iv" / "unit.json"
    range_path = PROMPT_DIR / "mimic-iv" / "range.json"

    return {
        "SYSTEMPROMPT": _extract_assignment_literal(
            source, "SYSTEMPROMPT"
        ),
        "TASK_DESCRIPTION": _extract_assignment_literal(
            source, "TASK_DESCRIPTION"
        ),
        "RESPONSE_FORMAT": _extract_assignment_literal(
            source, "RESPONSE_FORMAT"
        ),
        "EXAMPLE": _extract_assignment_literal(
            source, "EXAMPLE"
        ),
        "USERPROMPT": _extract_assignment_literal(
            source, "USERPROMPT"
        ),
        "UNIT": json.loads(unit_path.read_text()),
        "REFERENCE_RANGE": json.loads(range_path.read_text()),
    }


def _format_value(feature: str, value) -> str:
    categorical_features = {
        "Glascow coma scale eye opening": {
            1: "No Response",
            2: "To Pain",
            3: "To Speech",
            4: "Spontaneously",
        },
        "Glascow coma scale motor response": {
            1: "No Response",
            2: "Abnormal Extension",
            3: "Abnormal Flexion",
            4: "Flex-withdraws",
            5: "Localizes Pain",
            6: "Obeys Commands",
        },
        "Glascow coma scale verbal response": {
            1: "No Response",
            2: "Incomprehensible sounds",
            3: "Inappropriate Words",
            4: "Confused",
            5: "Oriented",
        },
    }

    if pd.isna(value):
        return "NaN"

    if feature in categorical_features:
        try:
            key = int(value)
        except (TypeError, ValueError):
            return "NaN"

        return categorical_features[feature].get(key, "NaN")

    return str(value)


def format_clinicrealm_input(
    *,
    patient: list[list],
    features: list[str],
    units: dict,
    reference_ranges: dict,
    unit: bool = False,
    reference_range: bool = False,
    format_type: Literal["list", "text"] = "list",
) -> str:
    """
    Reproduce ClinicRealm's structured-EHR format_input() behavior.

    Unlike the upstream CLI implementation, this function does not depend
    on the global argparse `args` object.
    """
    feature_values: dict[str, list[str]] = {}

    for i, feature in enumerate(features):
        feature_values[feature] = [
            _format_value(feature, visit[i])
            for visit in patient
        ]

    detail = ""

    if format_type == "list":
        for feature in features:
            unit_range = ""

            if unit or reference_range:
                unit_range = " ("

                if unit:
                    unit_range += f"Unit: {units[feature]}. "

                if reference_range:
                    unit_range += (
                        "Reference Range: "
                        f"{reference_ranges[feature]}. "
                    )

                unit_range = unit_range.rstrip() + ")"

            detail += (
                f"- {feature}{unit_range}: "
                f"[{', '.join(feature_values[feature])}]\n"
            )

    elif format_type == "text":
        for feature in features:
            values = feature_values[feature]

            if unit and units[feature] != "/":
                values = [
                    value + " " + units[feature]
                    for value in values
                ]

            if len(values) > 1:
                values_str = ", ".join(values[:-1]) + " and " + values[-1]
            else:
                values_str = values[-1]

            reference_range_str = ""

            if (
                reference_range
                and reference_ranges[feature] != "/"
            ):
                reference_range_str = (
                    ", with a reference range of "
                    f"{reference_ranges[feature]}"
                )

            visit_count = len(values)
            visit_word = "visits" if visit_count > 1 else "visit"

            detail += (
                f"The feature `{feature}` has {visit_count} "
                f"{visit_word} of values {values_str}"
                f"{reference_range_str}. "
            )

    else:
        raise ValueError(
            f"Unsupported format_type: {format_type}"
        )

    return detail.strip()


def _normalise_sex(value) -> str:
    """
    Convert the processed MIMIC sex representation into ClinicRealm's
    'male'/'female' prompt representation.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"m", "male"}:
            return "male"

        if normalized in {"f", "female"}:
            return "female"

    if value == 1:
        return "male"

    if value == 0:
        return "female"

    raise ValueError(f"Unexpected Sex value: {value!r}")


def build_clinicrealm_prompt(
    *,
    patient_row: pd.Series,
    day_cols: list,
    record_times: list,
    features: list[str],
    task: str,
    n_shot: int = 0,
    unit: bool = True,
    reference_range: bool = True,
    format_type: Literal["list", "text"] = "list",
    dataset: str = "mimic-iv",
    retained_chunks: list[str] | None = None,
) -> tuple[str, str]:
    """
    Build system/user prompts using the pinned ClinicRealm templates.
    """
    assets = load_prompt_assets()

    if dataset not in assets["SYSTEMPROMPT"]:
        raise KeyError(
            f"Unknown dataset {dataset!r}. "
            f"Available: {list(assets['SYSTEMPROMPT'])}"
        )

    if task not in assets["TASK_DESCRIPTION"]:
        raise KeyError(
            f"Unknown task {task!r}. "
            f"Available: {list(assets['TASK_DESCRIPTION'])}"
        )

    if retained_chunks is None:
        retained_chunks = CHUNK_NAMES
        
    keep = set(retained_chunks)
    
    selected_features = []
    
    for chunk, chunk_features in CHUNK_DEFINITIONS.items():
        if chunk == "Demographics":
            continue
        
        if chunk in keep:
            selected_features.extend(chunk_features)

    system_prompt = assets["SYSTEMPROMPT"][dataset].strip()

    task_description = assets["TASK_DESCRIPTION"][task].strip()

    if n_shot == 0:
        example = ""
    elif n_shot == 1:
        example = "Example:\n"
        example += assets["EXAMPLE"][dataset][task]
    else:
        raise ValueError(
            f"Invalid n_shot value: {n_shot}"
        )

    example = example.strip()

    response_format = assets["RESPONSE_FORMAT"][task].strip()

    patient = [
        [
            patient_row.get((day, feature), float("nan"))
            for feature in selected_features
        ]
        for day in day_cols
    ]

    detail = format_clinicrealm_input(
        patient=patient,
        features=selected_features,
        units=assets["UNIT"],
        reference_ranges=assets["REFERENCE_RANGE"],
        unit=unit,
        reference_range=reference_range,
        format_type=format_type,
    )

    if "Demographics" in keep:
        sex = _normalise_sex(patient_row["Sex"])
        age = patient_row["Age"]
    else:
        sex = "unknown"
        age = "unknown"
    length = len(day_cols)

    user_prompt = assets["USERPROMPT"].format(
        TASK_DESCRIPTION=task_description,
        EXAMPLE=example,
        SEX=sex,
        AGE=age,
        LENGTH=length,
        RECORD_TIME_LIST=", ".join(map(str, record_times)),
        DETAIL=detail,
        RESPONSE_FORMAT=response_format,
    )

    return system_prompt, user_prompt