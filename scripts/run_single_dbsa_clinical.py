from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

from clinical_llm_conformal.clinical_data import (
    build_clinical_record,
    load_diagnoses,
    load_prescriptions,
)
from clinical_llm_conformal.data import (
    LABTEST_FEATURES,
    build_patient_records,
    load_clinicrealm_ehr,
)
from clinical_llm_conformal.dbsa import dbsa_risk_rank
from clinical_llm_conformal.extended_prompt import (
    EXTENDED_LAB_FEATURES,
    build_extended_clinical_prompt,
)
from clinical_llm_conformal.llm import OpenAICompatibleLLM
from clinical_llm_conformal.parsing import parse_llm_response
from clinical_llm_conformal.prompt import CHUNK_DEFINITIONS


CONFIG_PATH = Path("configs/clinical_domains.yaml")
OUTPUT_PATH = Path("tmp/clinical_domains/dbsa_single_clinical.json")

PATIENT_KEY = (10032409, 28020206)

DOMAINS = [
    "demographics",
    "clinicrealm_17",
    "labs",
    "diagnoses",
    "medications",
]

EXTRA_CHUNKS = ["Diagnoses", "Medications"]

# Integration-test settings, not statistical calibration.
N_FULL = 2
N_PERTURBED = 2
N_PERMUTATIONS = 200
MAX_WORKERS = 2
TEMPERATURE = 0.7


def remove_section(text: str, heading: str) -> str:
    """Remove a complete bullet-list section from the prompt."""
    pattern = (
        r"(?m)^"
        + re.escape(heading)
        + r"\n(?:- [^\n]*(?:\n|$))*"
    )
    result, count = re.subn(pattern, "", text)

    if count != 1:
        raise ValueError(
            f"Expected exactly one section: {heading!r}; found {count}"
        )
    return result


def perturb_prompt(
    full_prompt: str,
    retained_chunks: list[str],
    chunk_names: list[str],
) -> str:
    """Delete omitted clinical chunks from the original full prompt."""
    keep = set(retained_chunks)
    unknown = keep - set(chunk_names)
    if unknown:
        raise ValueError(f"Unknown DBSA chunks: {sorted(unknown)}")

    prompt = full_prompt

    for chunk, features in CHUNK_DEFINITIONS.items():
        if chunk == "Demographics" or chunk in keep:
            continue

        for feature in features:
            pattern = (
                r"(?m)^- "
                + re.escape(feature)
                + r" \(Unit: [^\n]*(?:\n|$)"
            )
            prompt, count = re.subn(pattern, "", prompt)

            if count != 1:
                raise ValueError(
                    f"Expected one prompt line for {feature!r}, "
                    f"found {count}"
                )

    if "Demographics" not in keep:
        prompt, sex_count = re.subn(
            r"(?m)^- Sex: [^\n]*$",
            "- Sex: unknown",
            prompt,
        )
        prompt, age_count = re.subn(
            r"(?m)^- Age: [^\n]*$",
            "- Age: unknown years",
            prompt,
        )

        if sex_count != 1 or age_count != 1:
            raise ValueError("Could not perturb demographics")

    if "Diagnoses" not in keep:
        prompt = remove_section(
            prompt,
            "Diagnoses (admission-associated ICD codes; timing not verified):",
        )

    if "Medications" not in keep:
        prompt = remove_section(
            prompt,
            "Medication orders (prescriptions, not confirmed administrations):",
        )

    return prompt


def main() -> None:
    load_dotenv()

    config = yaml.safe_load(CONFIG_PATH.read_text())

    data = config["data"]
    options = config["prompt"]
    generation = config["generation"]

    if not config.get("experiment", {}).get(
        "allow_unverified_events", False
    ):
        raise ValueError(
            "DBSA with retrospective diagnoses and prescriptions "
            "requires allow_unverified_events: true"
        )

    ehr = load_clinicrealm_ehr(Path(data["extended_ehr_path"]))

    features = list(LABTEST_FEATURES) + list(EXTENDED_LAB_FEATURES)

    patients = build_patient_records(ehr, features=features)

    if PATIENT_KEY not in patients:
        raise KeyError(f"Admission not found: {PATIENT_KEY}")

    diagnoses = load_diagnoses(Path(data["diagnoses_path"]))
    prescriptions = load_prescriptions(Path(data["prescriptions_path"]))

    record = build_clinical_record(
        key=PATIENT_KEY,
        patient=patients[PATIENT_KEY],
        diagnoses_df=diagnoses,
        prescriptions_df=prescriptions,
    )

    system_prompt, full_prompt = build_extended_clinical_prompt(
        clinical_record=record,
        domains=DOMAINS,
        task=options["task"],
        n_shot=int(options["n_shot"]),
        unit=bool(options["unit"]),
        reference_range=bool(options["reference_range"]),
        format_type=options["format_type"],
        allow_unverified_events=True,
        max_medications=int(options["max_medications"]),
    )

    chunk_names = list(CHUNK_DEFINITIONS) + EXTRA_CHUNKS

    if len(chunk_names) != len(set(chunk_names)):
        raise ValueError("Duplicate DBSA chunk names")

    assigned = {
        feature
        for chunk, members in CHUNK_DEFINITIONS.items()
        if chunk != "Demographics"
        for feature in members
    }

    unassigned = set(features) - assigned

    if unassigned:
        raise ValueError(
            f"Unassigned measurement features: {sorted(unassigned)}"
        )

    def prompt_builder(retained_chunks):
        retained = list(retained_chunks)

        user_prompt = perturb_prompt(
            full_prompt,
            retained,
            chunk_names,
        )

        return system_prompt, user_prompt

    # Validate that the unperturbed prompt remains byte-for-byte identical.
    assert prompt_builder(chunk_names) == (
        system_prompt,
        full_prompt,
    )

    # Validate that every chunk can be removed independently.
    for removed in chunk_names:
        retained = [
            chunk for chunk in chunk_names if chunk != removed
        ]

        system, user = prompt_builder(retained)

        assert system == system_prompt
        assert user != full_prompt

        if removed == "Diagnoses":
            assert "Diagnoses (admission-associated" not in user
            assert "Medication orders (prescriptions" in user

        if removed == "Medications":
            assert "Medication orders (prescriptions" not in user
            assert "Diagnoses (admission-associated" in user

    prompt_hash = hashlib.sha256(
        (system_prompt + "\n" + full_prompt).encode()
    ).hexdigest()

    print("Admission:", PATIENT_KEY)
    print("Outcome:", record["label"])
    print("Visits:", len(record["measurements"]["day_cols"]))
    print("Diagnoses:", len(record["diagnoses"]))
    print("Prescription orders:", len(record["medications"]))
    print("DBSA chunks:", len(chunk_names))
    print("Full prompt SHA-256:", prompt_hash)
    print("Perturbation checks: PASS")

    llm = OpenAICompatibleLLM(
        model=os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ.get("LLM_API_KEY", "local"),
    )

    def sampler(system: str, user: str) -> float:
        response = llm.generate(
            system_prompt=system,
            user_prompt=user,
            temperature=TEMPERATURE,
            max_tokens=int(generation["max_tokens"]),
            enable_thinking=bool(generation["enable_thinking"]),
            reasoning_effort=generation["reasoning_effort"],
            top_k=int(generation["top_k"]),
            min_p=float(generation["min_p"]),
        )

        parsed = parse_llm_response(response.text)

        if parsed.failed or parsed.probability is None:
            raise RuntimeError(
                "DBSA sample did not return a valid mortality probability. "
                f"finish_reason={response.finish_reason}"
            )

        return float(parsed.probability)

    print("\nRunning single-admission DBSA...")
    print("This is an integration test, not statistical calibration.\n")

    result = dbsa_risk_rank(
        chunk_names=chunk_names,
        prompt_builder=prompt_builder,
        sampler=sampler,
        n_full=N_FULL,
        n_perturbed=N_PERTURBED,
        n_permutations=N_PERMUTATIONS,
        max_workers=MAX_WORKERS,
        seed=int(config["seed"]),
    )

    output = {
        "patient_id": PATIENT_KEY[0],
        "admission_id": PATIENT_KEY[1],
        "outcome": int(record["label"]),
        "model": os.environ["LLM_MODEL"],
        "prompt_sha256": prompt_hash,
        "exploratory_unverified_events": True,
        "n_full": N_FULL,
        "n_perturbed": N_PERTURBED,
        "n_permutations": N_PERMUTATIONS,
        "ranked_chunks": list(result.ranked_chunks),
        "chunks": {},
    }

    print("\n===== DBSA RESULTS =====")

    for rank, chunk in enumerate(result.ranked_chunks, start=1):
        item = result.chunks[chunk]

        output["chunks"][chunk] = {
            "energy_distance": float(item.energy_distance),
            "p_value": float(item.p_value),
            "full_samples": list(item.full_samples),
            "perturbed_samples": list(item.perturbed_samples),
        }

        print(
            f"{rank:2d}. {chunk:30s} "
            f"E={item.energy_distance:.4f} "
            f"p={item.p_value:.3f}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2) + "\n"
    )

    print(f"\nResults saved to {OUTPUT_PATH}")
    print("Single-admission clinical DBSA: PASS")
    print(
        "WARNING: Diagnoses and prescriptions have unverified "
        "prediction-time availability."
    )


if __name__ == "__main__":
    main()
