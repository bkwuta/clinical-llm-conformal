"""Run configurable DBSA over one admission (retrospective integration test).

Examples:
  uv run python scripts/run_single_dbsa_clinical.py --chunk-set detailed_15 --check-only
  uv run python scripts/run_single_dbsa_clinical.py --chunk-set broad_6 --check-only
  uv run python scripts/run_single_dbsa_clinical.py --chunk-set detailed_15
  uv run python scripts/run_single_dbsa_clinical.py --chunk-set broad_6
"""
from __future__ import annotations

import argparse
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
from clinical_llm_conformal.dbsa_chunk_config import load_chunk_set
from clinical_llm_conformal.extended_prompt import (
    EXTENDED_LAB_FEATURES,
    build_extended_clinical_prompt,
)
from clinical_llm_conformal.llm import OpenAICompatibleLLM
from clinical_llm_conformal.parsing import parse_llm_response

DEFAULT_CONFIG = Path("configs/clinical_domains.yaml")
DEFAULT_CHUNKS = Path("configs/dbsa_chunks.yaml")
DEFAULT_KEY = (10032409, 28020206)
DOMAINS = ["demographics", "clinicrealm_17", "labs", "diagnoses", "medications"]
DIAGNOSIS_HEADING = "Diagnoses (admission-associated ICD codes; timing not verified):"
MEDICATION_HEADING = "Medication orders (prescriptions, not confirmed administrations):"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chunk-config", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--chunk-set", default="detailed_15")
    parser.add_argument("--patient-id", type=int, default=DEFAULT_KEY[0])
    parser.add_argument("--admission-id", type=int, default=DEFAULT_KEY[1])
    parser.add_argument("--check-only", action="store_true", help="Validate every perturbation, without LLM requests")
    parser.add_argument("--n-full", type=int, default=2)
    parser.add_argument("--n-perturbed", type=int, default=2)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def remove_section(text: str, heading: str) -> str:
    pattern = r"(?m)^" + re.escape(heading) + r"\n(?:- [^\n]*(?:\n|$))*"
    result, count = re.subn(pattern, "", text)
    if count != 1:
        raise ValueError(f"Expected exactly one section {heading!r}; found {count}")
    return result


def perturb_prompt(
    full_prompt: str,
    retained_chunks: list[str],
    chunks: dict[str, dict[str, list[str]]],
) -> str:
    retained = set(retained_chunks)
    if len(retained) != len(retained_chunks):
        raise ValueError("Duplicate retained chunk")
    if retained - set(chunks):
        raise ValueError(f"Unknown retained chunk(s): {sorted(retained - set(chunks))}")

    user = full_prompt
    for name, spec in chunks.items():
        if name in retained:
            continue
        for feature in spec["features"]:
            # Match precisely one rendered ClinicRealm feature line.
            pattern = r"(?m)^- " + re.escape(feature) + r" \(Unit: [^\n]*(?:\n|$)"
            user, count = re.subn(pattern, "", user)
            if count != 1:
                raise ValueError(f"{name}: expected one line for {feature!r}; found {count}")
        for domain in spec["domains"]:
            if domain == "demographics":
                user, sex_count = re.subn(r"(?m)^- Sex: [^\n]*$", "- Sex: unknown", user)
                user, age_count = re.subn(r"(?m)^- Age: [^\n]*$", "- Age: unknown years", user)
                if (sex_count, age_count) != (1, 1):
                    raise ValueError("Could not mask demographics")
            elif domain == "diagnoses":
                user = remove_section(user, DIAGNOSIS_HEADING)
            elif domain == "medications":
                user = remove_section(user, MEDICATION_HEADING)
            else:
                raise ValueError(f"Unexpected domain: {domain}")
    return user


def main() -> None:
    args = parse_args()
    load_dotenv()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not config.get("experiment", {}).get("allow_unverified_events", False):
        raise ValueError("Retrospective diagnoses/medications require allow_unverified_events: true")

    features = list(LABTEST_FEATURES) + list(EXTENDED_LAB_FEATURES)
    chunks = load_chunk_set(args.chunk_config, args.chunk_set, available_features=features)
    key = (args.patient_id, args.admission_id)
    data = config["data"]
    options = config["prompt"]
    generation = config["generation"]

    ehr = load_clinicrealm_ehr(Path(data["extended_ehr_path"]))
    patients = build_patient_records(ehr, features=features)
    if key not in patients:
        raise KeyError(f"Admission {key} not found")
    diagnoses = load_diagnoses(Path(data["diagnoses_path"]))
    prescriptions = load_prescriptions(Path(data["prescriptions_path"]))
    record = build_clinical_record(
        key=key, patient=patients[key], diagnoses_df=diagnoses, prescriptions_df=prescriptions
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

    def prompt_builder(retained_chunks: list[str]) -> tuple[str, str]:
        return system_prompt, perturb_prompt(full_prompt, list(retained_chunks), chunks)

    chunk_names = list(chunks)
    if prompt_builder(chunk_names) != (system_prompt, full_prompt):
        raise AssertionError("Full prompt changed when all chunks are retained")
    checksums = {}
    for removed in chunk_names:
        kept = [name for name in chunk_names if name != removed]
        system, user = prompt_builder(kept)
        if system != system_prompt or user == full_prompt:
            raise AssertionError(f"Failed to perturb {removed}")
        checksums[removed] = hashlib.sha256(user.encode("utf-8")).hexdigest()

    full_digest = hashlib.sha256(
        (system_prompt + "\n" + full_prompt).encode("utf-8")
    ).hexdigest()
    chunk_digest = hashlib.sha256(args.chunk_config.read_bytes()).hexdigest()
    print("Admission:", key)
    print("Outcome:", record["label"])
    print("Visits:", len(record["measurements"]["day_cols"]))
    print("Diagnoses:", len(record["diagnoses"]))
    print("Prescription orders:", len(record["medications"]))
    print("Chunk set:", args.chunk_set)
    print("DBSA chunks:", len(chunk_names))
    print("Full prompt SHA-256:", full_digest)
    print("Perturbation checks: PASS")

    if args.check_only:
        print("Check-only mode: no LLM requests were made.")
        return

    for n in (args.n_full, args.n_perturbed, args.n_permutations, args.max_workers):
        if n < 1:
            raise ValueError("All DBSA sampling parameters must be positive")
    llm = OpenAICompatibleLLM(
        model=os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ.get("LLM_API_KEY", "local"),
    )

    def sampler(system: str, user: str) -> float:
        response = llm.generate(
            system_prompt=system, user_prompt=user,
            temperature=0.7,  # Same setting as single-admission integration run.
            max_tokens=int(generation["max_tokens"]),
            enable_thinking=bool(generation["enable_thinking"]),
            reasoning_effort=generation["reasoning_effort"],
            top_k=int(generation["top_k"]),
            min_p=float(generation["min_p"]),
        )
        parsed = parse_llm_response(response.text)
        if parsed.failed or parsed.probability is None:
            raise RuntimeError(
                "DBSA sample failed to produce a mortality probability; "
                f"finish_reason={response.finish_reason}"
            )
        return float(parsed.probability)

    result = dbsa_risk_rank(
        chunk_names=chunk_names,
        prompt_builder=prompt_builder,
        sampler=sampler,
        n_full=args.n_full,
        n_perturbed=args.n_perturbed,
        n_permutations=args.n_permutations,
        max_workers=args.max_workers,
        seed=int(config["seed"] if args.seed is None else args.seed),
    )
    print("\n===== DBSA RESULTS =====")
    output = {
        "patient_id": key[0], "admission_id": key[1],
        "outcome": int(record["label"]),
        "model": os.environ["LLM_MODEL"],
        "chunk_set": args.chunk_set,
        "chunk_config_sha256": chunk_digest,
        "chunks_definition": chunks,
        "full_prompt_sha256": full_digest,
        "perturbed_prompt_sha256": checksums,
        "exploratory_unverified_events": True,
        "n_full": args.n_full,
        "n_perturbed": args.n_perturbed,
        "n_permutations": args.n_permutations,
        "ranked_chunks": list(result.ranked_chunks),
        "chunks": {},
    }
    for rank, chunk in enumerate(result.ranked_chunks, start=1):
        item = result.chunks[chunk]
        output["chunks"][chunk] = {
            "energy_distance": float(item.energy_distance),
            "p_value": float(item.p_value),
            "full_samples": list(item.full_samples),
            "perturbed_samples": list(item.perturbed_samples),
        }
        print(f"{rank:2d}. {chunk:32s} E={item.energy_distance:.4f} p={item.p_value:.3f}")

    target = Path("tmp/clinical_domains") / f"dbsa_single_{args.chunk_set}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Results saved to:", target)
    print("Single-admission configurable DBSA: PASS")
    print("WARNING: Retrospective diagnoses and prescriptions; not prospective evidence.")


if __name__ == "__main__":
    main()
