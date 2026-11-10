"""Compare configurable clinical-domain prompts for the same MIMIC admissions.

Run from the clinical-llm-conformal repository root:
  uv run python scripts/run_clinical_domains.py            # prompt validation only
  uv run python scripts/run_clinical_domains.py --infer    # perform LLM inference

Outputs are local, potentially sensitive research artifacts: do not commit them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from datetime import datetime, timezone
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
from clinical_llm_conformal.extended_prompt import (
    EXTENDED_LAB_FEATURES,
    SUPPORTED_DOMAINS,
    UNVERIFIED_DOMAINS,
    build_extended_clinical_prompt,
)
from clinical_llm_conformal.llm import OpenAICompatibleLLM
from clinical_llm_conformal.parsing import parse_llm_response

CONFIG_PATH = Path("configs/clinical_domains.yaml")
OUTPUT_PATH = Path("tmp/clinical_domains/experiment_results.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--infer", action="store_true",
        help="Call the configured LLM. By default, only validate prompts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    config = yaml.safe_load(args.config.read_text())

    sets = config["feature_sets"]
    unverified_allowed = bool(
        config.get("experiment", {}).get("allow_unverified_events", False)
    )
    for name, spec in sets.items():
        domains = spec["domains"]
        unknown = set(domains) - SUPPORTED_DOMAINS
        if unknown:
            raise ValueError(f"{name}: unknown domains {sorted(unknown)}")
        if set(domains) & UNVERIFIED_DOMAINS and not unverified_allowed:
            raise ValueError(
                f"{name}: unverified diagnosis/medication domains require "
                "experiment.allow_unverified_events: true"
            )

    ehr = load_clinicrealm_ehr(Path(config["data"]["extended_ehr_path"]))
    patients = build_patient_records(
        ehr, features=LABTEST_FEATURES + EXTENDED_LAB_FEATURES
    )
    diagnoses = load_diagnoses(Path(config["data"]["diagnoses_path"]))
    prescriptions = load_prescriptions(Path(config["data"]["prescriptions_path"]))
    opts = config["prompt"]

    llm = None
    if args.infer:
        llm = OpenAICompatibleLLM(
            model=os.environ["LLM_MODEL"],
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ.get("LLM_API_KEY", "local"),
        )

    rows = []
    for entry in config["patients"]:
        key = (int(entry["patient_id"]), int(entry["admission_id"]))
        if key not in patients:
            raise KeyError(f"Admission {key} is absent from EHR cohort")
        record = build_clinical_record(
            key=key,
            patient=patients[key],
            diagnoses_df=diagnoses,
            prescriptions_df=prescriptions,
        )
        print(f"\nAdmission {key}: visits={len(record['measurements']['day_cols'])}")

        for name, spec in sets.items():
            domains = list(spec["domains"])
            exploratory = bool(set(domains) & UNVERIFIED_DOMAINS)
            system_prompt, user_prompt = build_extended_clinical_prompt(
                clinical_record=record,
                domains=domains,
                task=opts["task"],
                n_shot=int(opts["n_shot"]),
                unit=bool(opts["unit"]),
                reference_range=bool(opts["reference_range"]),
                format_type=opts["format_type"],
                allow_unverified_events=unverified_allowed,
                max_medications=int(opts["max_medications"]),
            )
            digest = hashlib.sha256(
                (system_prompt + "\n" + user_prompt).encode("utf-8")
            ).hexdigest()
            row = {
                "patient_id": key[0],
                "admission_id": key[1],
                "feature_set": name,
                "domains": ";".join(domains),
                "exploratory_unverified_events": exploratory,
                "outcome": record["label"],
                "visits": len(record["measurements"]["day_cols"]),
                "prompt_chars": len(user_prompt),
                "prompt_sha256": digest,
                "model": os.environ.get("LLM_MODEL", "") if args.infer else "",
                "probability": "",
                "parse_failed": "",
                "error": "",
            }
            if args.infer:
                generation = config["generation"]
                
                try:
                    response = llm.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=float(generation["temperature"]),
                        max_tokens=int(generation["max_tokens"]),
                        enable_thinking=bool(generation["enable_thinking"]),
                        reasoning_effort=generation["reasoning_effort"],
                        top_k=int(generation["top_k"]),
                        min_p=float(generation["min_p"]),
                    )
                    parsed = parse_llm_response(response.text)
                    row["probability"] = (
                        parsed.probability if parsed.probability is not None else ""
                    )
                    row["parse_failed"] = bool(parsed.failed)
                except Exception as exc:
                    row["parse_failed"] = True
                    row["error"] = f"{type(exc).__name__}: {exc}"

            status = (
                f"p={row['probability']} failed={row['parse_failed']}"
                if args.infer else "prompt validated"
            )
            print(
                f"  {name:20s} chars={len(user_prompt):5d} "
                f"exploratory={exploratory} sha256={digest[:12]} {status}"
            )
            rows.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {OUTPUT_PATH}")
    if not args.infer:
        print("No LLM requests were made. Use --infer to run inference.")
    if any(row["exploratory_unverified_events"] for row in rows):
        print("WARNING: diagnosis/prescription conditions are exploratory only; cutoff availability unverified.")


if __name__ == "__main__":
    main()
