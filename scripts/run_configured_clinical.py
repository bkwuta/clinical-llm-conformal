"""Run prompt validation or optional local LLM inference from a cohort manifest.

Both sources preserve their own temporal preprocessing conventions. The generic
path does not import retrospective ICD diagnoses/prescriptions into prompts.
Run from the repository root with ``uv run python scripts/run_configured_clinical.py
--config configs/generic_clinical.yaml``. Default is prompt-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from clinical_llm_conformal.cohort import load_cohort
from clinical_llm_conformal.data import LABTEST_FEATURES, load_clinicrealm_ehr
from clinical_llm_conformal.extended_prompt import (
    EXTENDED_LAB_FEATURES,
    SUPPORTED_DOMAINS,
    UNVERIFIED_DOMAINS,
    build_extended_clinical_prompt,
)
from clinical_llm_conformal.feature_catalog import load_feature_catalog
from clinical_llm_conformal.generic_prompt import build_generic_clinical_prompt
from clinical_llm_conformal.llm import OpenAICompatibleLLM
from clinical_llm_conformal.measurement import check_cohort_coverage, load_measurements
from clinical_llm_conformal.measurement_adapters import clinicrealm_records, generic_records
from clinical_llm_conformal.parsing import parse_llm_response

COLUMNS = [
    "subject_id", "hadm_id", "feature_set", "measurement_provider",
    "prompt_provider", "selected_features", "label", "n_visits", "status",
    "prompt_chars", "prompt_sha256", "model", "probability", "parse_failed",
    "finish_reason", "prompt_tokens", "completion_tokens", "error",
]


def _load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Expected CSV or Parquet: {path}")


def _ensure_subset(table: pd.DataFrame, selected: set[tuple[int, int]], label: str) -> None:
    if not {"subject_id", "hadm_id"}.issubset(table.columns):
        raise ValueError(f"{label} missing subject_id/hadm_id")
    ids = table[["subject_id", "hadm_id"]].copy()
    for col in ids:
        vals = pd.to_numeric(ids[col], errors="raise")
        if vals.isna().any() or (vals % 1 != 0).any():
            raise ValueError(f"{label}: invalid {col}")
        ids[col] = vals.astype("int64")
    extra = set(map(tuple, ids.itertuples(index=False, name=None))) - selected
    if extra:
        raise ValueError(f"{label} has {len(extra)} admission(s) outside the cohort; example: {next(iter(extra))}")


def _validate_config(config: dict) -> tuple[str, str]:
    provider = config["measurements"]["provider"]
    prompt_provider = config["prompt"]["provider"]
    if provider not in {"generic", "clinicrealm"}:
        raise ValueError("measurements.provider must be generic or clinicrealm")
    if prompt_provider != provider:
        raise ValueError("For this version prompt.provider must match measurements.provider")
    if provider == "generic":
        if not config.get("features", {}).get("selected"):
            raise ValueError("Generic mode requires features.selected")
    else:
        if not config.get("feature_sets"):
            raise ValueError("ClinicRealm mode requires feature_sets")
        for name, spec in config["feature_sets"].items():
            domains = spec["domains"]
            unknown = set(domains) - SUPPORTED_DOMAINS
            if unknown:
                raise ValueError(f"{name}: unknown domains {sorted(unknown)}")
            if set(domains) & UNVERIFIED_DOMAINS:
                raise ValueError(
                    "Stage 6E does not include retrospective diagnoses/prescriptions. "
                    "Use the existing exploratory run_clinical_domains.py instead."
                )
    return provider, prompt_provider


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--infer", action="store_true")
    cli.add_argument("--limit", type=int, default=None, help="First N admissions, sorted by IDs; useful for a smoke test")
    cli.add_argument("--output", type=Path, default=Path("tmp/configured_clinical/results.csv"))
    args = cli.parse_args()
    if args.limit is not None and args.limit < 1:
        cli.error("--limit must be positive")
    load_dotenv()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    provider, prompt_provider = _validate_config(config)
    cohort = load_cohort(config["cohort"]["path"])
    if args.limit is not None:
        cohort = cohort.head(args.limit).copy()
    selected = set(map(tuple, cohort[["subject_id", "hadm_id"]].itertuples(index=False, name=None)))
    prompt_opts = config["prompt"]

    if provider == "generic":
        # Filter before validating coverage so a --limit smoke test does not
        # mistake measurements for other selected admissions as unexpected.
        measurements = load_measurements(config["measurements"]["path"])
        if args.limit is None:
            check_cohort_coverage(measurements, cohort)  # Reject records outside cohort.
        measurements = measurements.loc[
            pd.MultiIndex.from_frame(measurements[["subject_id", "hadm_id"]]).isin(
                pd.MultiIndex.from_frame(cohort[["subject_id", "hadm_id"]])
            )
        ].copy()
        coverage = check_cohort_coverage(measurements, cohort)
        print("Generic measurement coverage:", coverage)
        demographics = None
        if config["measurements"].get("demographics_path"):
            demographics = _load_table(config["measurements"]["demographics_path"])
            _ensure_subset(demographics, set(map(tuple, load_cohort(config["cohort"]["path"])[["subject_id", "hadm_id"]].itertuples(index=False, name=None))), "Demographics")
            demographics = demographics.loc[
                pd.MultiIndex.from_frame(demographics[["subject_id", "hadm_id"]]).isin(
                    pd.MultiIndex.from_frame(cohort[["subject_id", "hadm_id"]])
                )
            ].copy()
        records = generic_records(
            measurements, demographics=demographics,
            labels=cohort[["subject_id", "hadm_id", "label"]] if "label" in cohort else None,
        )
        catalog = load_feature_catalog(config["features"]["catalog"])
        feature_sets = config["features"].get("sets") or {"generic": config["features"]["selected"]}
        for name, features in feature_sets.items():
            if not features or len(features) != len(set(features)):
                raise ValueError(f"{name}: empty or duplicate selected features")
            missing = set(features) - set(catalog)
            if missing:
                raise ValueError(f"{name}: features missing from catalog: {sorted(missing)}")
    else:
        ehr = load_clinicrealm_ehr(Path(config["measurements"]["path"]))
        all_features = LABTEST_FEATURES + EXTENDED_LAB_FEATURES
        if not all(f in ehr.columns for f in all_features):
            # Original 17-only EHR inputs must continue to work.
            all_features = LABTEST_FEATURES
        records = clinicrealm_records(ehr, features=all_features)
        extra = set(records) - set(map(tuple, load_cohort(config["cohort"]["path"])[["subject_id", "hadm_id"]].itertuples(index=False, name=None)))
        if extra:
            raise ValueError(f"ClinicRealm EHR contains {len(extra)} admissions outside manifest")
        feature_sets = {name: spec["domains"] for name, spec in config["feature_sets"].items()}
        if any("labs" in domains for domains in feature_sets.values()) and len(all_features) == len(LABTEST_FEATURES):
            raise ValueError("Configured labs but the ClinicRealm EHR does not contain the six extended lab columns")

    llm = None
    if args.infer:
        llm = OpenAICompatibleLLM(
            model=os.environ["LLM_MODEL"], base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ.get("LLM_API_KEY", "local"),
        )
    rows = []
    for entry in cohort.itertuples(index=False):
        key = (int(entry.subject_id), int(entry.hadm_id))
        patient = records.get(key)
        for name, selected_values in feature_sets.items():
            row = dict.fromkeys(COLUMNS, "")
            row.update(
                subject_id=key[0], hadm_id=key[1], feature_set=name,
                measurement_provider=provider, prompt_provider=prompt_provider,
                selected_features=";".join(selected_values),
                label="" if not hasattr(entry, "label") or pd.isna(entry.label) else int(entry.label),
                n_visits=0 if patient is None else len(patient["day_cols"]),
                status="no_measurements" if patient is None else "prompt_validated",
            )
            if patient is None:
                rows.append(row)
                print(f"{key} {name}: no measurements (retained in results)")
                continue
            record = {
                "patient_id": key[0], "admission_id": key[1],
                "measurements": patient,
                "demographics": {"age": patient["row"].get("Age"), "sex": patient["row"].get("Sex")},
                "label": row["label"], "diagnoses": [], "medications": [],
            }
            if provider == "generic":
                system, user = build_generic_clinical_prompt(
                    clinical_record=record, catalog=catalog, features=selected_values,
                    task=prompt_opts.get("task", "mortality"),
                    n_shot=int(prompt_opts.get("n_shot", 0)),
                    include_units=bool(prompt_opts.get("unit", True)),
                    include_reference_ranges=bool(prompt_opts.get("reference_range", True)),
                )
            else:
                system, user = build_extended_clinical_prompt(
                    clinical_record=record, domains=selected_values,
                    task=prompt_opts.get("task", "mortality"),
                    n_shot=int(prompt_opts.get("n_shot", 0)),
                    unit=bool(prompt_opts.get("unit", True)),
                    reference_range=bool(prompt_opts.get("reference_range", True)),
                    format_type=prompt_opts.get("format_type", "list"),
                    allow_unverified_events=False,
                )
            digest = hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()
            row.update(prompt_chars=len(user), prompt_sha256=digest)
            if llm is not None:
                gen = config.get("generation", {})
                try:
                    response = llm.generate(
                        system_prompt=system, user_prompt=user,
                        temperature=float(gen.get("temperature", 0.0)),
                        max_tokens=int(gen.get("max_tokens", 2048)),
                        enable_thinking=bool(gen.get("enable_thinking", False)),
                        reasoning_effort=gen.get("reasoning_effort", "none"),
                        top_k=int(gen["top_k"]) if gen.get("top_k") is not None else None,
                        min_p=float(gen["min_p"]) if gen.get("min_p") is not None else None,
                    )
                    parsed = parse_llm_response(response.text)
                    row["probability"] = "" if parsed.probability is None else parsed.probability
                    row["parse_failed"] = bool(parsed.failed)
                    row["finish_reason"] = getattr(response, "finish_reason", "") or ""
                    row["prompt_tokens"] = response.prompt_tokens or ""
                    row["completion_tokens"] = response.completion_tokens or ""
                    row["status"] = "parse_failed" if parsed.failed else "inferred"
                except Exception as exc:
                    row["status"] = "inference_error"
                    row["parse_failed"] = True
                    row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(f"{key} {name}: {row['status']} chars={len(user)} sha256={digest[:12]}")

    if not rows:
        raise ValueError("No experiments were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.output}")
    if not args.infer:
        print("Prompt-only mode: no LLM requests were made.")
    print("Prediction-time eligibility is not established by the adapters.")


if __name__ == "__main__":
    main()
