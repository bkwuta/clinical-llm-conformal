from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import yaml
from dotenv import load_dotenv

from clinical_llm_conformal.data import (
    LABTEST_FEATURES,
    build_patient_records,
    load_clinicrealm_ehr,
)
from clinical_llm_conformal.dbsa import dbsa_risk_rank
from clinical_llm_conformal.llm import OpenAICompatibleLLM
from clinical_llm_conformal.parsing import parse_llm_response
from clinical_llm_conformal.prompt import (
    CHUNK_DEFINITIONS,
    build_clinicrealm_prompt,
)

REFERENCE_CONFIG_PATH = Path("configs/clinicrealm_reference.yaml")
DBSA_CONFIG_PATH = Path("configs/dbsa.yaml")

EXTENDED_LAB_FEATURES = [
    "Creatinine",
    "BUN",
    "Lactate",
    "WBC",
    "Hemoglobin",
    "Platelets",
]

FEATURE_SETS = {
    "clinicrealm_17": list(LABTEST_FEATURES),
    "clinicrealm_23": list(LABTEST_FEATURES) + EXTENDED_LAB_FEATURES,
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def active_chunks_for(features: list[str]) -> list[str]:
    selected = set(features)
    return [
        name
        for name, members in CHUNK_DEFINITIONS.items()
        if name == "Demographics" or bool(selected.intersection(members))
    ]


def run_patient_dbsa(
    *,
    key: tuple[int, int],
    patient: dict,
    llm: OpenAICompatibleLLM,
    config: dict,
    feature_set_name: str,
    features: list[str],
) -> None:
    active_chunks = active_chunks_for(features)
    assigned_features = {
        feature
        for chunk in active_chunks
        for feature in CHUNK_DEFINITIONS[chunk]
    }
    unassigned = set(features) - assigned_features
    if unassigned:
        raise ValueError(
            f"Features missing DBSA chunk assignments: {sorted(unassigned)}"
        )

    print("\n" + "=" * 70)
    print(f"FEATURE SET: {feature_set_name} ({len(features)} features)")
    print("Patient/admission:", key)
    print("Outcome:", patient["label"])
    print("Visits:", len(patient["day_cols"]))
    print(f"Active DBSA chunks ({len(active_chunks)}): {active_chunks}")
    print("=" * 70)

    def prompt_builder(
        retained_chunks: Sequence[str],
    ) -> tuple[str, str]:
        return build_clinicrealm_prompt(
            patient_row=patient["row"],
            day_cols=patient["day_cols"],
            record_times=patient["record_times"],
            features=features,
            task=config["prompt"]["task"],
            n_shot=int(config["prompt"]["n_shot"]),
            unit=bool(config["prompt"]["unit"]),
            reference_range=bool(config["prompt"]["reference_range"]),
            format_type=config["prompt"]["format_type"],
            retained_chunks=list(retained_chunks),
        )

    def sampler(system_prompt: str, user_prompt: str) -> float:
        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(config["generation"]["temperature"]),
            max_tokens=int(config["generation"]["max_tokens"]),
            enable_thinking=bool(config["generation"]["enable_thinking"]),
            reasoning_effort=config["generation"]["reasoning_effort"],
            top_k=int(config["generation"]["top_k"]),
            min_p=float(config["generation"]["min_p"]),
        )
        parsed = parse_llm_response(response.text)
        if parsed.failed or parsed.probability is None:
            raise RuntimeError(
                "DBSA sample failed to produce a valid mortality probability."
            )
        return parsed.probability

    result = dbsa_risk_rank(
        chunk_names=active_chunks,
        prompt_builder=prompt_builder,
        sampler=sampler,
        n_full=int(config["dbsa"]["n_full"]),
        n_perturbed=int(config["dbsa"]["n_perturbed"]),
        n_permutations=int(config["dbsa"]["n_permutations"]),
        max_workers=int(config["dbsa"]["max_workers"]),
        seed=int(config["seed"]),
    )

    print("\n===== DBSA RESULTS =====")
    for chunk in result.ranked_chunks:
        item = result.chunks[chunk]
        print(
            f"{chunk:30s} "
            f"E={item.energy_distance:.4f} "
            f"p={item.p_value:.3f} "
            f"full={list(item.full_samples)} "
            f"pert={list(item.perturbed_samples)}"
        )

    print("\n===== RANKING =====")
    for rank, chunk in enumerate(result.ranked_chunks, start=1):
        item = result.chunks[chunk]
        print(
            f"{rank:2d}. {chunk:30s} "
            f"E={item.energy_distance:.4f} p={item.p_value:.3f}"
        )


def main() -> None:
    load_dotenv()
    reference_config = load_yaml(REFERENCE_CONFIG_PATH)
    config = load_yaml(DBSA_CONFIG_PATH)
    ehr_path = Path(reference_config["data"]["test_ehr_path"])
    ehr = load_clinicrealm_ehr(ehr_path)

    # Build every record with all 23 variables; each DBSA run then selects
    # its own feature set when constructing the prompt.
    all_features = FEATURE_SETS["clinicrealm_23"]
    patients = build_patient_records(ehr, features=all_features)

    patient_cfgs = config.get("patients")
    if not patient_cfgs:
        raise ValueError(
            "configs/dbsa.yaml must contain at least one patient under 'patients'."
        )

    keys: list[tuple[int, int]] = []
    for patient_cfg in patient_cfgs:
        key = (
            int(patient_cfg["patient_id"]),
            int(patient_cfg["admission_id"]),
        )
        if key not in patients:
            raise KeyError(
                f"Patient/admission {key} not found in configured EHR cohort."
            )
        keys.append(key)

    llm = OpenAICompatibleLLM(
        model=os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ.get("LLM_API_KEY", "local"),
    )

    print(f"Running DBSA for {len(keys)} admission(s).")
    for index, key in enumerate(keys, start=1):
        print(f"\nPATIENT [{index}/{len(keys)}]")
        for feature_set_name, features in FEATURE_SETS.items():
            run_patient_dbsa(
                key=key,
                patient=patients[key],
                llm=llm,
                config=config,
                feature_set_name=feature_set_name,
                features=features,
            )


if __name__ == "__main__":
    main()