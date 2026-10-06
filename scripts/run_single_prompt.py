from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path

import yaml

from clinical_llm_conformal.data import (
    LABTEST_FEATURES,
    build_patient_records,
    load_clinicrealm_ehr,
)
from clinical_llm_conformal.llm import OpenAICompatibleLLM
from clinical_llm_conformal.parsing import parse_llm_response
from clinical_llm_conformal.prompt import build_clinicrealm_prompt

CONFIG_PATH = Path("configs/clinicrealm_reference.yaml")

config = yaml.safe_load(CONFIG_PATH.read_text())

EHR_PATH = Path(
    config["data"]["test_ehr_path"]
)


def main() -> None:
    base_url = os.environ["LLM_BASE_URL"]
    model = os.environ["LLM_MODEL"]
    api_key = os.environ.get("LLM_API_KEY", "dummy")

    ehr = load_clinicrealm_ehr(EHR_PATH)
    patients = build_patient_records(ehr)

    # Use the richest longitudinal example for the smoke test.
    key = max(
        patients,
        key=lambda k: len(patients[k]["day_cols"]),
    )

    patient = patients[key]

    system_prompt, user_prompt = build_clinicrealm_prompt(
        patient_row=patient["row"],
        day_cols=patient["day_cols"],
        record_times=patient["record_times"],
        features=LABTEST_FEATURES,
        task="mortality",
        n_shot=0,
        unit=True,
        reference_range=True,
        format_type="list",
    )

    print("Patient/admission:", key)
    print("Ground truth:", patient["label"])
    print("Visits:", len(patient["day_cols"]))
    print("Model:", model)

    llm = OpenAICompatibleLLM(
        model=model,
        base_url=base_url,
        api_key=api_key,
    )

    result = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    print("\n===== RAW MODEL RESPONSE =====\n")
    print(result.text)

    prediction = parse_llm_response(result.text)

    print("\n===== PARSED RESULT =====")
    print("Probability:", prediction.probability)
    print("Failed:", prediction.failed)
    print("Ground truth:", patient["label"])

    print("\n===== TOKEN USAGE =====")
    print("Prompt:", result.prompt_tokens)
    print("Completion:", result.completion_tokens)


if __name__ == "__main__":
    main()