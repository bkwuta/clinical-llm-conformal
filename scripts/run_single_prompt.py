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

FEATURE_SETS = {
    "clinicrealm_17": LABTEST_FEATURES,

    "respiratory": [
        "Respiratory rate",
        "Fraction inspired oxygen",
        "Oxygen saturation",
    ],

    "hemodynamics": [
        "Heart Rate",
        "Diastolic blood pressure",
        "Mean blood pressure",
        "Systolic blood pressure",
    ],

    "neuro": [
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale verbal response",
        "Glascow coma scale total",
    ],
}

EXTENDED_LAB_FEATURES = [
    "Creatinine",
    "BUN",
    "Lactate",
    "WBC",
    "Hemoglobin",
    "Platelets",
]

FEATURE_SETS = {
    "clinicrealm_17": LABTEST_FEATURES,
    "clinicrealm_23": LABTEST_FEATURES + EXTENDED_LAB_FEATURES,
}

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
    patients = build_patient_records(ehr, features=LABTEST_FEATURES + EXTENDED_LAB_FEATURES,)

    # Use the richest longitudinal example for the smoke test.
    key = max(
        patients,
        key=lambda k: len(patients[k]["day_cols"]),
    )

    patient = patients[key]

    llm = OpenAICompatibleLLM(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    print("Patient/admission:", key)
    print("Ground truth:", patient["label"])
    print("Visits:", len(patient["day_cols"]))
    print("Model:", model)

    for feature_set_name, features in FEATURE_SETS.items():
        print("\n" + "=" * 70)
        print(
            f"FEATURE SET: {feature_set_name} "
            f"({len(features)} features)"
        )
        print("=" * 70)

        system_prompt, user_prompt = build_clinicrealm_prompt(
            patient_row=patient["row"],
            day_cols=patient["day_cols"],
            record_times=patient["record_times"],
            features=features,
            task="mortality",
            n_shot=0,
            unit=True,
            reference_range=True,
            format_type="list",
        )
        print("\n===== PROMPT DIAGNOSTICS =====")
        print("Feature set:", feature_set_name)
        print("Prompt characters:", len(user_prompt))

        for lab in EXTENDED_LAB_FEATURES:
            print(
                f"{lab:12s}: "
                f"{lab in user_prompt}"
                )

        result = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1024,
            enable_thinking=False,
            reasoning_effort="none",
            top_k=20,
            min_p=0.0,
        )

        prediction = parse_llm_response(
            result.text
        )

        print("Features:")
        for feature in features:
            print(f"  - {feature}")

        print("\nRaw response:")
        print(result.text)

        print(
            "\nMortality probability:",
            prediction.probability,
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