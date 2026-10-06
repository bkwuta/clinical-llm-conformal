from pathlib import Path

import yaml

from clinical_llm_conformal.data import (
    LABTEST_FEATURES,
    build_patient_records,
    load_clinicrealm_ehr,
)
from clinical_llm_conformal.prompt import (
    build_clinicrealm_prompt,
    load_prompt_assets,
)

CONFIG_PATH = Path("configs/clinicrealm_reference.yaml")

config = yaml.safe_load(CONFIG_PATH.read_text())

EHR_PATH = Path(
    config["data"]["test_ehr_path"]
)


def main() -> None:
    ehr = load_clinicrealm_ehr(EHR_PATH)

    print("Loaded EHR:", ehr.shape)

    print("Sex values:", ehr["Sex"].unique().tolist())

    patients = build_patient_records(ehr)

    print("Patient/admission records:", len(patients))

    key = max(
        patients,
        key=lambda k: len(patients[k]["day_cols"]),
        )
    patient = patients[key]

    print("Selected admission:", key)
    print("Ground-truth outcome:", patient["label"])
    print("Time points:", patient["day_cols"])
    print("Record times:", patient["record_times"])

    assets = load_prompt_assets()

    print(
        "Available tasks:",
        list(assets["TASK_DESCRIPTION"]),
    )

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

    print("\n===== SYSTEM PROMPT =====\n")
    print(system_prompt)

    print("\n===== USER PROMPT =====\n")
    print(user_prompt)

    print("\n===== SUMMARY =====")
    print("System prompt chars:", len(system_prompt))
    print("User prompt chars:", len(user_prompt))


if __name__ == "__main__":
    main()