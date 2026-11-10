"""Stage 6D: verify a custom feature reaches a full task prompt without inference."""
from pathlib import Path

import pandas as pd

from clinical_llm_conformal.feature_catalog import load_feature_catalog
from clinical_llm_conformal.generic_prompt import build_generic_clinical_prompt
from clinical_llm_conformal.measurement_adapters import generic_records


def main() -> None:
    catalog = load_feature_catalog("configs/feature_catalog_example.yaml")
    records = generic_records(
        pd.DataFrame([
            (1001, 2001, "2133-09-18 08:00", "Heart Rate", 84.0),
            (1001, 2001, "2133-09-18 08:00", "Custom Biomarker", 1.2),
            (1001, 2001, "2133-09-19 08:00", "Heart Rate", 92.0),
            (1001, 2001, "2133-09-19 08:00", "Mental Status", "Alert"),
        ], columns=["subject_id", "hadm_id", "timestamp", "feature", "value"]),
        demographics=pd.DataFrame([{"subject_id": 1001, "hadm_id": 2001, "Age": 73, "Sex": 0}]),
        labels=pd.DataFrame([{"subject_id": 1001, "hadm_id": 2001, "label": 0}]),
    )
    patient = records[(1001, 2001)]
    record = {
        "measurements": patient,
        "demographics": {"age": patient["row"]["Age"], "sex": patient["row"]["Sex"]},
    }
    system, user = build_generic_clinical_prompt(
        clinical_record=record,
        catalog=catalog,
        features=["Heart Rate", "Custom Biomarker", "Mental Status"],
    )
    assert system
    assert "Custom Biomarker (Unit: ng/mL): [1.2, NaN]" in user
    assert "Mental Status: [unknown, Alert]" in user
    assert "Heart Rate (Unit: bpm): [84.0, 92.0]" in user
    assert "mortality" in user.lower() or "not surviving" in user.lower()
    print("Stage 6D generic clinical prompt: PASS")
    print("Selected features: Heart Rate, Custom Biomarker, Mental Status")
    print("User prompt characters:", len(user))


if __name__ == "__main__":
    main()
