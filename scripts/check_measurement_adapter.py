"""Small end-to-end shape test for the source-independent measurement adapter."""
from __future__ import annotations

import pandas as pd

from clinical_llm_conformal.measurement_adapters import generic_records

measurements = pd.DataFrame([
    {"subject_id": 1, "hadm_id": 2, "timestamp": "2133-09-18 08:00", "feature": "Heart Rate", "value": 84, "unit": "bpm"},
    {"subject_id": 1, "hadm_id": 2, "timestamp": "2133-09-18 08:00", "feature": "Creatinine", "value": 1.2, "unit": "mg/dL"},
    {"subject_id": 1, "hadm_id": 2, "timestamp": "2133-09-19 08:00", "feature": "Heart Rate", "value": 92, "unit": "bpm"},
])
demographics = pd.DataFrame([{"subject_id": 1, "hadm_id": 2, "Age": 73, "Sex": 0}])
labels = pd.DataFrame([{"subject_id": 1, "hadm_id": 2, "label": 1}])
record = generic_records(measurements, demographics=demographics, labels=labels)[(1, 2)]
assert len(record["day_cols"]) == 2
assert record["row"].get((0, "Heart Rate")) == 84
assert record["row"].get((1, "Heart Rate")) == 92
assert record["row"].get((0, "Creatinine")) == 1.2
assert pd.isna(record["row"].get((1, "Creatinine"), float("nan")))
assert record["row"]["Age"] == 73
assert record["label"] == 1
print("Stage 6C generic measurement adapter: PASS")
